"""Qdrant dense + sparse vector store (full mode). Tenant scoping is mandatory (C31)."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Sequence

from ...contracts import Chunk, ScoredChunk
from ...interfaces.types import SparseVector, VectorRecord

if TYPE_CHECKING:
    from ...config import VectorStoreConfig


class QdrantVectorStore:
    """VectorStore backed by Qdrant with a named "dense" (cosine) + "sparse" vector."""

    def __init__(self, host: str, collection: str, dim: int, cfg: "VectorStoreConfig") -> None:
        try:
            from qdrant_client import AsyncQdrantClient
        except ImportError as e:
            raise ImportError(
                "QdrantVectorStore needs the 'stores' extra: pip install -e '.[stores]'"
            ) from e
        self._client = AsyncQdrantClient(url=host)
        self._collection = collection
        self._dim = dim
        self._cfg = cfg
        self._ready = False
        self._lock = asyncio.Lock()

    async def upsert(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        await self._ensure_collection()
        from qdrant_client import models as qm

        points = []
        for rec in records:
            chunk = rec.chunk
            payload = chunk.model_dump()
            for key, val in chunk.extra_metadata.items():
                payload.setdefault(key, val)
            vectors: dict = {"dense": rec.dense}
            if rec.sparse:
                vectors["sparse"] = qm.SparseVector(
                    indices=list(rec.sparse.keys()), values=list(rec.sparse.values())
                )
            points.append(
                qm.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                    vector=vectors,
                    payload=payload,
                )
            )
        await self._client.upsert(collection_name=self._collection, points=points)

    async def search(
        self,
        query_dense: list[float],
        *,
        tenant_id: str,
        top_k: int = 100,
        query_sparse: SparseVector | None = None,
        filters: dict | None = None,
    ) -> list[ScoredChunk]:
        await self._ensure_collection()
        from qdrant_client import models as qm

        qfilter = self._build_filter(qm, tenant_id, filters)
        params = qm.SearchParams(hnsw_ef=self._cfg.ef_search)

        if query_sparse:
            sparse = qm.SparseVector(
                indices=list(query_sparse.keys()), values=list(query_sparse.values())
            )
            prefetch = [
                qm.Prefetch(
                    query=query_dense, using="dense", filter=qfilter, params=params, limit=top_k
                ),
                qm.Prefetch(query=sparse, using="sparse", filter=qfilter, limit=top_k),
            ]
            resp = await self._client.query_points(
                collection_name=self._collection,
                prefetch=prefetch,
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )
        else:
            resp = await self._client.query_points(
                collection_name=self._collection,
                query=query_dense,
                using="dense",
                query_filter=qfilter,
                search_params=params,
                limit=top_k,
                with_payload=True,
            )

        out: list[ScoredChunk] = []
        for i, hit in enumerate(resp.points):
            out.append(
                ScoredChunk(chunk=Chunk.model_validate(hit.payload), score=hit.score, dense_rank=i)
            )
        return out

    async def delete_doc(self, doc_id: str, tenant_id: str) -> None:
        await self._ensure_collection()
        from qdrant_client import models as qm

        flt = qm.Filter(
            must=[
                qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id)),
                qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
            ]
        )
        await self._client.delete(
            collection_name=self._collection, points_selector=qm.FilterSelector(filter=flt)
        )

    async def count(self, tenant_id: str | None = None) -> int:
        await self._ensure_collection()
        from qdrant_client import models as qm

        flt = None
        if tenant_id is not None:
            flt = qm.Filter(
                must=[qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id))]
            )
        resp = await self._client.count(
            collection_name=self._collection, count_filter=flt, exact=True
        )
        return resp.count

    async def _ensure_collection(self) -> None:
        if self._ready:
            return
        async with self._lock:
            if self._ready:
                return
            from qdrant_client import models as qm

            if not await self._client.collection_exists(self._collection):
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config={
                        "dense": qm.VectorParams(
                            size=self._dim,
                            distance=qm.Distance.COSINE,
                            hnsw_config=qm.HnswConfigDiff(
                                m=self._cfg.hnsw_m, ef_construct=self._cfg.ef_construction
                            ),
                            quantization_config=self._quantization(qm),
                        )
                    },
                    sparse_vectors_config={"sparse": qm.SparseVectorParams()},
                )
            self._ready = True

    def _quantization(self, qm):
        if self._cfg.quantize == "int8":
            return qm.ScalarQuantization(
                scalar=qm.ScalarQuantizationConfig(type=qm.ScalarType.INT8, always_ram=True)
            )
        if self._cfg.quantize == "binary":
            return qm.BinaryQuantization(binary=qm.BinaryQuantizationConfig(always_ram=True))
        return None

    def _build_filter(self, qm, tenant_id: str, filters: dict | None):
        must = [qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=tenant_id))]
        for key, cond in (filters or {}).items():
            must.append(self._condition(qm, key, cond))
        return qm.Filter(must=must)

    def _condition(self, qm, key: str, cond):
        if isinstance(cond, dict):
            if "$in" in cond:
                return qm.FieldCondition(key=key, match=qm.MatchAny(any=list(cond["$in"])))
            rng = {k[1:]: v for k, v in cond.items() if k in ("$gte", "$lte", "$gt", "$lt")}
            if rng:
                return qm.FieldCondition(key=key, range=qm.Range(**rng))
            if "$eq" in cond:
                return qm.FieldCondition(key=key, match=qm.MatchValue(value=cond["$eq"]))
        if isinstance(cond, (list, tuple, set)):
            return qm.FieldCondition(key=key, match=qm.MatchAny(any=list(cond)))
        return qm.FieldCondition(key=key, match=qm.MatchValue(value=cond))
