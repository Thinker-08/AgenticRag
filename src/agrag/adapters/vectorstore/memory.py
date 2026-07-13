from __future__ import annotations

from typing import Sequence

import numpy as np

from ...contracts import ScoredChunk
from ...interfaces.types import SparseVector, VectorRecord
from .filters import matches


class MemoryVectorStore:
    def __init__(self) -> None:
        self._dense: dict[str, np.ndarray] = {}
        self._sparse: dict[str, SparseVector] = {}
        self._rec: dict[str, VectorRecord] = {}

    async def upsert(self, records: Sequence[VectorRecord]) -> None:
        for r in records:
            key = r.chunk.chunk_id
            self._dense[key] = np.asarray(r.dense, dtype=np.float32)
            self._sparse[key] = r.sparse or {}
            self._rec[key] = r

    def _sparse_score(self, q: SparseVector, d: SparseVector) -> float:
        if not q or not d:
            return 0.0
        return sum(w * d.get(t, 0.0) for t, w in q.items())

    async def search(
        self,
        query_dense: list[float],
        *,
        tenant_id: str,
        top_k: int = 100,
        query_sparse: SparseVector | None = None,
        filters: dict | None = None,
    ) -> list[ScoredChunk]:
        if not self._dense:
            return []
        q = np.asarray(query_dense, dtype=np.float32)
        qn = float(np.linalg.norm(q)) or 1.0
        scored: list[ScoredChunk] = []
        for key, vec in self._dense.items():
            rec = self._rec[key]
            if rec.chunk.tenant_id != tenant_id:
                continue
            if not matches(
                {
                    **rec.chunk.extra_metadata,
                    "page_no": rec.chunk.page_no,
                    "kind": rec.chunk.kind,
                    "doc_id": rec.chunk.doc_id,
                    "lang": rec.chunk.lang,
                },
                filters,
            ):
                continue
            dn = float(np.linalg.norm(vec)) or 1.0
            dense = float(np.dot(q, vec) / (qn * dn))
            sparse = self._sparse_score(query_sparse or {}, self._sparse.get(key, {}))
            scored.append(ScoredChunk(chunk=rec.chunk, score=dense + 0.001 * sparse))
        scored.sort(key=lambda s: s.score, reverse=True)
        top = scored[:top_k]
        for i, s in enumerate(top):
            s.dense_rank = i
        return top

    async def delete_doc(self, doc_id: str, tenant_id: str) -> None:
        drop = [
            k
            for k, r in self._rec.items()
            if r.chunk.doc_id == doc_id and r.chunk.tenant_id == tenant_id
        ]
        for k in drop:
            self._dense.pop(k, None)
            self._sparse.pop(k, None)
            self._rec.pop(k, None)

    async def count(self, tenant_id: str | None = None) -> int:
        if tenant_id is None:
            return len(self._rec)
        return sum(1 for r in self._rec.values() if r.chunk.tenant_id == tenant_id)
