"""The retrieval funnel (04): over-fetch dense + BM25 → RRF fuse → dedupe → cross-encoder rerank.

Over-fetch cheaply and broadly, then narrow expensively and precisely. First-stage similarity is a
candidate generator, never the final ranking (C4). Strategy selects which legs run and how they weight.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

from ..config import RetrievalConfig
from ..contracts import Budget, ScoredChunk, Strategy
from ..interfaces import Cache, EmbeddingModel, LexicalIndex, Reranker, VectorStore
from ..reliability import CircuitBreaker
from .dedupe import dedupe
from .rrf import rrf_fuse

_LEGS = {
    Strategy.SEMANTIC: (True, False),
    Strategy.BM25: (False, True),
    Strategy.HYBRID: (True, True),
    Strategy.TABLE: (True, True),
    Strategy.METADATA_FILTER: (True, True),
    Strategy.DOC_SUMMARY: (True, True),
    Strategy.CODE: (True, True),
    Strategy.GRAPH: (True, True),
}


class EmbeddingContractError(RuntimeError):
    """Query embedder and index vectors come from different (model, version) spaces (C3).
    A cosine across spaces is meaningless noise — fail loud, force the blue/green re-embed."""


def reorder_for_context(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    """Place the best at the head, 2nd at the tail, 3rd at position 2… (lost-in-the-middle, 04 §8)."""
    head: list[ScoredChunk] = []
    tail: list[ScoredChunk] = []
    for i, sc in enumerate(chunks):
        (head if i % 2 == 0 else tail).append(sc)
    return head + list(reversed(tail))


class HybridRetriever:
    def __init__(
        self,
        *,
        embedding: EmbeddingModel,
        vectorstore: VectorStore,
        lexical: LexicalIndex,
        reranker: Reranker,
        cfg: RetrievalConfig,
        cache: Cache | None = None,
    ) -> None:
        self.embedding = embedding
        self.vectorstore = vectorstore
        self.lexical = lexical
        self.reranker = reranker
        self.cfg = cfg
        self.cache = cache
        self._dense_breaker = CircuitBreaker(failures=4, window_s=30, cooldown_s=15)
        self._rerank_breaker = CircuitBreaker(failures=4, window_s=30, cooldown_s=15)

    async def _cache_key(
        self, query: str, tenant_id: str, strategy: Strategy, k: int, filters: dict
    ) -> str:
        version = await self.vectorstore.count(tenant_id)
        payload = json.dumps(
            [query, tenant_id, strategy.value, k, filters, version], sort_keys=True, default=str
        )
        return "retr:" + hashlib.blake2b(payload.encode(), digest_size=12).hexdigest()

    async def retrieve(
        self,
        query: str,
        *,
        tenant_id: str,
        strategy: Strategy = Strategy.HYBRID,
        k: int = 100,
        filters: dict | None = None,
        budget: Budget | None = None,
    ) -> list[ScoredChunk]:
        if not tenant_id:
            from ..security import SecurityError

            raise SecurityError("retrieval attempted with empty tenant scope")
        if budget is not None and budget.exceeded():
            return []
        use_dense, use_bm25 = _LEGS.get(strategy, (True, True))
        filters = dict(filters or {})
        if strategy == Strategy.TABLE:
            filters.setdefault("kind", "table")

        key = None
        if self.cache is not None:
            key = await self._cache_key(query, tenant_id, strategy, k, filters)
            hit = await self.cache.get(key)
            if hit is not None:
                return [ScoredChunk.model_validate(sc) for sc in hit]

        over_fetch = self.cfg.over_fetch
        if use_dense and not self._dense_breaker.allow():
            use_dense = False
        dense_task = self._dense(query, tenant_id, over_fetch, filters) if use_dense else _empty()
        bm25_task = (
            self.lexical.search(query, tenant_id=tenant_id, top_k=over_fetch, filters=filters)
            if use_bm25
            else _empty()
        )
        dense_res, bm25_res = await asyncio.gather(dense_task, bm25_task)

        bm25_w = self.cfg.bm25_weight * (2.0 if strategy == Strategy.BM25 else 1.0)
        fused = rrf_fuse(
            [(dense_res, self.cfg.dense_weight), (bm25_res, bm25_w)],
            k=self.cfg.rrf_k,
        )
        if not fused and (dense_res or bm25_res):
            fused = dense_res or bm25_res
        deduped = dedupe(fused, threshold=self.cfg.dedupe_threshold)
        top_k = min(k, self.cfg.top_k) if k else self.cfg.top_k
        reranked = await self._rerank(query, deduped, top_k, budget)
        if key is not None and reranked:
            await self.cache.set(key, [sc.model_dump(mode="json") for sc in reranked], ttl_s=300)
        return reranked

    async def _rerank(self, query, deduped, top_k, budget) -> list[ScoredChunk]:
        if not self._rerank_breaker.allow():
            return deduped[:top_k]
        try:
            out = await self.reranker.rerank(query, deduped, top_k=top_k, budget=budget)
            self._rerank_breaker.record_success()
            return out
        except Exception:
            self._rerank_breaker.record_failure()
            return deduped[:top_k]

    async def _dense(
        self, query: str, tenant_id: str, top_k: int, filters: dict | None
    ) -> list[ScoredChunk]:
        try:
            emb = await asyncio.to_thread(self.embedding.encode_queries, [query])
            sparse = emb.sparse[0] if emb.sparse else None
            results = await self.vectorstore.search(
                emb.dense[0], tenant_id=tenant_id, top_k=top_k, query_sparse=sparse, filters=filters
            )
            self._dense_breaker.record_success()
        except EmbeddingContractError:
            raise
        except Exception:
            self._dense_breaker.record_failure()
            return []
        for sc in results:
            c = sc.chunk
            if c.embedding_model and (
                c.embedding_model != self.embedding.model
                or c.embedding_version != self.embedding.version
            ):
                raise EmbeddingContractError(
                    f"index vectors are ({c.embedding_model}, {c.embedding_version}) but the query "
                    f"embedder is ({self.embedding.model}, {self.embedding.version}); "
                    "full re-embed required — refusing to mix spaces"
                )
        return results


async def _empty() -> list[ScoredChunk]:
    return []
