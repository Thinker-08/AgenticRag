"""BM25 lexical index — the exact-token leg of hybrid retrieval (C5).
Rebuilds the per-tenant BM25 matrix on demand; fine at portfolio scale. Nails part numbers,
statute refs, tickers, and rare proper nouns that dense embeddings smear.
"""

from __future__ import annotations

import asyncio
import re
from typing import Sequence

from rank_bm25 import BM25Okapi

from ...contracts import Chunk, ScoredChunk
from ..vectorstore.filters import matches

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-_.]*")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Bm25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: dict[str, list[Chunk]] = {}
        self._models: dict[str, tuple[BM25Okapi, list[Chunk]]] = {}
        self._dirty: set[str] = set()

    async def add(self, chunks: Sequence[Chunk]) -> None:
        for c in chunks:
            bucket = self._chunks.setdefault(c.tenant_id, [])
            bucket[:] = [x for x in bucket if x.chunk_id != c.chunk_id]
            bucket.append(c)
            self._dirty.add(c.tenant_id)

    def _model(self, tenant_id: str):
        if tenant_id in self._dirty or tenant_id not in self._models:
            chunks = list(self._chunks.get(tenant_id, []))
            corpus = [_tokenize(c.linearized_text or c.text) for c in chunks]
            self._models[tenant_id] = (
                BM25Okapi(corpus, k1=self.k1, b=self.b) if corpus else None,
                chunks,
            )
            self._dirty.discard(tenant_id)
        return self._models[tenant_id]

    async def search(
        self, query: str, *, tenant_id: str, top_k: int = 100, filters: dict | None = None
    ) -> list[ScoredChunk]:
        model, chunks = await asyncio.to_thread(self._model, tenant_id)
        if not model or not chunks:
            return []
        scores = await asyncio.to_thread(model.get_scores, _tokenize(query))
        ranked = sorted(zip(chunks, scores), key=lambda p: p[1], reverse=True)
        out: list[ScoredChunk] = []
        for c, s in ranked:
            meta = {
                **c.extra_metadata,
                "page_no": c.page_no,
                "kind": c.kind,
                "doc_id": c.doc_id,
                "lang": c.lang,
            }
            if not matches(meta, filters):
                continue
            out.append(ScoredChunk(chunk=c, score=float(s), bm25_rank=len(out)))
            if len(out) >= top_k:
                break
        return out

    async def delete_doc(self, doc_id: str, tenant_id: str) -> None:
        bucket = self._chunks.get(tenant_id, [])
        bucket[:] = [c for c in bucket if c.doc_id != doc_id]
        self._dirty.add(tenant_id)
