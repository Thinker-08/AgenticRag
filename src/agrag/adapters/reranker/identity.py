"""Lexical-overlap reranker — the local stand-in for the BGE cross-encoder (C4).

A real cross-encoder scores the (query, chunk) pair jointly; here we approximate joint relevance
with token-overlap + fused-score priors so the two-stage funnel and reorder logic are exercised.
"""

from __future__ import annotations

import re
from typing import Sequence

from ...contracts import Budget, ScoredChunk

_TOKEN = re.compile(r"[a-z0-9]+")


def _toks(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


class IdentityReranker:
    async def rerank(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        *,
        top_k: int = 8,
        budget: Budget | None = None,
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        q = _toks(query)
        out: list[ScoredChunk] = []
        for sc in candidates:
            c = _toks(sc.chunk.text)
            overlap = len(q & c) / (len(q) or 1)
            sc.rerank_score = overlap + 0.05 * sc.score
            sc.score = sc.rerank_score
            out.append(sc)
        out.sort(key=lambda s: s.rerank_score or 0.0, reverse=True)
        return out[:top_k]
