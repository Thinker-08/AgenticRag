from __future__ import annotations

import re
from typing import Sequence

from ...contracts import Budget, ScoredChunk

_TOKEN = re.compile(r"[a-z0-9]+")


def toks(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


class IdentityReranker:
    async def rerank(self, query: str, candidates: Sequence[ScoredChunk], *, top_k: int = 8, budget: Budget | None = None) -> list[ScoredChunk]:
        if not candidates:
            return []

        q = toks(query)
        out: list[ScoredChunk] = []
        for sc in candidates:
            c = toks(sc.chunk.text)
            overlap = len(q & c) / (len(q) or 1)
            sc.rerank_score = overlap + 0.05 * sc.score
            sc.score = sc.rerank_score
            out.append(sc)

        out.sort(key=lambda s: s.rerank_score or 0.0, reverse=True)
        return out[:top_k]
