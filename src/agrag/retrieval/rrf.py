from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from ..contracts import ScoredChunk


def rrf_fuse(
    ranked_lists: Sequence[tuple[Sequence[ScoredChunk], float]],
    *,
    k: int = 60,
) -> list[ScoredChunk]:
    fused: dict[str, float] = defaultdict(float)
    keep: dict[str, ScoredChunk] = {}
    for lst, weight in ranked_lists:
        for rank, sc in enumerate(lst, start=1):
            cid = sc.chunk.chunk_id
            fused[cid] += weight / (k + rank)
            prev = keep.get(cid)
            if prev is None:
                keep[cid] = sc.model_copy(deep=True)
            else:
                prev.dense_rank = prev.dense_rank if prev.dense_rank is not None else sc.dense_rank
                prev.bm25_rank = prev.bm25_rank if prev.bm25_rank is not None else sc.bm25_rank
    out = []
    for cid, score in fused.items():
        sc = keep[cid]
        sc.score = score
        out.append(sc)
    out.sort(key=lambda s: s.score, reverse=True)
    return out
