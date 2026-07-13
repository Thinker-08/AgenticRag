from __future__ import annotations

import re
from typing import Sequence

from datasketch import MinHash, MinHashLSH

from ..contracts import ScoredChunk

_WORD = re.compile(r"\w+")


def shingles(text: str, n: int = 3) -> set[str]:
    toks = _WORD.findall(text.lower())
    if len(toks) < n:
        return set(toks)
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def minhash(text: str, num_perm: int = 64) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in shingles(text):
        m.update(sh.encode())
    return m


def dedupe(candidates: Sequence[ScoredChunk], *, threshold: float = 0.9) -> list[ScoredChunk]:
    if len(candidates) <= 1:
        return list(candidates)

    ordered = sorted(candidates, key=lambda s: s.score, reverse=True)
    lsh = MinHashLSH(threshold=threshold, num_perm=64)
    kept: list[ScoredChunk] = []

    for i, sc in enumerate(ordered):
        mh = minhash(sc.chunk.text)
        if lsh.query(mh):
            continue
        lsh.insert(f"c{i}", mh)
        kept.append(sc)

    return kept
