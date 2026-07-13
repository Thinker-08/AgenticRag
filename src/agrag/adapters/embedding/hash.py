from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Sequence

import numpy as np

from ...interfaces.types import EmbeddingResult

_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def tokenHash(token: str, salt: str = "") -> int:
    return int.from_bytes(hashlib.blake2b((salt + token).encode(), digest_size=8).digest(), "big")


class HashEmbedding:
    def __init__(self, dim: int = 1024, version: str = "hash-1") -> None:
        self.model = "hash-embedding"
        self.version = version
        self.dim = dim

    def _dense(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        toks = tokens(text)
        grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

        for tok, cnt in Counter(grams).items():
            idx = tokenHash(tok) % self.dim
            sign = 1.0 if tokenHash(tok, "sign") & 1 else -1.0
            vec[idx] += sign * (1.0 + math.log(cnt))

        norm = float(np.linalg.norm(vec))
        return (vec / norm).tolist() if norm > 0 else vec.tolist()

    def _sparse(self, text: str) -> dict[int, float]:
        toks = tokens(text)
        if not toks:
            return {}

        counts = Counter(toks)
        out: dict[int, float] = {}
        for tok, cnt in counts.items():
            out[tokenHash(tok, "sparse") % (2**31)] = 1.0 + math.log(cnt)

        return out

    def encode(self, texts: Sequence[str]) -> EmbeddingResult:
        return EmbeddingResult(dense=[self._dense(t) for t in texts], sparse=[self._sparse(t) for t in texts], model=self.model, version=self.version)

    def encodeDocuments(self, texts: Sequence[str]) -> EmbeddingResult:
        return self.encode(texts)

    def encodeQueries(self, texts: Sequence[str]) -> EmbeddingResult:
        return self.encode(texts)
