"""Deterministic hashing embedder — a dependency-free stand-in for BGE-M3 (dense + sparse).

Not semantically strong, but it is a real, normalized vector space: identical text -> identical
vector, related text shares n-gram features, and it emits a learned-sparse-style bag so the whole
hybrid + RRF + rerank funnel runs and is testable without a GPU. Swap for BgeM3Embedding in full mode.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Sequence

import numpy as np

from ...interfaces.types import EmbeddingResult

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _token_hash(token: str, salt: str = "") -> int:
    return int.from_bytes(hashlib.blake2b((salt + token).encode(), digest_size=8).digest(), "big")


class HashEmbedding:
    def __init__(self, dim: int = 1024, version: str = "hash-1") -> None:
        self.model = "hash-embedding"
        self.version = version
        self.dim = dim

    def _dense(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        toks = _tokens(text)
        grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for tok, cnt in Counter(grams).items():
            idx = _token_hash(tok) % self.dim
            sign = 1.0 if _token_hash(tok, "sign") & 1 else -1.0
            vec[idx] += sign * (1.0 + math.log(cnt))
        norm = float(np.linalg.norm(vec))
        return (vec / norm).tolist() if norm > 0 else vec.tolist()

    def _sparse(self, text: str) -> dict[int, float]:
        toks = _tokens(text)
        if not toks:
            return {}
        counts = Counter(toks)
        out: dict[int, float] = {}
        for tok, cnt in counts.items():
            out[_token_hash(tok, "sparse") % (2**31)] = 1.0 + math.log(cnt)
        return out

    def _encode(self, texts: Sequence[str]) -> EmbeddingResult:
        return EmbeddingResult(
            dense=[self._dense(t) for t in texts],
            sparse=[self._sparse(t) for t in texts],
            model=self.model,
            version=self.version,
        )

    def encode_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> EmbeddingResult:
        return self._encode(texts)
