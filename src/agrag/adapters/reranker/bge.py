"""BGE cross-encoder reranker: precise second-stage scoring of candidates (C4)."""

from __future__ import annotations

import asyncio
from typing import Sequence

from ...contracts import Budget, ScoredChunk


class BgeReranker:
    def __init__(self, model: str, device: str = "cpu") -> None:
        self._model = model
        self._device = device
        self._reranker = None
        self._cross = None

        flag_err: ImportError | None = None
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as e:
            flag_err = e
        else:
            try:
                self._reranker = FlagReranker(
                    model, use_fp16=device.startswith("cuda"), device=device
                )
            except TypeError:
                self._reranker = FlagReranker(model, use_fp16=device.startswith("cuda"))

        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError("BgeReranker needs the 'ml' extra: pip install -e '.[ml]'") from (
                    flag_err or e
                )
            self._cross = CrossEncoder(model, device=device)

    async def rerank(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        *,
        top_k: int = 8,
        budget: Budget | None = None,
    ) -> list[ScoredChunk]:
        cands = list(candidates)
        if not cands:
            return []
        if budget is not None and budget.exceeded():
            return cands[:top_k]

        pairs = [(query, sc.chunk.text) for sc in cands]
        scores = await asyncio.to_thread(self._score, pairs)
        for sc, s in zip(cands, scores):
            sc.rerank_score = float(s)
            sc.score = float(s)
        cands.sort(key=lambda sc: sc.rerank_score or 0.0, reverse=True)
        return cands[:top_k]

    def _score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self._reranker is not None:
            try:
                out = self._reranker.compute_score(pairs, normalize=True)
            except TypeError:
                out = self._reranker.compute_score(pairs)
            if isinstance(out, (int, float)):
                return [float(out)]
            return [float(x) for x in out]
        out = self._cross.predict(pairs)
        return [float(x) for x in out]
