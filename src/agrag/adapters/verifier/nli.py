"""NLI cross-encoder verifier — claim-level entailment against cited spans (06 §3).

A DeBERTa-style NLI head scores (premise, hypothesis) over contradiction/entailment/neutral;
we return the raw entailment probability as the score and let the caller apply tau thresholds.
"""

from __future__ import annotations

import asyncio

import numpy as np

from ...contracts import Budget, SupportLabel
from ...interfaces.types import VerdictResult


class NliVerifier:
    def __init__(self, model: str) -> None:
        self._model_name = model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError("NliVerifier needs the 'ml' extra: pip install -e '.[ml]'") from e
        self._model = CrossEncoder(model)
        self._entail_idx, self._contra_idx = self._resolve_labels()

    def _resolve_labels(self) -> tuple[int, int]:
        cfg = getattr(getattr(self._model, "model", None), "config", None)
        id2label = getattr(cfg, "id2label", None) or {}
        by_name = {str(name).lower(): int(idx) for idx, name in id2label.items()}
        entail = next((i for name, i in by_name.items() if "entail" in name), 1)
        contra = next((i for name, i in by_name.items() if "contradict" in name), 0)
        return entail, contra

    async def entail(
        self, premise: str, hypothesis: str, *, budget: Budget | None = None
    ) -> VerdictResult:
        p_entail, label = await asyncio.to_thread(self._predict, premise, hypothesis)
        return VerdictResult(label=label, score=p_entail, verifier="NLI")

    def _predict(self, premise: str, hypothesis: str) -> tuple[float, SupportLabel]:
        logits = np.asarray(self._model.predict([(premise, hypothesis)]))
        row = logits[0] if logits.ndim == 2 else logits
        probs = np.exp(row - row.max())
        probs = probs / probs.sum()
        p_entail = float(probs[self._entail_idx])
        argmax = int(probs.argmax())
        if argmax == self._entail_idx:
            label = SupportLabel.SUPPORTED
        elif argmax == self._contra_idx:
            label = SupportLabel.CONTRADICTED
        else:
            label = SupportLabel.UNSUPPORTED
        return p_entail, label
