"""Pure metric functions over `Answer` + `GoldenItem` (09 §5 metrics catalog).

Deterministic, no I/O, no model calls — so a red gate is a real regression, not noise.
Retrieval metrics operate on whatever chunk ids the answer actually surfaced
(`Answer.sources()`), which is the only retrieval signal visible at the answer plane.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import TYPE_CHECKING, Iterable, Sequence

from ..contracts import AnswerStatus, SupportLabel

if TYPE_CHECKING:
    from ..contracts import Answer
    from .golden import GoldenItem

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> str:
    text = text.lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _tokens(text: str) -> list[str]:
    return _normalize(text).split()


def token_f1(pred: str, gold: str) -> float:
    """SQuAD-style token overlap F1 between prediction and gold answer."""
    p, g = _tokens(pred), _tokens(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    overlap = sum((Counter(p) & Counter(g)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    """1.0 iff normalized prediction equals normalized gold."""
    return float(_normalize(pred) == _normalize(gold))


def recall_at_k(retrieved_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    """Fraction of labeled relevant chunks present in the top-k retrieved ids."""
    gold = set(gold_ids)
    if not gold:
        return 0.0
    topk = set(list(retrieved_ids)[:k])
    return sum(1 for g in gold if g in topk) / len(gold)


def context_precision(retrieved: Sequence[str], gold_ids: Iterable[str]) -> float:
    """Signal-to-noise of surfaced context: share of retrieved ids that are relevant."""
    ret = list(retrieved)
    gold = set(gold_ids)
    if not ret or not gold:
        return 0.0
    return sum(1 for r in ret if r in gold) / len(ret)


def faithfulness(answer: "Answer") -> float:
    """Fraction of claims labeled SUPPORTED (vacuously 1.0 when there are no claims)."""
    claims = answer.claims
    if not claims:
        return 1.0
    supported = sum(1 for c in claims if c.support == SupportLabel.SUPPORTED)
    return supported / len(claims)


def citation_accuracy(answer: "Answer") -> float:
    """Among cited claims, the share whose cited spans support the claim (SUPPORTED)."""
    cited = [c for c in answer.claims if c.citations]
    if not cited:
        return 0.0
    return sum(1 for c in cited if c.support == SupportLabel.SUPPORTED) / len(cited)


def correct_refusal(answer: "Answer", item: "GoldenItem") -> bool:
    """True when an unanswerable question is correctly abstained on."""
    return (not item.answerable) and answer.status == AnswerStatus.ABSTAINED


def over_abstention(answer: "Answer", item: "GoldenItem") -> bool:
    """True when an answerable question is wrongly abstained on (miscalibration)."""
    return item.answerable and answer.status == AnswerStatus.ABSTAINED


def metrics_over(results: list[dict]) -> dict:
    """Mean of every numeric metric across per-item result dicts, skipping absent keys."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for r in results:
        for key, val in r.items():
            if val is None:
                continue
            if isinstance(val, bool):
                val = float(val)
            elif not isinstance(val, (int, float)):
                continue
            sums[key] = sums.get(key, 0.0) + float(val)
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums}
