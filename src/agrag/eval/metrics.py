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


def normalize(text: str) -> str:
    text = text.lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def tokenF1(pred: str, gold: str) -> float:
    p, g = tokens(pred), tokens(gold)

    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0

    overlap = sum((Counter(p) & Counter(g)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def exactMatch(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def recallAtK(retrieved_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0

    topk = set(list(retrieved_ids)[:k])
    return sum(1 for g in gold if g in topk) / len(gold)


def contextPrecision(retrieved: Sequence[str], gold_ids: Iterable[str]) -> float:
    ret = list(retrieved)
    gold = set(gold_ids)

    if not ret or not gold:
        return 0.0
    return sum(1 for r in ret if r in gold) / len(ret)


def faithfulness(answer: "Answer") -> float:
    claims = answer.claims
    if not claims:
        return 1.0

    supported = sum(1 for c in claims if c.support == SupportLabel.SUPPORTED)
    return supported / len(claims)


def citationAccuracy(answer: "Answer") -> float:
    cited = [c for c in answer.claims if c.citations]
    if not cited:
        return 0.0
    return sum(1 for c in cited if c.support == SupportLabel.SUPPORTED) / len(cited)


def correctRefusal(answer: "Answer", item: "GoldenItem") -> bool:
    return (not item.answerable) and answer.status == AnswerStatus.ABSTAINED


def overAbstention(answer: "Answer", item: "GoldenItem") -> bool:
    return item.answerable and answer.status == AnswerStatus.ABSTAINED


def metricsOver(results: list[dict]) -> dict:
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
