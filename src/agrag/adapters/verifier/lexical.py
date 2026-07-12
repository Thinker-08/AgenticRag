"""Lexical entailment proxy — the local stand-in for the NLI cross-encoder (06 §3).

A real DeBERTa-NLI head decides entailment; here we approximate it with content-token containment
plus a hard numeric-grounding check: any number in the claim that is absent from the premise sharply
lowers the score, which is exactly the hallucinated-figure failure the verifier must catch.
"""

from __future__ import annotations

import re

from ...contracts import Budget, SupportLabel
from ...interfaces.types import VerdictResult

_WORD = re.compile(r"[a-z]+")
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "and", "or", "is", "are", "was", "were", "be",
    "for", "with", "as", "at", "by", "from", "that", "this", "it", "its", "s", "was", "has", "had",
}
_NEG = {"not", "no", "never", "without", "excluding", "fell", "decreased", "declined"}


def _content_words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


def _nums(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUM.findall(text)}


class LexicalVerifier:
    async def entail(self, premise: str, hypothesis: str, *, budget: Budget | None = None) -> VerdictResult:
        prem_words = set(_content_words(premise))
        hyp_words = _content_words(hypothesis)
        if not hyp_words:
            return VerdictResult(label=SupportLabel.UNSUPPORTED, score=0.0, verifier="lexical")

        covered = sum(1 for w in hyp_words if w in prem_words)
        word_score = covered / len(hyp_words)

        hyp_nums = _nums(hypothesis)
        prem_nums = _nums(premise)
        num_penalty = 0.0
        if hyp_nums:
            missing = hyp_nums - prem_nums
            num_penalty = 0.7 * (len(missing) / len(hyp_nums))

        neg_prem = len(_NEG & prem_words)
        neg_hyp = len({w for w in hyp_words} & _NEG)
        contradicted = (neg_prem != neg_hyp) and word_score > 0.6

        score = max(0.0, word_score - num_penalty)
        if contradicted and num_penalty == 0.0:
            return VerdictResult(label=SupportLabel.CONTRADICTED, score=0.05, verifier="lexical")
        label = SupportLabel.SUPPORTED if score >= 0.6 else SupportLabel.UNSUPPORTED
        return VerdictResult(label=label, score=round(score, 4), verifier="lexical")
