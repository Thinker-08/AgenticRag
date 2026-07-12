"""Assemble the final Answer (06 §5-§7): partial answers, calibrated abstention, intent-shaped formatting.

A partial answer with an honest gap beats both a confident hallucination and a blanket refusal. Abstention
phrasing is fixed and unambiguous — a document gap, never "I don't know" (a model gap).
"""

from __future__ import annotations

import uuid

from ..contracts import (
    ABSTENTION_TEXT,
    Answer,
    AnswerFormat,
    AnswerStatus,
    Claim,
    Computation,
    Intent,
    SupportLabel,
)

_FORMAT = {
    Intent.FACTOID: AnswerFormat.PROSE,
    Intent.AGGREGATION: AnswerFormat.LIST,
    Intent.COMPARISON: AnswerFormat.TABLE,
    Intent.SUMMARIZATION: AnswerFormat.PROSE,
    Intent.MULTI_HOP: AnswerFormat.PROSE,
}


def _new_id() -> str:
    return "ans_" + uuid.uuid4().hex[:8]


def abstain(trace_id: str, reason: str, *, gaps: list[str] | None = None) -> Answer:
    return Answer(answer_id=_new_id(), trace_id=trace_id, status=AnswerStatus.ABSTAINED,
                  answer_text=ABSTENTION_TEXT, abstention_reason=reason, gaps=gaps or [])


def _render(intent: Intent, claims: list[Claim], comps: list[Computation]) -> str:
    fmt = _FORMAT.get(intent, AnswerFormat.PROSE)
    parts = [c.text for c in claims]
    if fmt == AnswerFormat.LIST and len(parts) > 1:
        body = "\n".join(f"- {p}" for p in parts)
    else:
        body = " ".join(parts)
    for comp in comps:
        if comp.result is not None:
            body += f"\n({comp.code} = {comp.result})"
    return body.strip()


def build_answer(
    question: str,
    intent: Intent,
    claims: list[Claim],
    *,
    trace_id: str,
    computations: list[Computation] | None = None,
    gaps: list[str] | None = None,
) -> Answer:
    computations = computations or []
    gaps = gaps or []
    supported = [c for c in claims if c.support == SupportLabel.SUPPORTED]
    contradicted = [c for c in claims if c.support == SupportLabel.CONTRADICTED]

    if not supported:
        return abstain(trace_id, "contradicted" if contradicted else "no_evidence", gaps=gaps)

    dropped = len(claims) - len(supported)
    status = AnswerStatus.PARTIAL if (gaps or dropped > 0) else AnswerStatus.ANSWERED
    return Answer(
        answer_id=_new_id(), trace_id=trace_id, status=status,
        format=_FORMAT.get(intent, AnswerFormat.PROSE),
        answer_text=_render(intent, supported, computations),
        claims=supported, computations=computations, gaps=gaps,
    )
