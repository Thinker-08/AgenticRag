"""Assemble the final Answer (06 §5-§7): partial answers, calibrated abstention, intent-shaped formatting.

A partial answer with an honest gap beats both a confident hallucination and a blanket refusal. Abstention
phrasing is fixed and unambiguous — a document gap, never "I don't know" (a model gap). Contradiction is a
hard signal (surfaced, never silently dropped) and a computed value must equal what the answer states.
"""

from __future__ import annotations

import re
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

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _new_id() -> str:
    return "ans_" + uuid.uuid4().hex[:8]


def abstain(trace_id: str, reason: str, *, gaps: list[str] | None = None) -> Answer:
    return Answer(
        answer_id=_new_id(),
        trace_id=trace_id,
        status=AnswerStatus.ABSTAINED,
        answer_text=ABSTENTION_TEXT,
        abstention_reason=reason,
        gaps=gaps or [],
    )


def _fmt_num(v) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return f"{v}"


def _render(intent: Intent, claims: list[Claim], comps: list[Computation]) -> str:
    fmt = _FORMAT.get(intent, AnswerFormat.PROSE)
    parts = [c.text for c in claims]
    if fmt == AnswerFormat.LIST and len(parts) > 1:
        body = "\n".join(f"- {p}" for p in parts)
    elif fmt == AnswerFormat.TABLE and parts:
        body = "\n".join(f"| {p} |" for p in parts)          # one row per compared target (06 §7)
    else:
        body = " ".join(parts)
    for comp in comps:
        if comp.result is not None:
            body += f"\n({comp.code} = {_fmt_num(comp.result)})"
    return body.strip()


def numeric_drift(answer_text: str, comps: list[Computation]) -> list[Computation]:
    """A computed value must appear in the answer verbatim; a written number that drifts from the
    sandbox result is the 15%-vs-18.4% failure the code tool exists to prevent (06 §4)."""
    present = {n.replace(",", "").rstrip(".") for n in _NUM.findall(answer_text)}
    drifted = []
    for c in comps:
        if c.result is None:
            continue
        want = _fmt_num(c.result).replace(",", "")
        if want not in present and want.rstrip("0").rstrip(".") not in {p.rstrip("0").rstrip(".") for p in present}:
            drifted.append(c)
    return drifted


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
    gaps = list(gaps or [])
    supported = [c for c in claims if c.support == SupportLabel.SUPPORTED]
    contradicted = [c for c in claims if c.support == SupportLabel.CONTRADICTED]

    # Contradiction is surfaced, never silently dropped (05 §8 / 06 §6c): with no supported claim it
    # is a hard whole-answer block; alongside supported claims it is flagged in the answer's gaps.
    if not supported:
        return abstain(trace_id, "contradicted" if contradicted else "no_evidence", gaps=gaps)
    for c in contradicted:
        gaps.append(f"conflicting evidence contradicts a claim: {c.text[:120]}")

    text = _render(intent, supported, computations)

    # Grounded arithmetic integrity: the stated number must equal the sandbox result (06 §4).
    drift = numeric_drift(text, computations)
    if drift:
        gaps.append("computed value not reflected verbatim in the answer (numeric drift)")

    dropped = len(claims) - len(supported)
    status = (AnswerStatus.PARTIAL if (gaps or dropped > 0 or contradicted) else AnswerStatus.ANSWERED)
    return Answer(
        answer_id=_new_id(),
        trace_id=trace_id,
        status=status,
        format=_FORMAT.get(intent, AnswerFormat.PROSE),
        answer_text=text,
        claims=supported,
        computations=computations,
        gaps=gaps,
    )
