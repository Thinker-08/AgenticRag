from __future__ import annotations

import re
import uuid

from ..contracts import ABSTENTION_TEXT, Answer, AnswerFormat, AnswerStatus, Claim, Computation, Intent, SupportLabel

_FORMAT = {Intent.FACTOID: AnswerFormat.PROSE, Intent.AGGREGATION: AnswerFormat.LIST, Intent.COMPARISON: AnswerFormat.TABLE, Intent.SUMMARIZATION: AnswerFormat.PROSE, Intent.MULTI_HOP: AnswerFormat.PROSE}

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def newId() -> str:
    return "ans_" + uuid.uuid4().hex[:8]


def abstain(trace_id: str, reason: str, *, gaps: list[str] | None = None) -> Answer:
    return Answer(answer_id=newId(), trace_id=trace_id, status=AnswerStatus.ABSTAINED, answer_text=ABSTENTION_TEXT, abstention_reason=reason, gaps=gaps or [])


def fmtNum(v) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return f"{v}"


def render(intent: Intent, claims: list[Claim], comps: list[Computation]) -> str:
    fmt = _FORMAT.get(intent, AnswerFormat.PROSE)
    parts = [c.text for c in claims]

    if fmt == AnswerFormat.LIST and len(parts) > 1:
        body = "\n".join(f"- {p}" for p in parts)
    elif fmt == AnswerFormat.TABLE and parts:
        body = "\n".join(f"| {p} |" for p in parts)
    else:
        body = " ".join(parts)

    for comp in comps:
        if comp.result is not None:
            body += f"\n({comp.code} = {fmtNum(comp.result)})"

    return body.strip()


def numericDrift(answer_text: str, comps: list[Computation]) -> list[Computation]:
    present = {n.replace(",", "").rstrip(".") for n in _NUM.findall(answer_text)}
    drifted = []

    for c in comps:
        if c.result is None:
            continue
        want = fmtNum(c.result).replace(",", "")
        if want not in present and want.rstrip("0").rstrip(".") not in {p.rstrip("0").rstrip(".") for p in present}:
            drifted.append(c)

    return drifted


def buildAnswer(question: str, intent: Intent, claims: list[Claim], *, trace_id: str, computations: list[Computation] | None = None, gaps: list[str] | None = None) -> Answer:
    computations = computations or []
    gaps = list(gaps or [])
    supported = [c for c in claims if c.support == SupportLabel.SUPPORTED]
    contradicted = [c for c in claims if c.support == SupportLabel.CONTRADICTED]

    if not supported:
        return abstain(trace_id, "contradicted" if contradicted else "no_evidence", gaps=gaps)

    for c in contradicted:
        gaps.append(f"conflicting evidence contradicts a claim: {c.text[:120]}")

    text = render(intent, supported, computations)

    drift = numericDrift(text, computations)
    if drift:
        gaps.append("computed value not reflected verbatim in the answer (numeric drift)")

    dropped = len(claims) - len(supported)
    status = AnswerStatus.PARTIAL if (gaps or dropped > 0 or contradicted) else AnswerStatus.ANSWERED
    return Answer(answer_id=newId(), trace_id=trace_id, status=status, format=_FORMAT.get(intent, AnswerFormat.PROSE), answer_text=text, claims=supported, computations=computations, gaps=gaps)
