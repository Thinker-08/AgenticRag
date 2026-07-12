"""Corrective-RAG evidence grader (05 §5): relevance and slot-sufficiency kept separate.

Relevant-but-insufficient ("3 chunks all stating the same half") must not read as SUFFICIENT — a
distinction a single 0-1 score hides. This heuristic backend is deterministic and calibratable;
swap an NLI/LLM grader behind the same interface once validated against human labels (C28).
"""

from __future__ import annotations

import re
from typing import Sequence

from ..contracts import Budget, Grade, GradeVerdict, ScoredChunk, SubStep

_WORD = re.compile(r"[a-z0-9]+")
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")   # matches 2019 even glued to letters (FY2019)
_STOP = {"the", "a", "an", "of", "to", "in", "on", "and", "or", "is", "are", "was", "were",
         "for", "with", "as", "at", "by", "from", "what", "how", "did", "does", "compare", "vs"}


def _content(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


class HeuristicGrader:
    def __init__(self, relevance_floor: float = 0.7) -> None:
        self.floor = relevance_floor

    async def grade(self, step: SubStep, candidates: Sequence[ScoredChunk], *, budget: Budget | None = None) -> Grade:
        q = _content(step.query)
        if not candidates or not q:
            return Grade(verdict=GradeVerdict.IRRELEVANT, max_relevance=0.0, missing_slots=list(q),
                         rationale="no candidates")
        rel = 0.0
        union: set[str] = set()
        ev_text = []
        for sc in candidates:
            toks = _content(sc.chunk.text)
            rel = max(rel, len(q & toks) / len(q))
            union |= toks
            ev_text.append(sc.chunk.text)
        covered = q & union
        coverage = len(covered) / len(q)
        missing = sorted(q - covered)

        # temporal mismatch: a queried year/period absent from all evidence can't be covered (04 §6).
        q_years = set(_YEAR.findall(step.query))
        if q_years and not (q_years & set(_YEAR.findall(" ".join(ev_text)))):
            return Grade(verdict=GradeVerdict.IRRELEVANT, max_relevance=round(rel, 3),
                         missing_slots=sorted(q_years), rationale="queried period absent from evidence")

        if coverage >= 0.6 or (rel >= min(self.floor, 0.5) and coverage >= 0.4):
            verdict = GradeVerdict.SUFFICIENT
        elif coverage >= 0.2:
            verdict = GradeVerdict.AMBIGUOUS
        else:
            verdict = GradeVerdict.IRRELEVANT
        return Grade(verdict=verdict, max_relevance=round(rel, 3),
                     covered_slots=sorted(covered), missing_slots=missing, rationale="heuristic")
