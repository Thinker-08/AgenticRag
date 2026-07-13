"""`Answer` / `Claim` / `Computation` — the verified, cited output of grounding (06)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .evidence import Citation


class AnswerStatus(str, Enum):
    ANSWERED = "ANSWERED"
    PARTIAL = "PARTIAL"
    ABSTAINED = "ABSTAINED"


class SupportLabel(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"


class AnswerFormat(str, Enum):
    PROSE = "prose"
    LIST = "list"
    TABLE = "table"
    SCALAR = "scalar"


class ComputationInput(BaseModel):
    name: str
    value: float
    source_chunk_id: str
    cell_ref: str = ""


class Computation(BaseModel):
    comp_id: str
    inputs: list[ComputationInput] = Field(default_factory=list)
    code: str = ""
    result: float | str | None = None
    sandbox_run_id: str = ""


class Claim(BaseModel):
    claim_id: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    support: SupportLabel = SupportLabel.UNSUPPORTED
    entail_score: float = 0.0
    verifier: str = ""


class DraftClaim(BaseModel):
    """The generator's constrained-decoded output shape (untrusted until verified)."""

    text: str
    citations: list[Citation] = Field(default_factory=list)


class Draft(BaseModel):
    answer_text: str
    format: AnswerFormat = AnswerFormat.PROSE
    claims: list[DraftClaim] = Field(default_factory=list)
    computations: list[Computation] = Field(default_factory=list)
    degraded: bool = False


ABSTENTION_TEXT = "This is not stated in the provided document(s)."


class Answer(BaseModel):
    answer_id: str
    trace_id: str = ""
    status: AnswerStatus = AnswerStatus.ANSWERED
    format: AnswerFormat = AnswerFormat.PROSE
    answer_text: str = ""
    claims: list[Claim] = Field(default_factory=list)
    computations: list[Computation] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    abstention_reason: str | None = None
    degraded: bool = False
    from_cache: bool = False
    carried_entities: list[str] = Field(default_factory=list)

    def sources(self) -> list[Citation]:
        out: list[Citation] = []
        for c in self.claims:
            out.extend(c.citations)
        return out
