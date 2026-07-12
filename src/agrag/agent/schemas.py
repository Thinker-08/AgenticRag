"""Small constrained-decoding schemas the FSM consumes (rewrite, code plan). All grammar-emitted (C3)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RewriteResult(BaseModel):
    """Follow-up resolution: elliptical/pronoun query -> standalone (05 §9)."""

    standalone_query: str
    carried_entities: list[str] = Field(default_factory=list)
    resolved: bool = True


class CodeInput(BaseModel):
    name: str
    value: float
    source_chunk_id: str = ""
    cell_ref: str = ""


class CodePlan(BaseModel):
    """The LLM sets up arithmetic symbolically over NAMED inputs; the sandbox executes it (06 §4)."""

    inputs: list[CodeInput] = Field(default_factory=list)
    code: str = ""
    claim_template: str = ""


class ExtractedItems(BaseModel):
    """Map step of aggregation map-reduce (05 §8): items in one chunk matching the query."""

    items: list[str] = Field(default_factory=list)


class PlanCritique(BaseModel):
    """Self-RAG plan critique before retrieval (05 §6): flag redundant steps cheaply here rather
    than after k failed retrievals. `redundant_step_ids` are dropped (unless a kept step depends on them)."""

    redundant_step_ids: list[str] = Field(default_factory=list)
    rationale: str = ""
