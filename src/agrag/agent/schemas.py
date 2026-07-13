from __future__ import annotations

from pydantic import BaseModel, Field


class RewriteResult(BaseModel):
    standalone_query: str
    carried_entities: list[str] = Field(default_factory=list)
    resolved: bool = True


class CodeInput(BaseModel):
    name: str
    value: float
    source_chunk_id: str = ""
    cell_ref: str = ""


class CodePlan(BaseModel):
    inputs: list[CodeInput] = Field(default_factory=list)
    code: str = ""
    claim_template: str = ""


class ExtractedItems(BaseModel):
    items: list[str] = Field(default_factory=list)


class PlanCritique(BaseModel):
    redundant_step_ids: list[str] = Field(default_factory=list)
    rationale: str = ""
