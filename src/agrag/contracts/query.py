from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Intent(str, Enum):
    FACTOID = "factoid"
    AGGREGATION = "aggregation"
    COMPARISON = "comparison"
    SUMMARIZATION = "summarization"
    MULTI_HOP = "multi_hop"
    CHITCHAT = "chitchat"


class Strategy(str, Enum):
    SEMANTIC = "semantic"
    BM25 = "bm25"
    HYBRID = "hybrid"
    TABLE = "table"
    CODE = "code"
    METADATA_FILTER = "metadata_filter"
    DOC_SUMMARY = "doc_summary"
    GRAPH = "graph"


class Route(BaseModel):
    intent: Intent
    needs_retrieval: bool
    history_answerable: bool = False
    rationale: str = ""


class SubStep(BaseModel):
    step_id: str
    tool: Strategy = Strategy.HYBRID
    query: str
    k: int = 8
    depends_on: list[str] = Field(default_factory=list)


class QueryPlan(BaseModel):
    query_id: str
    trace_id: str = ""
    intent: Intent = Intent.FACTOID
    sub_steps: list[SubStep]
    merge: str = "concat"
    token_budget: int = 0


class GradeVerdict(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    AMBIGUOUS = "AMBIGUOUS"
    IRRELEVANT = "IRRELEVANT"
    EXHAUSTED = "EXHAUSTED"


class Grade(BaseModel):
    verdict: GradeVerdict
    max_relevance: float = 0.0
    covered_slots: list[str] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    rationale: str = ""
