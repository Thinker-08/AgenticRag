"""Typed LangGraph state for the agent FSM (05 §1). One key per branch point the machine consumes."""

from __future__ import annotations

from typing import TypedDict

from ..contracts import (
    Answer,
    Budget,
    Claim,
    Computation,
    Draft,
    Evidence,
    Grade,
    QueryPlan,
    Route,
    Turn,
)


class AgentState(TypedDict, total=False):
    query: str
    tenant_id: str
    trace_id: str
    history: list[Turn]
    budget: Budget

    standalone_q: str
    carried_entities: list[str]
    route: Route
    plan: QueryPlan

    evidence: Evidence
    computations: list[Computation]
    gaps: list[str]
    grade: Grade

    draft: Draft
    claims: list[Claim]
    answer: Answer
