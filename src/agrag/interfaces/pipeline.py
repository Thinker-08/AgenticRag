"""Chunker, Retriever, Reranker, Grader, Verifier, ToolRunner, Parser, Tracer."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol, Sequence, runtime_checkable

from ..contracts import (
    Budget,
    Chunk,
    Grade,
    ParsedDoc,
    ScoredChunk,
    Strategy,
    SubStep,
)
from .types import ToolResult, VerdictResult


@runtime_checkable
class Parser(Protocol):
    """Parse/OCR: extract text + geometry, cascading digital -> OCR -> vision by text-ratio (03 §1)."""

    async def parse(
        self, data: bytes, *, doc_id: str, tenant_id: str, filename: str = ""
    ) -> ParsedDoc: ...


@runtime_checkable
class Chunker(Protocol):
    """Split a parsed doc into Chunks with breadcrumbs + parents (03 §4)."""

    name: str

    def split(self, doc: ParsedDoc) -> list[Chunk]: ...


@runtime_checkable
class Retriever(Protocol):
    """Return candidate chunks for a query under a strategy (04)."""

    async def retrieve(
        self,
        query: str,
        *,
        tenant_id: str,
        strategy: Strategy = Strategy.HYBRID,
        k: int = 100,
        filters: dict | None = None,
        budget: Budget | None = None,
    ) -> list[ScoredChunk]: ...


@runtime_checkable
class Reranker(Protocol):
    """Reorder candidates with a precise cross-encoder (C4)."""

    async def rerank(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        *,
        top_k: int = 8,
        budget: Budget | None = None,
    ) -> list[ScoredChunk]: ...


@runtime_checkable
class Grader(Protocol):
    """CRAG evidence grader: relevance + slot-sufficiency (05 §5). Calibrated against labels (C28)."""

    async def grade(
        self, step: SubStep, candidates: Sequence[ScoredChunk], *, budget: Budget | None = None
    ) -> Grade: ...


@runtime_checkable
class Verifier(Protocol):
    """Claim-level NLI entailment against cited spans (06 §3). Not the generator judging itself."""

    async def entail(
        self, premise: str, hypothesis: str, *, budget: Budget | None = None
    ) -> VerdictResult: ...


@runtime_checkable
class ToolRunner(Protocol):
    """Execute allowlisted, model-generated arithmetic in an isolated sandbox (C30)."""

    def run(self, code: str, inputs: dict, *, timeout_s: float = 2.0) -> ToolResult: ...


@runtime_checkable
class Tracer(Protocol):
    """One trace per query, one span per step/tool (C25)."""

    def start_trace(
        self, name: str, *, trace_id: str, tenant_id: str, **attrs
    ) -> AbstractContextManager: ...

    def span(self, name: str, **attrs) -> AbstractContextManager: ...

    def event(self, name: str, **attrs) -> None: ...
