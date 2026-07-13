from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol, Sequence, runtime_checkable

from ..contracts import Budget, Chunk, Grade, ParsedDoc, ScoredChunk, Strategy, SubStep
from .types import ToolResult, VerdictResult


@runtime_checkable
class Parser(Protocol):
    async def parse(self, data: bytes, *, doc_id: str, tenant_id: str, filename: str = "") -> ParsedDoc: ...


@runtime_checkable
class Chunker(Protocol):
    name: str

    def split(self, doc: ParsedDoc) -> list[Chunk]: ...


@runtime_checkable
class Retriever(Protocol):
    async def retrieve(self, query: str, *, tenant_id: str, strategy: Strategy = Strategy.HYBRID, k: int = 100, filters: dict | None = None, budget: Budget | None = None) -> list[ScoredChunk]: ...


@runtime_checkable
class Reranker(Protocol):
    async def rerank(self, query: str, candidates: Sequence[ScoredChunk], *, top_k: int = 8, budget: Budget | None = None) -> list[ScoredChunk]: ...


@runtime_checkable
class Grader(Protocol):
    async def grade(self, step: SubStep, candidates: Sequence[ScoredChunk], *, budget: Budget | None = None) -> Grade: ...


@runtime_checkable
class Verifier(Protocol):
    async def entail(self, premise: str, hypothesis: str, *, budget: Budget | None = None) -> VerdictResult: ...


@runtime_checkable
class ToolRunner(Protocol):
    def run(self, code: str, inputs: dict, *, timeout_s: float = 2.0) -> ToolResult: ...


@runtime_checkable
class Tracer(Protocol):
    def startTrace(self, name: str, *, trace_id: str, tenant_id: str, **attrs) -> AbstractContextManager: ...

    def span(self, name: str, **attrs) -> AbstractContextManager: ...

    def event(self, name: str, **attrs) -> None: ...
