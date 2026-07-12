"""`Evidence` / `Citation` — what generation cites and verification checks (02 §4.2)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .chunk import Chunk


class ScoredChunk(BaseModel):
    """A retrieved candidate with per-stage scores, before it becomes a Citation."""
    chunk: Chunk
    score: float = 0.0
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None
    why_relevant: str = ""


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    page_no: int
    char_span: tuple[int, int] = (0, 0)
    quote: str = ""             # exact substring of the NORMALIZED chunk text
    score: float = 0.0
    why_relevant: str = ""


class Evidence(BaseModel):
    """Frozen bundle handed to generation: deduped, reranked, tenant-scoped."""
    scored: list[ScoredChunk] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)   # sub-questions with no sufficient evidence

    @property
    def ids(self) -> set[str]:
        return {sc.chunk.chunk_id for sc in self.scored}

    def by_id(self, chunk_id: str) -> Chunk | None:
        for sc in self.scored:
            if sc.chunk.chunk_id == chunk_id:
                return sc.chunk
        return None

    def chunks(self) -> list[Chunk]:
        return [sc.chunk for sc in self.scored]
