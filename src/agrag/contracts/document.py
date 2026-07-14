from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class JobState(str, Enum):
    QUEUED = "queued"
    PARSING = "parsing"
    LAYING_OUT = "laying_out"
    NORMALIZING = "normalizing"
    CHUNKING = "chunking"
    CONTEXTUALIZING = "contextualizing"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"
    QUARANTINED = "quarantined"
    SUPERSEDED = "superseded"


TERMINAL_STATES = {JobState.READY, JobState.FAILED, JobState.QUARANTINED, JobState.SUPERSEDED}


class Document(BaseModel):
    doc_id: str
    tenant_id: str
    content_hash: str
    filename: str = ""
    status: JobState = JobState.QUEUED
    page_count: int = 0
    pages_done: int = 0
    created_at: str = ""
    indexed_at: Optional[str] = None
    embedding_model: str = ""
    embedding_version: str = ""
    error: Optional[str] = None

    def progress(self) -> float:
        return (self.pages_done / self.page_count) if self.page_count else 0.0


class Job(BaseModel):
    job_id: str
    doc_id: str
    tenant_id: str
    content_hash: str
    state: JobState = JobState.QUEUED
    trace_id: str = ""
    attempts: int = 0
    error: Optional[str] = None

    def isTerminal(self) -> bool:
        return self.state in TERMINAL_STATES


class JobHandle(BaseModel):
    doc_id: str
    job_id: Optional[str] = None
    status: str = "queued"
    deduped: bool = False


class PageProgress(BaseModel):
    doc_id: str
    status: JobState
    page: int = 0
    page_count: int = 0
    eta_s: Optional[float] = None
    message: str = ""
