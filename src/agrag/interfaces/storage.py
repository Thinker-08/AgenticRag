"""VectorStore, LexicalIndex, DocStore, Cache — the persistence seams."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol, Sequence, runtime_checkable

from ..contracts import Chunk, Conversation, Document, ScoredChunk, Turn
from .types import SparseVector, VectorRecord


@runtime_checkable
class VectorStore(Protocol):
    """Upsert and ANN-search vectors with metadata filters. Tenant scoping is mandatory (C31)."""

    async def upsert(self, records: Sequence[VectorRecord]) -> None: ...

    async def search(
        self,
        query_dense: list[float],
        *,
        tenant_id: str,
        top_k: int = 100,
        query_sparse: SparseVector | None = None,
        filters: dict | None = None,
    ) -> list[ScoredChunk]: ...

    async def delete_doc(self, doc_id: str, tenant_id: str) -> None: ...

    async def count(self, tenant_id: str | None = None) -> int: ...


@runtime_checkable
class LexicalIndex(Protocol):
    """BM25 lexical retrieval over linearized chunk text."""

    async def add(self, chunks: Sequence[Chunk]) -> None: ...

    async def search(
        self, query: str, *, tenant_id: str, top_k: int = 100, filters: dict | None = None
    ) -> list[ScoredChunk]: ...

    async def delete_doc(self, doc_id: str, tenant_id: str) -> None: ...


@runtime_checkable
class DocStore(Protocol):
    """Durable document/job status (read-your-writes, C16) + chunk payload store."""

    async def get_by_hash(self, tenant_id: str, content_hash: str) -> Document | None: ...

    async def get_doc(self, tenant_id: str, doc_id: str) -> Document | None: ...

    async def list_docs(self, tenant_id: str) -> list[Document]: ...

    async def upsert_doc(self, doc: Document) -> None: ...

    async def put_chunks(self, chunks: Sequence[Chunk]) -> None: ...

    async def get_chunk(self, tenant_id: str, chunk_id: str) -> Chunk | None: ...

    async def list_chunks(self, tenant_id: str, *, limit: int | None = None) -> list[Chunk]:
        """All retrievable (non-parent) chunks for a tenant — the full-scan source for aggregation."""
        ...


@runtime_checkable
class SessionStore(Protocol):
    """Externalized conversation history (C16) so any stateless replica resolves a follow-up."""

    async def get(self, tenant_id: str, session_id: str) -> Conversation: ...

    async def append(
        self, tenant_id: str, session_id: str, turn: Turn, *, max_turns: int = 20
    ) -> None: ...


@runtime_checkable
class Cache(Protocol):
    """Multi-level get/set with content-hash keys + single-flight (C18, C19)."""

    async def get(self, key: str) -> Any | None: ...

    async def set(self, key: str, value: Any, ttl_s: int | None = None) -> None: ...

    async def get_or_compute(
        self, key: str, compute: Callable[[], Awaitable[Any]], ttl_s: int | None = None
    ) -> Any:
        """Single-flight: concurrent callers for the same key compute once."""
        ...

    async def invalidate(self, key: str) -> None: ...
