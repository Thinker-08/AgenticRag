from __future__ import annotations

from ...contracts import Chunk, Document


class MemoryDocStore:
    def __init__(self) -> None:
        self._docs: dict[tuple[str, str], Document] = {}
        self._by_hash: dict[tuple[str, str], str] = {}
        self._chunks: dict[tuple[str, str], Chunk] = {}

    async def getByHash(self, tenant_id: str, content_hash: str) -> Document | None:
        doc_id = self._by_hash.get((tenant_id, content_hash))
        return self._docs.get((tenant_id, doc_id)) if doc_id else None

    async def getDoc(self, tenant_id: str, doc_id: str) -> Document | None:
        return self._docs.get((tenant_id, doc_id))

    async def listDocs(self, tenant_id: str) -> list[Document]:
        return [d for (t, _), d in self._docs.items() if t == tenant_id]

    async def upsertDoc(self, doc: Document) -> None:
        self._docs[(doc.tenant_id, doc.doc_id)] = doc
        if doc.content_hash:
            self._by_hash[(doc.tenant_id, doc.content_hash)] = doc.doc_id

    async def putChunks(self, chunks) -> None:
        for c in chunks:
            self._chunks[(c.tenant_id, c.chunk_id)] = c

    async def deleteChunks(self, doc_id: str, tenant_id: str) -> None:
        self._chunks = {k: c for k, c in self._chunks.items() if not (k[0] == tenant_id and c.doc_id == doc_id)}

    async def getChunk(self, tenant_id: str, chunk_id: str) -> Chunk | None:
        return self._chunks.get((tenant_id, chunk_id))

    async def listChunks(self, tenant_id: str, *, limit: int | None = None) -> list[Chunk]:
        out = [c for (t, _), c in self._chunks.items() if t == tenant_id and not c.extra_metadata.get("is_parent")]
        return out[:limit] if limit else out
