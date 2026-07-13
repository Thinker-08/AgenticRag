from __future__ import annotations

from typing import Sequence

from ...contracts import Chunk, Document


class RedisDocStore:
    def __init__(self, host: str) -> None:
        try:
            import redis.asyncio as aioredis
        except ImportError as e:
            raise ImportError(
                "RedisDocStore needs the 'stores' extra: pip install -e '.[stores]'"
            ) from e
        self._redis = aioredis.from_url(host, decode_responses=True)

    async def get_by_hash(self, tenant_id: str, content_hash: str) -> Document | None:
        doc_id = await self._redis.get(self._hash_key(tenant_id, content_hash))
        if not doc_id:
            return None
        return await self.get_doc(tenant_id, doc_id)

    async def get_doc(self, tenant_id: str, doc_id: str) -> Document | None:
        raw = await self._redis.get(self._doc_key(tenant_id, doc_id))
        return Document.model_validate_json(raw) if raw else None

    async def list_docs(self, tenant_id: str) -> list[Document]:
        doc_ids = await self._redis.smembers(self._index_key(tenant_id))
        if not doc_ids:
            return []
        raws = await self._redis.mget([self._doc_key(tenant_id, d) for d in sorted(doc_ids)])
        return [Document.model_validate_json(r) for r in raws if r]

    async def upsert_doc(self, doc: Document) -> None:
        await self._redis.set(self._doc_key(doc.tenant_id, doc.doc_id), doc.model_dump_json())
        await self._redis.sadd(self._index_key(doc.tenant_id), doc.doc_id)
        if doc.content_hash:
            await self._redis.set(self._hash_key(doc.tenant_id, doc.content_hash), doc.doc_id)

    async def put_chunks(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        pipe = self._redis.pipeline(transaction=False)
        for c in chunks:
            pipe.set(self._chunk_key(c.tenant_id, c.chunk_id), c.model_dump_json())
            if not c.extra_metadata.get("is_parent"):
                pipe.sadd(self._chunk_index_key(c.tenant_id), c.chunk_id)
        await pipe.execute()

    async def get_chunk(self, tenant_id: str, chunk_id: str) -> Chunk | None:
        raw = await self._redis.get(self._chunk_key(tenant_id, chunk_id))
        return Chunk.model_validate_json(raw) if raw else None

    async def list_chunks(self, tenant_id: str, *, limit: int | None = None) -> list[Chunk]:
        ids = sorted(await self._redis.smembers(self._chunk_index_key(tenant_id)))
        if limit:
            ids = ids[:limit]
        if not ids:
            return []
        raws = await self._redis.mget([self._chunk_key(tenant_id, i) for i in ids])
        return [Chunk.model_validate_json(r) for r in raws if r]

    @staticmethod
    def _index_key(tenant_id: str) -> str:
        return f"agrag:docs:{tenant_id}"

    @staticmethod
    def _doc_key(tenant_id: str, doc_id: str) -> str:
        return f"agrag:doc:{tenant_id}:{doc_id}"

    @staticmethod
    def _hash_key(tenant_id: str, content_hash: str) -> str:
        return f"agrag:dochash:{tenant_id}:{content_hash}"

    @staticmethod
    def _chunk_key(tenant_id: str, chunk_id: str) -> str:
        return f"agrag:chunk:{tenant_id}:{chunk_id}"

    @staticmethod
    def _chunk_index_key(tenant_id: str) -> str:
        return f"agrag:chunks:{tenant_id}"
