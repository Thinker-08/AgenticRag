"""Redis-backed conversation store (C16): tenant-scoped, capped rolling window."""

from __future__ import annotations

from ...contracts import Conversation, Turn


class RedisSessionStore:
    def __init__(self, host: str) -> None:
        try:
            import redis.asyncio as aioredis
        except ImportError as e:
            raise ImportError(
                "RedisSessionStore needs the 'stores' extra: pip install -e '.[stores]'"
            ) from e
        self._redis = aioredis.from_url(host, decode_responses=True)

    @staticmethod
    def _key(tenant_id: str, session_id: str) -> str:
        return f"agrag:session:{tenant_id}:{session_id}"

    async def get(self, tenant_id: str, session_id: str) -> Conversation:
        raws = await self._redis.lrange(self._key(tenant_id, session_id), 0, -1)
        turns = [Turn.model_validate_json(r) for r in raws]
        return Conversation(session_id=session_id, tenant_id=tenant_id, turns=turns)

    async def append(self, tenant_id: str, session_id: str, turn: Turn, *, max_turns: int = 20) -> None:
        key = self._key(tenant_id, session_id)
        pipe = self._redis.pipeline(transaction=True)
        pipe.rpush(key, turn.model_dump_json())
        pipe.ltrim(key, -max_turns, -1)
        await pipe.execute()
