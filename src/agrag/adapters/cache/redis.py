from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable

_LOCK_TTL_S = 30
_POLL_INTERVAL_S = 0.1
_POLL_MAX = 300


class RedisCache:
    def __init__(self, host: str) -> None:
        try:
            import redis.asyncio as aioredis
        except ImportError as e:
            raise ImportError(
                "RedisCache needs the 'stores' extra: pip install -e '.[stores]'"
            ) from e
        self._redis = aioredis.from_url(host, decode_responses=True)

    async def get(self, key: str) -> Any | None:
        raw = await self._redis.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    async def set(self, key: str, value: Any, ttl_s: int | None = None) -> None:
        raw = json.dumps(value, default=str)
        await self._redis.set(key, raw, ex=ttl_s if ttl_s else None)

    async def invalidate(self, key: str) -> None:
        await self._redis.delete(key)

    async def get_or_compute(
        self, key: str, compute: Callable[[], Awaitable[Any]], ttl_s: int | None = None
    ) -> Any:
        cached = await self.get(key)
        if cached is not None:
            return cached

        lock_key = f"{key}:lock"
        for _ in range(_POLL_MAX):
            if await self._redis.set(lock_key, "1", nx=True, ex=_LOCK_TTL_S):
                try:
                    value = await compute()
                    await self.set(key, value, ttl_s)
                    return value
                finally:
                    await self._redis.delete(lock_key)
            await asyncio.sleep(_POLL_INTERVAL_S)
            cached = await self.get(key)
            if cached is not None:
                return cached

        value = await compute()
        await self.set(key, value, ttl_s)
        return value
