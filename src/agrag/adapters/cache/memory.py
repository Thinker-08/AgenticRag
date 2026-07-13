from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable


class MemoryCache:
    def __init__(self) -> None:
        self._data: dict[str, tuple[Any, float | None]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def fresh(self, key: str) -> bool:
        if key not in self._data:
            return False

        _, exp = self._data[key]
        if exp is not None and time.monotonic() > exp:
            self._data.pop(key, None)
            return False
        return True

    async def get(self, key: str) -> Any | None:
        return self._data[key][0] if self.fresh(key) else None

    async def set(self, key: str, value: Any, ttl_s: int | None = None) -> None:
        self._data[key] = (value, (time.monotonic() + ttl_s) if ttl_s else None)

    async def getOrCompute(self, key: str, compute: Callable[[], Awaitable[Any]], ttl_s: int | None = None) -> Any:
        if self.fresh(key):
            return self._data[key][0]

        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            if self.fresh(key):
                return self._data[key][0]

            value = await compute()
            await self.set(key, value, ttl_s)
            return value

    async def invalidate(self, key: str) -> None:
        self._data.pop(key, None)
