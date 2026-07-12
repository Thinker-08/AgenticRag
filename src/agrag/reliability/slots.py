"""GPU-slot admission pool (C8, 07 §1.2): bounded concurrency with deadline-gated queueing.

Waiting past the caller's queue budget relocates the queue into the GPU — strictly worse — so a
slot that isn't free within `slot_wait_fraction × remaining` raises Backpressure and the caller
drops into its fallback tier instead of blocking indefinitely.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from ..contracts import Budget
from .errors import Backpressure


class SlotPool:
    def __init__(self, slots: int, *, wait_fraction: float = 0.3) -> None:
        self._sem = asyncio.Semaphore(max(1, slots))
        self.wait_fraction = wait_fraction

    @asynccontextmanager
    async def acquire(self, budget: Budget | None = None):
        timeout = None
        if budget is not None:
            timeout = max(0.05, budget.remaining_s() * self.wait_fraction)
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            raise Backpressure(f"no serving slot within {timeout:.2f}s") from None
        try:
            yield
        finally:
            self._sem.release()
