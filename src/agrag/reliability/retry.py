"""Deadline-gated retries with exponential backoff + full jitter (C11, 07 §2.2).

Only retriable faults retry; deterministic errors propagate immediately — retrying them just
burns the deadline. Full jitter is deliberate: synchronized retries after a model-server hiccup
arrive as a thundering herd and re-trip the very slot limit that caused the failure.
"""

from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, TypeVar

from ..contracts import Budget
from .errors import RetriesExhausted

T = TypeVar("T")

_MIN_ATTEMPT_S = 0.05


async def retry_async(
    op: Callable[[], Awaitable[T]],
    *,
    retriable: tuple[type[BaseException], ...],
    max_retries: int = 2,
    base_s: float = 0.2,
    cap_s: float = 2.0,
    budget: Budget | None = None,
    rng: random.Random | None = None,
    sleep=asyncio.sleep,
) -> T:
    rng = rng or random.Random()
    last: BaseException | None = None
    for attempt in range(max_retries + 1):
        if budget is not None and budget.remaining_s() < _MIN_ATTEMPT_S:
            break                                    # never start an attempt we cannot finish (C13)
        try:
            return await op()
        except retriable as exc:
            last = exc
            if attempt >= max_retries:
                break
            delay = min(cap_s, base_s * (2**attempt)) * rng.uniform(0.5, 1.0)
            if budget is not None:
                delay = min(delay, max(0.0, budget.remaining_s() - _MIN_ATTEMPT_S))
            await sleep(delay)
    raise RetriesExhausted(f"gave up after {max_retries + 1} attempts") from last
