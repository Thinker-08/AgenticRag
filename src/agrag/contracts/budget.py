"""The single hard budget threaded through every FSM node and tool call (C13)."""

from __future__ import annotations

import time
from dataclasses import dataclass


class BudgetExceeded(Exception):
    """Raised when a step tries to spend against an exhausted budget."""


@dataclass
class Budget:
    deadline: float          # absolute monotonic time
    tokens_left: int         # shared pool across ALL steps + tools
    iters_left: int          # per-query reformulation cap, strictly decreasing
    _clock: object = time    # injectable for deterministic tests

    @classmethod
    def start(cls, wall_clock_s: float, token_budget: int, max_iters: int = 3, clock=time) -> "Budget":
        return cls(
            deadline=clock.monotonic() + wall_clock_s,
            tokens_left=token_budget,
            iters_left=max_iters,
            _clock=clock,
        )

    def exceeded(self) -> bool:
        return self._clock.monotonic() >= self.deadline or self.tokens_left <= 0

    def iters_exhausted(self) -> bool:
        return self.iters_left <= 0

    def charge(self, tokens: int) -> None:
        self.tokens_left -= max(0, tokens)

    def consume_iter(self) -> None:
        self.iters_left -= 1

    def remaining_s(self) -> float:
        return max(0.0, self.deadline - self._clock.monotonic())

    def call_timeout_s(self, floor: float = 0.5) -> float:
        return max(floor, self.remaining_s())

    def snapshot(self) -> dict:
        return {
            "remaining_s": round(self.remaining_s(), 3),
            "tokens_left": self.tokens_left,
            "iters_left": self.iters_left,
        }
