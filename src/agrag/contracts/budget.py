from __future__ import annotations

import time
from dataclasses import dataclass


class BudgetExceeded(Exception):
    pass


@dataclass
class Budget:
    deadline: float
    tokens_left: int
    iters_left: int
    _clock: object = time

    @classmethod
    def start(cls, wall_clock_s: float, token_budget: int, max_iters: int = 3, clock=time) -> "Budget":
        return cls(deadline=clock.monotonic() + wall_clock_s, tokens_left=token_budget, iters_left=max_iters, _clock=clock)

    def exceeded(self) -> bool:
        return self._clock.monotonic() >= self.deadline or self.tokens_left <= 0

    def itersExhausted(self) -> bool:
        return self.iters_left <= 0

    def charge(self, tokens: int) -> None:
        self.tokens_left -= max(0, tokens)

    def consumeIter(self) -> None:
        self.iters_left -= 1

    def remainingS(self) -> float:
        return max(0.0, self.deadline - self._clock.monotonic())

    def callTimeoutS(self, floor: float = 0.5) -> float:
        return max(floor, self.remainingS())

    def snapshot(self) -> dict:
        return {"remaining_s": round(self.remainingS(), 3), "tokens_left": self.tokens_left, "iters_left": self.iters_left}
