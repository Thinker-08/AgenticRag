"""Per-dependency circuit breaker (C12, 07 §2.3): Closed → Open → HalfOpen → Closed.

Open short-circuits instantly to the next fallback tier; HalfOpen admits exactly one probe.
Thresholds are per-instance so a degraded reranker cannot open the generation breaker.
"""

from __future__ import annotations

import time


class CircuitBreaker:
    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, *, failures: int = 5, window_s: float = 30.0, cooldown_s: float = 15.0,
                 clock=time) -> None:
        self.failures = failures
        self.window_s = window_s
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._fail_times: list[float] = []
        self._state = self.CLOSED
        self._opened_at = 0.0
        self._probe_out = False

    @property
    def state(self) -> str:
        self._maybe_half_open()
        return self._state

    def _maybe_half_open(self) -> None:
        if self._state == self.OPEN and self._clock.monotonic() - self._opened_at >= self.cooldown_s:
            self._state = self.HALF_OPEN
            self._probe_out = False

    def allow(self) -> bool:
        """True iff a call may proceed. In HalfOpen only the single probe passes."""
        self._maybe_half_open()
        if self._state == self.CLOSED:
            return True
        if self._state == self.HALF_OPEN and not self._probe_out:
            self._probe_out = True
            return True
        return False

    def record_success(self) -> None:
        self._fail_times.clear()
        self._state = self.CLOSED
        self._probe_out = False

    def record_failure(self) -> None:
        now = self._clock.monotonic()
        if self._state == self.HALF_OPEN:      # probe failed: back to Open, restart cooldown
            self._state = self.OPEN
            self._opened_at = now
            self._probe_out = False
            return
        self._fail_times = [t for t in self._fail_times if now - t <= self.window_s]
        self._fail_times.append(now)
        if len(self._fail_times) >= self.failures:
            self._state = self.OPEN
            self._opened_at = now
