"""Serving ops: per-tenant rate limiting (08 threat 6) and latency percentile tracking (09, C26)."""

from __future__ import annotations

import time
from collections import deque


class RateLimiter:
    """Sliding-window per-tenant QPM cap. 0 disables. Defends GPU slots from a single-tenant flood."""

    def __init__(self, qpm: int, *, clock=time) -> None:
        self.qpm = qpm
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}

    def allow(self, tenant_id: str) -> bool:
        if self.qpm <= 0:
            return True
        now = self._clock.monotonic()
        window = self._hits.setdefault(tenant_id, deque())
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= self.qpm:
            return False
        window.append(now)
        return True


class LatencyStats:
    """Ring buffer of per-request latencies -> p50/p95/p99 + cost, the ship-blocking SLOs (09)."""

    def __init__(self, window: int = 1000) -> None:
        self._ms: deque[float] = deque(maxlen=window)
        self._abstained = 0
        self._cached = 0
        self._degraded = 0
        self._total = 0

    def record(self, ms: float, *, abstained: bool, cached: bool, degraded: bool) -> None:
        self._ms.append(ms)
        self._total += 1
        self._abstained += int(abstained)
        self._cached += int(cached)
        self._degraded += int(degraded)

    @staticmethod
    def _pct(sorted_ms: list[float], p: float) -> float:
        if not sorted_ms:
            return 0.0
        idx = min(len(sorted_ms) - 1, int(round((p / 100) * (len(sorted_ms) - 1))))
        return round(sorted_ms[idx], 1)

    def snapshot(self) -> dict:
        ms = sorted(self._ms)
        n = self._total or 1
        return {
            "count": self._total,
            "p50_ms": self._pct(ms, 50),
            "p95_ms": self._pct(ms, 95),
            "p99_ms": self._pct(ms, 99),
            "abstention_rate": round(self._abstained / n, 4),
            "cache_hit_rate": round(self._cached / n, 4),
            "degraded_rate": round(self._degraded / n, 4),
        }
