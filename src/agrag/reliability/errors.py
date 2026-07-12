"""Typed failure signals that drive the degradation ladder (07 §2.5)."""

from __future__ import annotations


class CircuitOpen(Exception):
    """Dependency breaker is Open: fail fast to the next fallback tier (C12)."""


class Backpressure(Exception):
    """No serving slot within the queue budget: shed/degrade instead of blocking (C8)."""


class RetriesExhausted(Exception):
    """All deadline-gated retry attempts failed (C11)."""
