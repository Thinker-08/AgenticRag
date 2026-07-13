from __future__ import annotations


class CircuitOpen(Exception):
    pass


class Backpressure(Exception):
    pass


class RetriesExhausted(Exception):
    pass
