from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar

import structlog

_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")
_depth: ContextVar[int] = ContextVar("span_depth", default=0)

structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))


class LoggingTracer:
    def __init__(self) -> None:
        self._log = structlog.get_logger("agrag.trace")

    @contextmanager
    def start_trace(self, name: str, *, trace_id: str, tenant_id: str, **attrs):
        tok = _trace_id.set(trace_id)
        self._log.info("trace.start", span=name, trace_id=trace_id, tenant_id=tenant_id, **attrs)
        t0 = time.monotonic()
        try:
            yield self
        finally:
            self._log.info(
                "trace.end",
                span=name,
                trace_id=trace_id,
                ms=round((time.monotonic() - t0) * 1000, 1),
            )
            _trace_id.reset(tok)

    @contextmanager
    def span(self, name: str, **attrs):
        from ...security.pii import scrub_attrs

        attrs = scrub_attrs(attrs)
        depth = _depth.set(_depth.get() + 1)
        t0 = time.monotonic()
        self._log.info("span.start", span=name, trace_id=_trace_id.get(), **attrs)
        try:
            yield self
        finally:
            self._log.info(
                "span.end",
                span=name,
                trace_id=_trace_id.get(),
                ms=round((time.monotonic() - t0) * 1000, 1),
            )
            _depth.reset(depth)

    def event(self, name: str, **attrs) -> None:
        from ...security.pii import scrub_attrs

        self._log.info(name, trace_id=_trace_id.get(), **scrub_attrs(attrs))
