from __future__ import annotations

import contextlib
import contextvars
from contextlib import AbstractContextManager

_current: contextvars.ContextVar = contextvars.ContextVar("agrag_otel_span", default=None)


class OtelTracer:
    def __init__(self, host: str = "") -> None:
        self._tracer = None
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider

            provider = TracerProvider(resource=Resource.create({"service.name": "agrag"}))
            self.maybeAddOtlp(provider, host)
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer("agrag")
        except Exception:
            self._tracer = None

    def maybeAddOtlp(self, provider, host: str) -> None:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            endpoint = host or None
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        except Exception:
            pass

    @contextlib.contextmanager
    def startTrace(self, name: str, *, trace_id: str, tenant_id: str, **attrs) -> AbstractContextManager:
        if self._tracer is None:
            yield None
            return

        with self._tracer.start_as_current_span(name) as span:
            span.set_attribute("trace_id", trace_id)
            span.set_attribute("tenant_id", tenant_id)
            for k, v in attrs.items():
                span.set_attribute(k, str(v))

            token = _current.set(span)
            try:
                yield span
            finally:
                _current.reset(token)

    @contextlib.contextmanager
    def span(self, name: str, **attrs) -> AbstractContextManager:
        if self._tracer is None:
            yield None
            return

        with self._tracer.start_as_current_span(name) as span:
            for k, v in attrs.items():
                span.set_attribute(k, str(v))
            yield span

    def event(self, name: str, **attrs) -> None:
        from ...security.pii import scrubAttrs

        span = _current.get()
        if span is not None:
            try:
                span.add_event(name, {k: str(v) for k, v in scrubAttrs(attrs).items()})
            except Exception:
                pass
