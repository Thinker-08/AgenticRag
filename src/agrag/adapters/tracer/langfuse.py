from __future__ import annotations

import contextlib
import contextvars
import logging
from contextlib import AbstractContextManager

_log = logging.getLogger("agrag.tracer.langfuse")
_current: contextvars.ContextVar = contextvars.ContextVar("agrag_langfuse_current", default=None)


class LangfuseTracer:
    def __init__(self, host: str) -> None:
        self._client = None
        try:
            from langfuse import Langfuse

            self._client = Langfuse(host=host)
        except Exception as e:
            _log.warning("Langfuse unavailable; tracing disabled: %s", e)

    @contextlib.contextmanager
    def start_trace(
        self, name: str, *, trace_id: str, tenant_id: str, **attrs
    ) -> AbstractContextManager:
        if self._client is None:
            yield None
            return
        handle = None
        token = None
        try:
            handle = self._client.trace(
                id=trace_id,
                name=name,
                user_id=tenant_id,
                metadata={"tenant_id": tenant_id, **attrs},
            )
            token = _current.set(handle)
        except Exception as e:
            _log.debug("start_trace failed: %s", e)
        try:
            yield handle
        finally:
            if token is not None:
                _current.reset(token)
            try:
                self._client.flush()
            except Exception:
                pass

    @contextlib.contextmanager
    def span(self, name: str, **attrs) -> AbstractContextManager:
        parent = _current.get()
        if self._client is None or parent is None:
            yield None
            return
        handle = None
        token = None
        try:
            handle = parent.span(name=name, metadata=attrs)
            token = _current.set(handle)
        except Exception as e:
            _log.debug("span failed: %s", e)
        try:
            yield handle
        finally:
            if token is not None:
                _current.reset(token)
            try:
                if handle is not None:
                    handle.end()
            except Exception:
                pass

    def event(self, name: str, **attrs) -> None:
        from ...security.pii import scrub_attrs

        parent = _current.get()
        if self._client is None or parent is None:
            return
        try:
            parent.event(name=name, metadata=scrub_attrs(attrs))
        except Exception as e:
            _log.debug("event failed: %s", e)
