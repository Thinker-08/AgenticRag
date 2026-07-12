"""Stateless FastAPI serving plane (02 §2): ingest submit + query, every request trace-id'd.

State lives in backing stores, so any replica serves any request identically. The app + ingestion
share one Deps bundle (one shared index) — the seam between the two planes. Per-tenant rate limiting
(08 threat 6) and latency percentiles (09) live here; tenant_id is a placeholder for an auth-derived
principal (production must NOT trust it from the body — see 08 threat 3).
"""

from __future__ import annotations

import base64
import binascii
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..config import load_settings
from ..container import build_deps
from ..contracts import Answer, AnswerStatus, JobHandle, JobState, Turn
from ..deps import Deps
from ..ingestion.service import IngestionService
from .ops import LatencyStats, RateLimiter


class IngestRequest(BaseModel):
    tenant_id: str = "default"
    filename: str = ""
    text: str | None = None
    content_base64: str | None = None


class AskRequest(BaseModel):
    tenant_id: str = "default"
    query: str
    history: list[Turn] = []
    session_id: str | None = None


def create_app(settings=None) -> FastAPI:
    deps: Deps = build_deps(settings or load_settings())
    ingestion = IngestionService(deps)
    app_obj = _build_query_app(deps)
    limiter = RateLimiter(deps.settings.serving.rate_limit_qpm)
    stats = LatencyStats(deps.settings.serving.stats_window)
    api = FastAPI(title="Agentic RAG", version="0.1.0")

    @api.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "mode": deps.settings.mode,
            "agent_mode": deps.settings.agent_mode,
            "chunks": await deps.vectorstore.count(),
        }

    @api.get("/stats")
    async def get_stats() -> dict:
        return stats.snapshot()

    @api.post("/ingest", response_model=JobHandle)
    async def ingest(req: IngestRequest) -> JobHandle:
        if req.text is not None:
            doc = await ingestion.ingest_text(
                req.text, tenant_id=req.tenant_id, filename=req.filename
            )
            if doc.status != JobState.READY:
                raise HTTPException(
                    422, f"ingest {doc.status.value}: {doc.error or 'see /docs status'}"
                )
            return JobHandle(doc_id=doc.doc_id, status="ready")
        if req.content_base64:
            try:
                data = base64.b64decode(req.content_base64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise HTTPException(400, f"content_base64 is not valid base64: {exc}") from exc
            return await ingestion.submit(data, tenant_id=req.tenant_id, filename=req.filename)
        raise HTTPException(400, "provide either `text` or `content_base64`")

    @api.get("/docs/{tenant_id}/{doc_id}")
    async def doc_status(tenant_id: str, doc_id: str) -> dict:
        doc = await ingestion.status(tenant_id, doc_id)
        if not doc:
            raise HTTPException(404, "unknown doc")
        return {
            "doc_id": doc.doc_id,
            "status": doc.status,
            "progress": doc.progress(),
            "pages_done": doc.pages_done,
            "page_count": doc.page_count,
            "error": doc.error,
        }

    @api.post("/ask", response_model=Answer)
    async def ask(req: AskRequest) -> Answer:
        if not limiter.allow(req.tenant_id):
            raise HTTPException(429, "per-tenant rate limit exceeded")
        t0 = time.monotonic()
        answer = await app_obj.answer(
            req.query, req.history or None, tenant_id=req.tenant_id, session_id=req.session_id
        )
        stats.record(
            (time.monotonic() - t0) * 1000,
            abstained=answer.status == AnswerStatus.ABSTAINED,
            cached=answer.from_cache,
            degraded=answer.degraded,
        )
        return answer

    return api


def _build_query_app(deps: Deps):
    if deps.settings.is_baseline():
        from ..baseline.vanilla import BaselineRAG

        return BaselineRAG(deps)
    from ..agent.app import AgentApp

    return AgentApp(deps)


app = create_app()
