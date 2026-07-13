from __future__ import annotations

import base64
import binascii

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..config import loadSettings
from ..container import buildDeps
from ..contracts import Answer, JobHandle, JobState, Turn
from ..deps import Deps
from ..ingestion.service import IngestionService


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


def createApp(settings=None) -> FastAPI:
    deps: Deps = buildDeps(settings or loadSettings())
    ingestion = IngestionService(deps)
    app_obj = buildQueryApp(deps)
    api = FastAPI(title="Agentic RAG", version="0.1.0")

    @api.get("/health")
    async def health() -> dict:
        return {"status": "ok", "mode": deps.settings.mode, "agent_mode": deps.settings.agent_mode, "chunks": await deps.vectorstore.count()}

    @api.post("/ingest", response_model=JobHandle)
    async def ingest(req: IngestRequest) -> JobHandle:
        if req.text is not None:
            doc = await ingestion.ingestText(req.text, tenant_id=req.tenant_id, filename=req.filename)
            if doc.status != JobState.READY:
                raise HTTPException(422, f"ingest {doc.status.value}: {doc.error or 'see /docs status'}")
            return JobHandle(doc_id=doc.doc_id, status="ready")

        if req.content_base64:
            try:
                data = base64.b64decode(req.content_base64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise HTTPException(400, f"content_base64 is not valid base64: {exc}") from exc
            return await ingestion.submit(data, tenant_id=req.tenant_id, filename=req.filename)

        raise HTTPException(400, "provide either `text` or `content_base64`")

    @api.get("/docs/{tenant_id}/{doc_id}")
    async def docStatus(tenant_id: str, doc_id: str) -> dict:
        doc = await ingestion.status(tenant_id, doc_id)
        if not doc:
            raise HTTPException(404, "unknown doc")
        return {"doc_id": doc.doc_id, "status": doc.status, "progress": doc.progress(), "pages_done": doc.pages_done, "page_count": doc.page_count, "error": doc.error}

    @api.post("/ask", response_model=Answer)
    async def ask(req: AskRequest) -> Answer:
        return await app_obj.answer(req.query, req.history or None, tenant_id=req.tenant_id, session_id=req.session_id)

    return api


def buildQueryApp(deps: Deps):
    if deps.settings.isBaseline():
        from ..baseline.vanilla import BaselineRAG

        return BaselineRAG(deps)
    from ..agent.app import AgentApp

    return AgentApp(deps)


app = createApp()
