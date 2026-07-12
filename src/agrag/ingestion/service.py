"""IngestionService — orchestrates the offline plane as an idempotent, queued job (02, 03)."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import structlog

from ..contracts import TERMINAL_STATES, Document, Job, JobHandle, JobState, ParsedDoc
from ..deps import Deps
from .crossref import link_crossrefs
from .hashing import sha256_bytes
from .jobs import JobQueue
from .stages import contextualize, embed_and_index, tag_pii

log = structlog.get_logger("agrag.ingestion")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IngestionService:
    def __init__(self, deps: Deps) -> None:
        self.deps = deps
        self.queue = JobQueue(
            self._run, concurrency=max(1, deps.settings.agent.slot_concurrency // 2)
        )

    def _oversized(self, data: bytes) -> bool:
        return len(data) > self.deps.settings.parser.max_upload_mb * 1024 * 1024

    async def _quarantine_oversized(
        self, doc_id: str, tenant_id: str, content_hash: str, filename: str
    ) -> Document:
        doc = self._new_doc(doc_id, tenant_id, content_hash, filename).model_copy(
            update={
                "status": JobState.QUARANTINED,
                "error": f"upload exceeds max_upload_mb={self.deps.settings.parser.max_upload_mb}",
            }
        )
        await self.deps.docstore.upsert_doc(doc)
        return doc

    async def submit(
        self, data: bytes, *, tenant_id: str = "default", filename: str = ""
    ) -> JobHandle:
        content_hash = sha256_bytes(data)
        existing = await self.deps.docstore.get_by_hash(tenant_id, content_hash)
        if existing and existing.status == JobState.READY:
            return JobHandle(doc_id=existing.doc_id, status="ready", deduped=True)
        if existing and existing.status not in TERMINAL_STATES:
            return JobHandle(doc_id=existing.doc_id, status=existing.status.value, deduped=True)
        doc_id = existing.doc_id if existing else uuid.uuid4().hex[:12]
        if self._oversized(data):
            doc = await self._quarantine_oversized(doc_id, tenant_id, content_hash, filename)
            return JobHandle(doc_id=doc.doc_id, status=doc.status.value)
        doc = self._new_doc(doc_id, tenant_id, content_hash, filename)
        await self.deps.docstore.upsert_doc(doc)
        job = Job(
            job_id=uuid.uuid4().hex[:12],
            doc_id=doc_id,
            tenant_id=tenant_id,
            content_hash=content_hash,
            trace_id=uuid.uuid4().hex[:12],
        )
        await self.queue.enqueue(job, data)
        return JobHandle(doc_id=doc_id, job_id=job.job_id, status="queued")

    async def ingest(
        self,
        data: bytes,
        *,
        tenant_id: str = "default",
        filename: str = "",
        doc_id: str | None = None,
    ) -> Document:
        """Synchronous convenience (eval / CLI / local): submit + process inline, return the final Document."""
        content_hash = sha256_bytes(data)
        existing = await self.deps.docstore.get_by_hash(tenant_id, content_hash)
        if existing and existing.status == JobState.READY:
            return existing
        doc_id = doc_id or (existing.doc_id if existing else uuid.uuid4().hex[:12])
        if self._oversized(data):
            return await self._quarantine_oversized(doc_id, tenant_id, content_hash, filename)
        doc = self._new_doc(doc_id, tenant_id, content_hash, filename)
        await self.deps.docstore.upsert_doc(doc)
        job = Job(
            job_id=uuid.uuid4().hex[:12],
            doc_id=doc_id,
            tenant_id=tenant_id,
            content_hash=content_hash,
        )
        await self._run(job, data)
        return await self.deps.docstore.get_doc(tenant_id, doc_id)

    async def ingest_text(
        self,
        text: str,
        *,
        tenant_id: str = "default",
        filename: str = "",
        doc_id: str | None = None,
    ) -> Document:
        return await self.ingest(
            text.encode("utf-8"),
            tenant_id=tenant_id,
            filename=filename or "text.txt",
            doc_id=doc_id,
        )

    async def status(self, tenant_id: str, doc_id: str) -> Document | None:
        return await self.deps.docstore.get_doc(tenant_id, doc_id)

    def _new_doc(self, doc_id: str, tenant_id: str, content_hash: str, filename: str) -> Document:
        return Document(
            doc_id=doc_id,
            tenant_id=tenant_id,
            content_hash=content_hash,
            filename=filename,
            status=JobState.QUEUED,
            created_at=_now(),
            embedding_model=self.deps.embedding.model,
            embedding_version=self.deps.embedding.version,
        )

    async def _set(self, doc: Document, state: JobState, **fields) -> Document:
        doc = doc.model_copy(update={"status": state, **fields})
        await self.deps.docstore.upsert_doc(doc)
        return doc

    async def _run(self, job: Job, data: bytes) -> None:
        doc = await self.deps.docstore.get_doc(job.tenant_id, job.doc_id)
        if doc is None:
            doc = self._new_doc(job.doc_id, job.tenant_id, job.content_hash, "")
            await self.deps.docstore.upsert_doc(doc)
        trace_id = job.trace_id or job.job_id
        with self.deps.tracer.start_trace(
            "ingest", trace_id=trace_id, tenant_id=job.tenant_id, doc_id=job.doc_id
        ):
            try:
                with self.deps.tracer.span("parse"):
                    doc = await self._set(doc, JobState.PARSING)
                    parsed: ParsedDoc = await asyncio.wait_for(
                        self.deps.parser.parse(
                            data, doc_id=job.doc_id, tenant_id=job.tenant_id, filename=doc.filename
                        ),
                        timeout=self.deps.settings.parser.parse_timeout_s,
                    )
                    parsed.content_hash = job.content_hash
                    doc = await self._set(doc, JobState.CHUNKING, page_count=parsed.page_count)

                with self.deps.tracer.span("chunk"):
                    link_crossrefs(parsed)
                    chunks = self.deps.chunker.split(parsed)
                    chunks = tag_pii(chunks)

                with self.deps.tracer.span("contextualize", n=len(chunks)):
                    doc = await self._set(doc, JobState.CONTEXTUALIZING)
                    chunks = await contextualize(
                        chunks, parsed.doc_summary, self.deps.small_llm, self.deps.cache
                    )

                with self.deps.tracer.span("embed_index"):
                    doc = await self._set(doc, JobState.EMBEDDING)
                    indexed = await embed_and_index(
                        chunks,
                        embedding=self.deps.embedding,
                        vectorstore=self.deps.vectorstore,
                        lexical=self.deps.lexical,
                        docstore=self.deps.docstore,
                        cache=self.deps.cache,
                    )

                await self._set(
                    doc, JobState.READY, indexed_at=_now(), pages_done=parsed.page_count
                )
                await self._supersede_prior_versions(doc)
                self.deps.tracer.event("ingest.ready", doc_id=job.doc_id, chunks=len(indexed))
            except asyncio.TimeoutError:
                await self._set(
                    doc,
                    JobState.QUARANTINED,
                    error=f"parse exceeded {self.deps.settings.parser.parse_timeout_s}s (suspected PDF-bomb)",
                )
                log.warning("ingest.timeout", doc_id=job.doc_id)
            except Exception as exc:
                state = JobState.QUARANTINED if _is_hostile(exc) else JobState.FAILED
                await self._set(doc, state, error=f"{type(exc).__name__}: {exc}")
                log.warning("ingest.failed", doc_id=job.doc_id, error=str(exc), state=state.value)

    async def _supersede_prior_versions(self, doc: Document) -> None:
        """Blue/green on re-upload of an edited doc (C17/C20): deindex prior READY versions of the
        same file so stale and current content never coexist in retrieval."""
        if not doc.filename:
            return
        for prior in await self.deps.docstore.list_docs(doc.tenant_id):
            if (
                prior.doc_id != doc.doc_id
                and prior.filename == doc.filename
                and prior.status == JobState.READY
                and prior.content_hash != doc.content_hash
            ):
                await self.deps.vectorstore.delete_doc(prior.doc_id, doc.tenant_id)
                await self.deps.lexical.delete_doc(prior.doc_id, doc.tenant_id)
                await self.deps.docstore.upsert_doc(
                    prior.model_copy(update={"status": JobState.SUPERSEDED})
                )
                self.deps.tracer.event(
                    "ingest.superseded", old_doc=prior.doc_id, new_doc=doc.doc_id
                )


def _is_hostile(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        w in msg
        for w in ("encrypted", "password", "bomb", "inflate", "too many pages", "max_pages")
    )
