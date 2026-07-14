from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import structlog

from ..contracts import TERMINAL_STATES, Document, Job, JobHandle, JobState, ParsedDoc
from ..deps import Deps
from .hashing import merkleDiff, sha256Bytes
from .jobs import JobQueue
from .metadata import enrichMetadata
from .stages import contextualize, embedAndIndex, tagPii

log = structlog.get_logger("agrag.ingestion")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IngestionService:
    def __init__(self, deps: Deps) -> None:
        self.deps = deps
        self.queue = JobQueue(self.run, concurrency=max(1, deps.settings.agent.slot_concurrency // 2))

    def oversized(self, data: bytes) -> bool:
        return len(data) > self.deps.settings.parser.max_upload_mb * 1024 * 1024

    async def quarantineOversized(self, doc_id: str, tenant_id: str, content_hash: str, filename: str) -> Document:
        doc = self.newDoc(doc_id, tenant_id, content_hash, filename).model_copy(update={"status": JobState.QUARANTINED, "error": f"upload exceeds max_upload_mb={self.deps.settings.parser.max_upload_mb}"})
        await self.deps.docstore.upsertDoc(doc)
        return doc

    async def submit(self, data: bytes, *, tenant_id: str = "default", filename: str = "") -> JobHandle:
        content_hash = sha256Bytes(data)
        existing = await self.deps.docstore.getByHash(tenant_id, content_hash)

        if existing and existing.status == JobState.READY:
            return JobHandle(doc_id=existing.doc_id, status="ready", deduped=True)
        if existing and existing.status not in TERMINAL_STATES:
            return JobHandle(doc_id=existing.doc_id, status=existing.status.value, deduped=True)

        doc_id = existing.doc_id if existing else uuid.uuid4().hex[:12]
        if self.oversized(data):
            doc = await self.quarantineOversized(doc_id, tenant_id, content_hash, filename)
            return JobHandle(doc_id=doc.doc_id, status=doc.status.value)

        doc = self.newDoc(doc_id, tenant_id, content_hash, filename)
        await self.deps.docstore.upsertDoc(doc)
        job = Job(job_id=uuid.uuid4().hex[:12], doc_id=doc_id, tenant_id=tenant_id, content_hash=content_hash, trace_id=uuid.uuid4().hex[:12])
        await self.queue.enqueue(job, data)

        return JobHandle(doc_id=doc_id, job_id=job.job_id, status="queued")

    async def ingest(self, data: bytes, *, tenant_id: str = "default", filename: str = "", doc_id: str | None = None) -> Document:
        content_hash = sha256Bytes(data)
        existing = await self.deps.docstore.getByHash(tenant_id, content_hash)

        if existing and existing.status == JobState.READY:
            return existing

        doc_id = doc_id or (existing.doc_id if existing else uuid.uuid4().hex[:12])
        if self.oversized(data):
            return await self.quarantineOversized(doc_id, tenant_id, content_hash, filename)

        doc = self.newDoc(doc_id, tenant_id, content_hash, filename)
        await self.deps.docstore.upsertDoc(doc)
        job = Job(job_id=uuid.uuid4().hex[:12], doc_id=doc_id, tenant_id=tenant_id, content_hash=content_hash)
        await self.run(job, data)

        return await self.deps.docstore.getDoc(tenant_id, doc_id)

    async def ingestText(self, text: str, *, tenant_id: str = "default", filename: str = "", doc_id: str | None = None) -> Document:
        return await self.ingest(text.encode("utf-8"), tenant_id=tenant_id, filename=filename or "text.txt", doc_id=doc_id)

    async def status(self, tenant_id: str, doc_id: str) -> Document | None:
        return await self.deps.docstore.getDoc(tenant_id, doc_id)

    def newDoc(self, doc_id: str, tenant_id: str, content_hash: str, filename: str) -> Document:
        return Document(doc_id=doc_id, tenant_id=tenant_id, content_hash=content_hash, filename=filename, status=JobState.QUEUED, created_at=now(), embedding_model=self.deps.embedding.model, embedding_version=self.deps.embedding.version)

    async def set(self, doc: Document, state: JobState, **fields) -> Document:
        doc = doc.model_copy(update={"status": state, **fields})
        await self.deps.docstore.upsertDoc(doc)
        return doc

    async def run(self, job: Job, data: bytes) -> None:
        doc = await self.deps.docstore.getDoc(job.tenant_id, job.doc_id)
        if doc is None:
            doc = self.newDoc(job.doc_id, job.tenant_id, job.content_hash, "")
            await self.deps.docstore.upsertDoc(doc)
        trace_id = job.trace_id or job.job_id

        with self.deps.tracer.startTrace("ingest", trace_id=trace_id, tenant_id=job.tenant_id, doc_id=job.doc_id):
            try:
                with self.deps.tracer.span("parse"):
                    doc = await self.set(doc, JobState.PARSING)
                    parsed: ParsedDoc = await asyncio.wait_for(self.deps.parser.parse(data, doc_id=job.doc_id, tenant_id=job.tenant_id, filename=doc.filename), timeout=self.deps.settings.parser.parse_timeout_s)
                    parsed.content_hash = job.content_hash
                    doc = await self.set(doc, JobState.CHUNKING, page_count=parsed.page_count)

                with self.deps.tracer.span("chunk"):
                    page_hashes = self.pageHashes(parsed)
                    await self.merkleReport(job.tenant_id, doc.filename, page_hashes)
                    chunks = self.deps.chunker.split(parsed)
                    doc_text = "\n".join(b.text for p in parsed.pages for b in p.blocks)
                    chunks = enrichMetadata(chunks, doc_text)
                    chunks = tagPii(chunks)
                    doc = doc.model_copy(update={"page_hashes": page_hashes})

                with self.deps.tracer.span("contextualize", n=len(chunks)):
                    doc = await self.set(doc, JobState.CONTEXTUALIZING)
                    chunks = await contextualize(chunks, parsed.doc_summary, self.deps.small_llm, self.deps.cache)

                with self.deps.tracer.span("embed_index"):
                    doc = await self.set(doc, JobState.EMBEDDING)
                    indexed = await embedAndIndex(chunks, embedding=self.deps.embedding, vectorstore=self.deps.vectorstore, lexical=self.deps.lexical, docstore=self.deps.docstore, cache=self.deps.cache)

                await self.set(doc, JobState.READY, indexed_at=now(), pages_done=parsed.page_count)
                await self.supersedePriorVersions(doc)
                self.deps.tracer.event("ingest.ready", doc_id=job.doc_id, chunks=len(indexed))
            except asyncio.TimeoutError:
                await self.set(doc, JobState.QUARANTINED, error=f"parse exceeded {self.deps.settings.parser.parse_timeout_s}s (suspected PDF-bomb)")
                log.warning("ingest.timeout", doc_id=job.doc_id)
            except Exception as exc:
                state = JobState.QUARANTINED if isHostile(exc) else JobState.FAILED
                await self.set(doc, state, error=f"{type(exc).__name__}: {exc}")
                log.warning("ingest.failed", doc_id=job.doc_id, error=str(exc), state=state.value)

    def pageHashes(self, parsed: ParsedDoc) -> dict[int, str]:
        from .hashing import contentHash

        return {p.page_no: contentHash("\n".join(b.text for b in p.blocks)) for p in parsed.pages}

    async def merkleReport(self, tenant_id: str, filename: str, new_hashes: dict[int, str]) -> None:
        if not filename:
            return

        prior = next((d for d in await self.deps.docstore.listDocs(tenant_id) if d.filename == filename and d.status == JobState.READY and d.page_hashes), None)
        if not prior:
            return

        changed = merkleDiff(prior.page_hashes, new_hashes)
        reused = len(set(prior.page_hashes) & set(new_hashes)) - len(changed & set(prior.page_hashes))
        self.deps.tracer.event("merkle.diff", changed_pages=sorted(changed), reused_pages=max(0, reused), total=len(new_hashes))

    async def supersedePriorVersions(self, doc: Document) -> None:
        if not doc.filename:
            return

        for prior in await self.deps.docstore.listDocs(doc.tenant_id):
            if prior.doc_id != doc.doc_id and prior.filename == doc.filename and prior.status == JobState.READY and prior.content_hash != doc.content_hash:
                await self.deps.vectorstore.deleteDoc(prior.doc_id, doc.tenant_id)
                await self.deps.lexical.deleteDoc(prior.doc_id, doc.tenant_id)
                await self.deps.docstore.upsertDoc(prior.model_copy(update={"status": JobState.SUPERSEDED}))
                self.deps.tracer.event("ingest.superseded", old_doc=prior.doc_id, new_doc=doc.doc_id)


def isHostile(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(w in msg for w in ("encrypted", "password", "bomb", "inflate", "too many pages", "max_pages"))
