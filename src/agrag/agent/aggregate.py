from __future__ import annotations

import asyncio

from ..contracts import Budget, Chunk, Computation, Evidence, JobState, ScoredChunk, SubStep
from ..deps import Deps
from .schemas import ExtractedItems


def norm(item: str) -> str:
    return " ".join(item.lower().split()).strip(" .;,")


class MapReduceAggregator:
    def __init__(self, deps: Deps) -> None:
        self.deps = deps

    async def scanChunks(self, tenant_id: str) -> tuple[list[Chunk], bool]:
        cap = self.deps.settings.agent.max_scan_chunks
        ready = {d.doc_id for d in await self.deps.docstore.listDocs(tenant_id) if d.status == JobState.READY}
        chunks = [c for c in await self.deps.docstore.listChunks(tenant_id) if c.doc_id in ready]
        truncated = len(chunks) > cap

        return chunks[:cap], truncated

    async def aggregate(self, step: SubStep, *, tenant_id: str, budget: Budget) -> tuple[Evidence, Computation | None]:
        chunks, truncated = await self.scanChunks(tenant_id)
        if not chunks:
            return Evidence(scored=[]), None

        async def extract(chunk: Chunk) -> tuple[Chunk, list[str]]:
            if budget.exceeded():
                return chunk, []
            prompt = f"<query>{step.query}</query>\n\n[chunk {chunk.chunk_id} | doc {chunk.doc_id} | page {chunk.page_no}]\n{chunk.text}\nEVIDENCE_END"
            try:
                res, meta = await self.deps.small_llm.generateStructured(prompt, ExtractedItems, timeout_s=budget.callTimeoutS())
                budget.charge(meta.totalTokens)
            except Exception:
                return chunk, []
            return chunk, res.items

        mapped = await asyncio.gather(*(extract(c) for c in chunks))

        seen: dict[str, str] = {}
        contributors: list[ScoredChunk] = []
        for chunk, items in mapped:
            contributed = False
            for it in items:
                key = norm(it)
                if key and key not in seen:
                    seen[key] = it.strip()
                    contributed = True
            if contributed:
                contributors.append(ScoredChunk(chunk=chunk, score=1.0, why_relevant="aggregation source"))

        unique = list(seen.values())
        inputs = {"n": len(unique)}
        tool = await asyncio.to_thread(self.deps.toolrunner.run, "result = n", inputs, timeout_s=self.deps.settings.sandbox.timeout_s)
        comp = Computation(comp_id="agg_" + step.step_id, code=f"result = count(unique_items)  # {len(unique)} of {len(chunks)} chunks scanned", result=tool.result if tool.ok else len(unique), sandbox_run_id=tool.run_id)
        self.deps.tracer.event("aggregate", unique=len(unique), scanned=len(chunks), truncated=truncated)
        gaps = ["aggregation scanned a truncated chunk set; count may be a lower bound"] if truncated else []

        return Evidence(scored=contributors[: self.deps.settings.retrieval.top_k], gaps=gaps), comp
