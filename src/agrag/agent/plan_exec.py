"""Execute a QueryPlan DAG (05 §4): topological layers, concurrent fan-out, sandboxed code steps.

Independent steps in a layer run concurrently under a GPU-slot semaphore (C7/C8). Dependent steps run
later so step-1's evidence feeds step-2. Code steps offload arithmetic to the sandbox (C30) — never the LLM.
"""

from __future__ import annotations

import asyncio
import uuid

from ..contracts import (
    Budget,
    Computation,
    ComputationInput,
    Evidence,
    QueryPlan,
    ScoredChunk,
    Strategy,
    SubStep,
)
from ..deps import Deps
from ..retrieval.dedupe import dedupe
from .schemas import CodePlan


def _layers(steps: list[SubStep]) -> list[list[SubStep]]:
    done: set[str] = set()
    remaining = list(steps)
    out: list[list[SubStep]] = []
    while remaining:
        layer = [s for s in remaining if all(d in done for d in s.depends_on)]
        if not layer:
            layer = remaining                              # broken DAG -> run the rest, don't deadlock
        out.append(layer)
        done.update(s.step_id for s in layer)
        remaining = [s for s in remaining if s.step_id not in done]
    return out


class PlanExecutor:
    def __init__(self, deps: Deps) -> None:
        self.deps = deps
        self._slot_sem = asyncio.Semaphore(deps.settings.agent.slot_concurrency)

    async def run(self, plan: QueryPlan, *, tenant_id: str, budget: Budget) -> tuple[Evidence, list[Computation], list[str]]:
        results: dict[str, list[ScoredChunk]] = {}
        computations: list[Computation] = []
        gaps: list[str] = []

        for layer in _layers(plan.sub_steps):
            async def run_one(step: SubStep):
                async with self._slot_sem:
                    if budget.exceeded():
                        return step, []
                    if step.tool == Strategy.CODE:
                        comp = await self._code_step(step, results, tenant_id=tenant_id, budget=budget)
                        if comp:
                            computations.append(comp)
                        return step, []
                    query, filters = step.query, None
                    if step.tool == Strategy.METADATA_FILTER:
                        query, filters = await self._self_query(step.query, budget)
                    docs = await self.deps.retriever.retrieve(
                        query, tenant_id=tenant_id, strategy=step.tool, k=step.k,
                        filters=filters, budget=budget)
                    if not docs and filters:
                        # a filter over an unpopulated field must degrade to recall, not to
                        # a false "not in the document" (C14)
                        docs = await self.deps.retriever.retrieve(
                            query, tenant_id=tenant_id, strategy=step.tool, k=step.k, budget=budget)
                    return step, docs

            for step, docs in await asyncio.gather(*(run_one(s) for s in layer)):
                results[step.step_id] = docs
                if not docs and step.tool != Strategy.CODE:
                    gaps.append(step.query)

        merged = await self._merge(results, tenant_id)
        return merged, computations, gaps

    async def _merge(self, results: dict[str, list[ScoredChunk]], tenant_id: str) -> Evidence:
        pool: dict[str, ScoredChunk] = {}
        for docs in results.values():
            for sc in docs:
                cur = pool.get(sc.chunk.chunk_id)
                if cur is None or sc.score > cur.score:
                    pool[sc.chunk.chunk_id] = sc
        scored = dedupe(list(pool.values()), threshold=self.deps.settings.retrieval.dedupe_threshold)
        scored.sort(key=lambda s: s.score, reverse=True)
        scored = scored[: self.deps.settings.retrieval.top_k]
        scored = await self._expand_parents(scored, tenant_id)
        return Evidence(scored=scored)

    async def _expand_parents(self, scored: list[ScoredChunk], tenant_id: str) -> list[ScoredChunk]:
        seen = {sc.chunk.chunk_id for sc in scored}
        extra: list[ScoredChunk] = []
        for sc in scored:
            pid = sc.chunk.parent_id
            if pid and pid not in seen:
                parent = await self.deps.docstore.get_chunk(tenant_id, pid)
                if parent:
                    seen.add(pid)
                    extra.append(ScoredChunk(chunk=parent, score=sc.score * 0.5, why_relevant="parent context"))
        return scored + extra

    async def _self_query(self, query: str, budget: Budget) -> tuple[str, dict | None]:
        from ..retrieval.selfquery import self_query
        sq = await self_query(self.deps.small_llm, query, timeout_s=budget.call_timeout_s())
        return sq.semantic_query, (sq.filters or None)

    async def _code_step(self, step: SubStep, results: dict[str, list[ScoredChunk]], *,
                         tenant_id: str, budget: Budget) -> Computation | None:
        dep_chunks: list[ScoredChunk] = []
        for dep in step.depends_on:
            dep_chunks.extend(results.get(dep, []))
        if not dep_chunks:
            return None
        from ..promptfmt import build_evidence_block, nonce_for
        block = build_evidence_block(Evidence(scored=dep_chunks[:6]), nonce_for(step.step_id))
        prompt = f"<query>{step.query}</query>\n\n{block}"
        try:
            plan, llm_result = await self.deps.small_llm.generate_structured(
                prompt, CodePlan, temperature=0.0, timeout_s=budget.call_timeout_s())
            budget.charge(llm_result.total_tokens)
        except Exception:
            return None
        if not plan.code or not plan.inputs:
            return None
        inputs = {ci.name: ci.value for ci in plan.inputs}
        tool_res = await asyncio.to_thread(   # subprocess wait must not stall the loop (C9)
            self.deps.toolrunner.run, plan.code, inputs, timeout_s=self.deps.settings.sandbox.timeout_s)
        if not tool_res.ok:
            return None
        return Computation(
            comp_id="k_" + uuid.uuid4().hex[:6],
            inputs=[ComputationInput(name=ci.name, value=ci.value, source_chunk_id=ci.source_chunk_id,
                                     cell_ref=ci.cell_ref) for ci in plan.inputs],
            code=plan.code, result=tool_res.result, sandbox_run_id=tool_res.run_id,
        )
