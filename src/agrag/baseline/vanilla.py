"""Vanilla single-shot RAG — the frozen control (page 01, page 13 step 1).

extract → fixed chunk → embed → top-k → one grounded prompt → answer. No routing, no grading, no
verification, no abstention. Built behind the SAME interfaces as the agent so it swaps in without
touching callers; every phase reports the agent's quantified delta over this baseline (C22, C27).
"""

from __future__ import annotations

import uuid

from ..contracts import (
    Answer,
    AnswerStatus,
    Budget,
    Claim,
    Evidence,
    Intent,
    Strategy,
    SupportLabel,
    Turn,
)
from ..deps import Deps
from ..grounding.generator import Generator


class BaselineRAG:
    def __init__(self, deps: Deps) -> None:
        self.deps = deps
        self.generator = Generator(deps)

    def _budget(self, budget: Budget | None) -> Budget:
        agent_cfg = self.deps.settings.agent
        return budget or Budget.start(agent_cfg.wall_clock_s, agent_cfg.token_budget, agent_cfg.max_iters)

    async def answer(
        self,
        query: str,
        history: list[Turn] | None = None,
        *,
        tenant_id: str = "default",
        budget: Budget | None = None,
    ) -> Answer:
        trace_id = uuid.uuid4().hex[:12]
        resolved_budget = self._budget(budget)
        with self.deps.tracer.start_trace("answer", trace_id=trace_id, tenant_id=tenant_id, mode="baseline"):
            docs = await self.deps.retriever.retrieve(
                query, tenant_id=tenant_id, strategy=Strategy.SEMANTIC, k=5, budget=resolved_budget)
            evidence = Evidence(scored=docs)
            draft = await self.generator.generate(query, evidence, intent=Intent.FACTOID,
                                                   trace_id=trace_id, budget=resolved_budget)
        claims = [Claim(claim_id=f"c{i}", text=dc.text, citations=dc.citations, support=SupportLabel.SUPPORTED)
                  for i, dc in enumerate(draft.claims, start=1)]
        text = draft.answer_text or " ".join(c.text for c in claims)
        return Answer(answer_id="ans_" + uuid.uuid4().hex[:8], trace_id=trace_id,
                      status=AnswerStatus.ANSWERED, answer_text=text, claims=claims)
