from __future__ import annotations

import uuid

from ..contracts import TERMINAL_STATES, Answer, AnswerStatus, Budget, Turn
from ..deps import Deps
from ..grounding.answer_builder import abstain
from ..promptfmt import nonceFor
from ..security.output_filter import scanAnswer
from .answer_cache import AnswerCache
from .graph import AgentGraph
from .state import AgentState


class AgentApp:
    def __init__(self, deps: Deps) -> None:
        self.deps = deps
        self.graph = AgentGraph(deps)
        self.cache = AnswerCache(deps)

    def budget(self, budget: Budget | None) -> Budget:
        agent_cfg = self.deps.settings.agent
        return budget or Budget.start(agent_cfg.wall_clock_s, agent_cfg.token_budget, agent_cfg.max_iters)

    def recursionLimit(self, budget: Budget) -> int:
        return max(25, 12 + 5 * max(0, budget.iters_left))

    async def answer(self, query: str, history: list[Turn] | None = None, *, tenant_id: str = "default", session_id: str | None = None, budget: Budget | None = None) -> Answer:
        if session_id and history is None:
            convo = await self.deps.sessions.get(tenant_id, session_id)
            history = convo.window(6)
        history = history or []

        async def compute() -> Answer:
            return await self.runOnce(query, history, tenant_id=tenant_id, budget=budget)

        answer = await self.cache.resolve(tenant_id, query, history, compute)
        if session_id:
            await self.persistTurns(tenant_id, session_id, query, answer)
        return answer

    async def runOnce(self, query: str, history: list[Turn], *, tenant_id: str, budget: Budget | None) -> Answer:
        trace_id = uuid.uuid4().hex[:12]
        resolved_budget = self.budget(budget)
        state: AgentState = {"query": query, "tenant_id": tenant_id, "trace_id": trace_id, "history": history, "budget": resolved_budget, "computations": [], "gaps": []}

        with self.deps.tracer.startTrace("answer", trace_id=trace_id, tenant_id=tenant_id, mode="agentic"):
            final = await self.graph.app.ainvoke(state, config={"recursion_limit": self.recursionLimit(resolved_budget)})
            answer = final["answer"]
            answer.carried_entities = final.get("carried_entities") or []

            violation = scanAnswer(answer, nonce=nonceFor(trace_id))
            if violation is not None:
                self.deps.tracer.event("output_filter.blocked", violation=violation)
                answer = abstain(trace_id, "injection_suspected")

            answer = await self.flagStillIndexing(answer, tenant_id)
        return answer

    async def persistTurns(self, tenant_id: str, session_id: str, query: str, answer: Answer) -> None:
        await self.deps.sessions.append(tenant_id, session_id, Turn(role="user", content=query))
        await self.deps.sessions.append(tenant_id, session_id, Turn(role="assistant", content=answer.answer_text, citations=answer.sources(), carried_entities=answer.carried_entities))

    async def flagStillIndexing(self, answer: Answer, tenant_id: str) -> Answer:
        if answer.status != AnswerStatus.ABSTAINED or answer.abstention_reason != "no_evidence":
            return answer

        try:
            docs = await self.deps.docstore.listDocs(tenant_id)
        except Exception:
            return answer

        pending = [d for d in docs if d.status not in TERMINAL_STATES]
        if not pending:
            return answer

        notes = [f"{d.filename or d.doc_id} is still indexing ({d.status.value})" for d in pending]
        answer.abstention_reason = "still_indexing"
        answer.gaps = list(answer.gaps) + notes
        answer.answer_text = "The document(s) are still being indexed — please retry shortly. " + "; ".join(notes)

        return answer
