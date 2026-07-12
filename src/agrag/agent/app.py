"""AgentApp — the public query-plane entry point wrapping the compiled FSM."""

from __future__ import annotations

import uuid

from ..contracts import TERMINAL_STATES, Answer, AnswerStatus, Budget, Turn
from ..deps import Deps
from .graph import AgentGraph
from .state import AgentState


class AgentApp:
    def __init__(self, deps: Deps) -> None:
        self.deps = deps
        self.graph = AgentGraph(deps)

    def _budget(self, budget: Budget | None) -> Budget:
        agent_cfg = self.deps.settings.agent
        return budget or Budget.start(
            agent_cfg.wall_clock_s, agent_cfg.token_budget, agent_cfg.max_iters
        )

    def _recursion_limit(self, budget: Budget) -> int:
        return max(25, 12 + 5 * max(0, budget.iters_left))

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
        state: AgentState = {
            "query": query,
            "tenant_id": tenant_id,
            "trace_id": trace_id,
            "history": history or [],
            "budget": resolved_budget,
            "computations": [],
            "gaps": [],
        }
        with self.deps.tracer.start_trace(
            "answer", trace_id=trace_id, tenant_id=tenant_id, mode="agentic"
        ):
            final = await self.graph.app.ainvoke(
                state, config={"recursion_limit": self._recursion_limit(resolved_budget)}
            )
            answer = final["answer"]
            answer = await self._flag_still_indexing(answer, tenant_id)
        return answer

    async def _flag_still_indexing(self, answer: Answer, tenant_id: str) -> Answer:
        """Read-your-writes (C16): an abstention while docs are still indexing must say so —
        'not stated in the document' would misreport index lag as a document gap."""
        if answer.status != AnswerStatus.ABSTAINED or answer.abstention_reason != "no_evidence":
            return answer
        try:
            docs = await self.deps.docstore.list_docs(tenant_id)
        except Exception:
            return answer
        pending = [d for d in docs if d.status not in TERMINAL_STATES]
        if not pending:
            return answer
        notes = [f"{d.filename or d.doc_id} is still indexing ({d.status.value})" for d in pending]
        answer.abstention_reason = "still_indexing"
        answer.gaps = list(answer.gaps) + notes
        answer.answer_text = (
            "The document(s) are still being indexed — please retry shortly. " + "; ".join(notes)
        )
        return answer
