"""AgentApp — the public query-plane entry point wrapping the compiled FSM."""

from __future__ import annotations

import uuid

from ..contracts import TERMINAL_STATES, Answer, AnswerStatus, Budget, Turn
from ..deps import Deps
from ..grounding.answer_builder import abstain
from ..promptfmt import nonce_for
from ..security.output_filter import scan_answer
from .answer_cache import AnswerCache
from .graph import AgentGraph
from .state import AgentState


class AgentApp:
    def __init__(self, deps: Deps) -> None:
        self.deps = deps
        self.graph = AgentGraph(deps)
        self.cache = AnswerCache(deps)

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
        session_id: str | None = None,
        budget: Budget | None = None,
    ) -> Answer:
        # server-side session (C16): stateless replica reconstructs history from the store
        if session_id and history is None:
            convo = await self.deps.sessions.get(tenant_id, session_id)
            history = convo.window(6)
        history = history or []

        async def _compute() -> Answer:
            return await self._run_once(query, history, tenant_id=tenant_id, budget=budget)

        answer = await self.cache.resolve(tenant_id, query, history, _compute)
        if session_id:
            await self._persist_turns(tenant_id, session_id, query, answer)
        return answer

    async def _run_once(
        self, query: str, history: list[Turn], *, tenant_id: str, budget: Budget | None
    ) -> Answer:
        trace_id = uuid.uuid4().hex[:12]
        resolved_budget = self._budget(budget)
        state: AgentState = {
            "query": query,
            "tenant_id": tenant_id,
            "trace_id": trace_id,
            "history": history,
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
            answer.carried_entities = final.get("carried_entities") or []
            violation = scan_answer(answer, nonce=nonce_for(trace_id))
            if violation is not None:
                # injection fingerprint in the output = failed verification: fail closed (07 §6)
                self.deps.tracer.event("output_filter.blocked", violation=violation)
                answer = abstain(trace_id, "injection_suspected")
            answer = await self._flag_still_indexing(answer, tenant_id)
        return answer

    async def _persist_turns(self, tenant_id: str, session_id: str, query: str, answer: Answer) -> None:
        await self.deps.sessions.append(tenant_id, session_id, Turn(role="user", content=query))
        await self.deps.sessions.append(
            tenant_id,
            session_id,
            Turn(role="assistant", content=answer.answer_text, citations=answer.sources(),
                 carried_entities=answer.carried_entities),
        )

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
