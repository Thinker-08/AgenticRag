"""The agent as an explicit finite state machine (05 §1, C23).

Named states, typed transitions, terminal conditions — a bounded-cycle graph that provably halts.
Two corrective cycles share one budget: Grade→Reformulate→Retrieve (CRAG) and Verify→Reformulate→Retrieve
(Self-RAG). Reformulate strictly decrements iters_left, so total LLM calls and wall-time are both bounded.
"""

from __future__ import annotations

import re
import uuid

from langgraph.graph import END, StateGraph

from ..contracts import (
    Answer,
    AnswerStatus,
    Grade,
    GradeVerdict,
    Intent,
    QueryPlan,
    Route,
    Strategy,
    SubStep,
    SupportLabel,
)
from ..deps import Deps
from ..grounding.answer_builder import abstain, build_answer
from ..grounding.generator import Generator
from ..grounding.verify import GroundednessVerifier
from .plan_exec import PlanExecutor
from .schemas import RewriteResult
from .state import AgentState

_REFORMULATE = {
    Strategy.HYBRID: Strategy.BM25,
    Strategy.SEMANTIC: Strategy.HYBRID,
    Strategy.BM25: Strategy.HYBRID,
}

_ENTITY = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b")
_ENTITY_STOP = {"What", "Which", "How", "Who", "When", "Where", "Why", "Tell", "Compare",
                "Summarize", "List", "The", "In", "On", "For", "FY"}


def _entities(text: str) -> list[str]:
    """Naive proper-noun spans for session carry-over; a real NER slots in behind the same shape."""
    out: list[str] = []
    for m in _ENTITY.finditer(text):
        cleaned = " ".join(w for w in m.group(1).split() if w not in _ENTITY_STOP)
        if cleaned and not cleaned.isdigit():
            out.append(cleaned)
    seen: set[str] = set()
    return [e for e in out if not (e.lower() in seen or seen.add(e.lower()))][:6]


class AgentGraph:
    def __init__(self, deps: Deps) -> None:
        self.deps = deps
        self.executor = PlanExecutor(deps)
        self.generator = Generator(deps)
        self.verifier = GroundednessVerifier(deps)
        self.app = self._build()

    async def n_contextualize(self, state: AgentState) -> dict:
        history = state.get("history") or []
        if not history:
            return {"standalone_q": state["query"], "carried_entities": _entities(state["query"])}
        last = next((t for t in reversed(history) if t.role == "assistant"), None)
        entities = last.carried_entities if last else []
        prompt = (
            f"<entities>{', '.join(entities)}</entities>\n<query>{state['query']}</query>\n"
            "Rewrite the query into a standalone question, resolving pronouns/ellipsis."
        )
        with self.deps.tracer.span("contextualize"):
            rewrite, llm_result = await self.deps.small_llm.generate_structured(
                prompt, RewriteResult, timeout_s=state["budget"].call_timeout_s()
            )
            state["budget"].charge(llm_result.total_tokens)
        return {
            "standalone_q": rewrite.standalone_query or state["query"],
            "carried_entities": rewrite.carried_entities,
        }

    async def n_classify(self, state: AgentState) -> dict:
        with self.deps.tracer.span("classify"):
            route, llm_result = await self.deps.small_llm.generate_structured(
                f"<query>{state['standalone_q']}</query>",
                Route,
                timeout_s=state["budget"].call_timeout_s(),
            )
            state["budget"].charge(llm_result.total_tokens)
        self.deps.tracer.event("route", intent=route.intent, retrieve=route.needs_retrieval)
        return {"route": route}

    async def n_respond(self, state: AgentState) -> dict:
        system = (
            "You are a document-grounded QA assistant. Reply briefly and conversationally. "
            "Never state facts about documents or the world here — invite a document question instead."
        )
        with self.deps.tracer.span("respond"):
            llm_result = await self.deps.llm.generate(
                f"<query>{state['standalone_q']}</query>",
                system=system,
                timeout_s=state["budget"].call_timeout_s(),
            )
        cites = []
        if state["route"].history_answerable:
            last = next(
                (t for t in reversed(state.get("history") or []) if t.role == "assistant"), None
            )
            cites = last.citations if last else []
        ans = Answer(
            answer_id="ans_" + uuid.uuid4().hex[:8],
            trace_id=state["trace_id"],
            status=AnswerStatus.ANSWERED,
            answer_text=llm_result.text,
        )
        if cites:
            from ..contracts import Claim

            ans.claims = [
                Claim(
                    claim_id="c1",
                    text=llm_result.text,
                    citations=cites,
                    support=SupportLabel.SUPPORTED,
                )
            ]
        return {"answer": ans}

    async def n_plan(self, state: AgentState) -> dict:
        prompt = (
            f"<query_id>{uuid.uuid4().hex[:8]}</query_id><trace_id>{state['trace_id']}</trace_id>"
            f"<query>{state['standalone_q']}</query>"
        )
        with self.deps.tracer.span("plan"):
            plan, llm_result = await self.deps.small_llm.generate_structured(
                prompt, QueryPlan, timeout_s=state["budget"].call_timeout_s()
            )
            state["budget"].charge(llm_result.total_tokens)
        plan = self._sanitize_plan(plan, state["standalone_q"])
        self.deps.tracer.event("plan", steps=len(plan.sub_steps), merge=plan.merge)
        return {"plan": plan}

    def _sanitize_plan(self, plan: QueryPlan, fallback_q: str) -> QueryPlan:
        """Constrained decoding pins types, not ranges: clamp k, cap fan-out, drop unknown deps."""
        steps = plan.sub_steps[:8]
        known = {st.step_id for st in steps}
        steps = [
            st.model_copy(
                update={
                    "k": max(1, min(st.k, 50)),
                    "depends_on": [d for d in st.depends_on if d in known and d != st.step_id],
                }
            )
            for st in steps
        ]
        if not steps:
            steps = [
                SubStep(
                    step_id="s1",
                    tool=Strategy.HYBRID,
                    query=fallback_q,
                    k=self.deps.settings.retrieval.top_k,
                )
            ]
        return plan.model_copy(update={"sub_steps": steps})

    async def n_retrieve(self, state: AgentState) -> dict:
        with self.deps.tracer.span(
            "retrieve", attempt=self.deps.settings.agent.max_iters - state["budget"].iters_left
        ):
            evidence, comps, gaps = await self.executor.run(
                state["plan"], tenant_id=state["tenant_id"], budget=state["budget"]
            )
        prior = state.get("computations") or []
        return {"evidence": evidence, "computations": comps or prior, "gaps": gaps}

    async def n_grade(self, state: AgentState) -> dict:
        # A computed result (aggregation count / comparison arithmetic) IS the answer — its
        # source chunks (bare list items) need not lexically overlap the query, so the code tool
        # succeeding short-circuits the relevance grade (05 §8).
        comps = state.get("computations") or []
        if state["evidence"].scored and any(c.result is not None for c in comps):
            grade = Grade(verdict=GradeVerdict.SUFFICIENT, max_relevance=1.0, rationale="tool result")
            self.deps.tracer.event("grade", verdict=grade.verdict, relevance=1.0, via="tool")
            return {"grade": grade}
        step = SubStep(step_id="grade", tool=Strategy.HYBRID, query=state["standalone_q"])
        with self.deps.tracer.span("grade"):
            grade = await self.deps.grader.grade(
                step, state["evidence"].scored, budget=state["budget"]
            )
        self.deps.tracer.event("grade", verdict=grade.verdict, relevance=grade.max_relevance)
        return {"grade": grade}

    async def n_reformulate(self, state: AgentState) -> dict:
        state["budget"].consume_iter()
        grade = state.get("grade")
        verdict = grade.verdict if grade else GradeVerdict.AMBIGUOUS
        missing_keyword = grade.missing_slots[0] if grade and grade.missing_slots else ""
        # CRAG reformulation ladder (05 §5): AMBIGUOUS -> targeted keyword rewrite; IRRELEVANT ->
        # switch strategy (dense->HyDE, then hybrid->BM25). Reformulate MUST change something each
        # iteration, or an identical re-run burns a budgeted iteration for no new evidence.
        hyde = ""
        if verdict == GradeVerdict.IRRELEVANT:
            hyde = await self._hyde(state["standalone_q"], state["budget"])
        new_steps = []
        for st in state["plan"].sub_steps:
            tool = _REFORMULATE.get(st.tool, st.tool)
            q = st.query
            if missing_keyword and missing_keyword not in q.lower():
                q = f"{q} {missing_keyword}"
            if hyde:
                q = f"{q} {hyde}"
            new_steps.append(st.model_copy(update={"tool": tool, "query": q}))
        self.deps.tracer.event(
            "reformulate", iters_left=state["budget"].iters_left, verdict=verdict, hyde=bool(hyde))
        return {"plan": state["plan"].model_copy(update={"sub_steps": new_steps})}

    async def _hyde(self, query: str, budget) -> str:
        """HyDE: draft a hypothetical answer and retrieve against it (04 §4). The hypothetical is
        used ONLY as a retrieval probe — never surfaced or cited — so a wrong guess can't leak in."""
        prompt = (f"<query>{query}</query>\nWrite a one-sentence hypothetical answer to this "
                  "question as it might appear in a document. Output only that sentence.")
        try:
            res = await self.deps.small_llm.generate(
                prompt, max_tokens=80, timeout_s=budget.call_timeout_s())
            budget.charge(res.total_tokens)
            return res.text.strip()[:200]
        except Exception:
            return ""

    async def n_generate(self, state: AgentState) -> dict:
        with self.deps.tracer.span("generate"):
            draft = await self.generator.generate(
                state["standalone_q"],
                state["evidence"],
                intent=state["route"].intent,
                trace_id=state["trace_id"],
                budget=state["budget"],
                computations=state.get("computations"),
            )
        return {"draft": draft}

    async def n_verify(self, state: AgentState) -> dict:
        with self.deps.tracer.span("verify", claims=len(state["draft"].claims)):
            claims = await self.verifier.verify(
                state["draft"].claims, state["evidence"], budget=state["budget"]
            )
        supported = sum(1 for c in claims if c.support == SupportLabel.SUPPORTED)
        self.deps.tracer.event("verify", supported=supported, total=len(claims))
        return {"claims": claims}

    async def n_finalize(self, state: AgentState) -> dict:
        ans = build_answer(
            state["standalone_q"],
            state["route"].intent,
            state.get("claims") or [],
            trace_id=state["trace_id"],
            computations=state.get("computations"),
            gaps=state.get("gaps"),
        )
        draft = state.get("draft")
        if draft is not None and draft.degraded:
            ans.degraded = True                       # surface the fallback tier to the client (C14)
        return {"answer": ans}

    async def n_abstain(self, state: AgentState) -> dict:
        reason = self._abstain_reason(state)
        self.deps.tracer.event("abstain", reason=reason)
        return {"answer": abstain(state["trace_id"], reason, gaps=state.get("gaps"))}

    def route_classify(self, state: AgentState) -> str:
        r: Route = state["route"]
        return "respond" if (r.intent == Intent.CHITCHAT or r.history_answerable) else "plan"

    def _budget_ok(self, state: AgentState) -> bool:
        return state["budget"].iters_left > 0 and not state["budget"].exceeded()

    def route_grade(self, state: AgentState) -> str:
        if state["grade"].verdict == GradeVerdict.SUFFICIENT:
            return "generate"
        return "reformulate" if self._budget_ok(state) else "abstain"

    def route_verify(self, state: AgentState) -> str:
        grounded = any(c.support == SupportLabel.SUPPORTED for c in (state.get("claims") or []))
        if grounded:
            return "finalize"
        return "reformulate" if self._budget_ok(state) else "abstain"

    def _abstain_reason(self, state: AgentState) -> str:
        if any(c.support == SupportLabel.CONTRADICTED for c in (state.get("claims") or [])):
            return "contradicted"
        grade = state.get("grade")
        evidence = state.get("evidence")
        no_evidence = (
            evidence is None
            or not evidence.scored
            or (grade is not None and grade.verdict == GradeVerdict.IRRELEVANT)
        )
        if no_evidence:
            return "no_evidence"
        return "budget_abstain"

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("contextualize", self.n_contextualize)
        g.add_node("classify", self.n_classify)
        g.add_node("respond", self.n_respond)
        g.add_node("plan", self.n_plan)
        g.add_node("retrieve", self.n_retrieve)
        g.add_node("grade", self.n_grade)
        g.add_node("reformulate", self.n_reformulate)
        g.add_node("generate", self.n_generate)
        g.add_node("verify", self.n_verify)
        g.add_node("finalize", self.n_finalize)
        g.add_node("abstain", self.n_abstain)

        g.set_entry_point("contextualize")
        g.add_edge("contextualize", "classify")
        g.add_conditional_edges(
            "classify", self.route_classify, {"respond": "respond", "plan": "plan"}
        )
        g.add_edge("respond", END)
        g.add_edge("plan", "retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges(
            "grade",
            self.route_grade,
            {"generate": "generate", "reformulate": "reformulate", "abstain": "abstain"},
        )
        g.add_edge("reformulate", "retrieve")
        g.add_edge("generate", "verify")
        g.add_conditional_edges(
            "verify",
            self.route_verify,
            {"finalize": "finalize", "reformulate": "reformulate", "abstain": "abstain"},
        )
        g.add_edge("finalize", END)
        g.add_edge("abstain", END)
        return g.compile()
