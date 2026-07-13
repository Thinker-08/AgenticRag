from __future__ import annotations

import re
import uuid

from langgraph.graph import END, StateGraph

from ..contracts import Answer, AnswerStatus, Grade, GradeVerdict, Intent, QueryPlan, Route, Strategy, SubStep, SupportLabel
from ..deps import Deps
from ..grounding.answer_builder import abstain, buildAnswer
from ..grounding.generator import Generator
from ..grounding.verify import GroundednessVerifier
from .plan_exec import PlanExecutor
from .schemas import PlanCritique, RewriteResult
from .state import AgentState

_REFORMULATE = {Strategy.HYBRID: Strategy.BM25, Strategy.SEMANTIC: Strategy.HYBRID, Strategy.BM25: Strategy.HYBRID}

_ENTITY = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b")
_ENTITY_STOP = {"What", "Which", "How", "Who", "When", "Where", "Why", "Tell", "Compare", "Summarize", "List", "The", "In", "On", "For", "FY"}


_STOP = {"what", "which", "how", "who", "when", "where", "the", "a", "an", "of", "to", "in", "on", "and", "or", "is", "are", "was", "were", "for", "with", "as", "at", "by", "from", "did", "does", "about", "its", "their", "tell", "me", "compare", "vs"}


def contentTokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1}


def extractEntities(text: str) -> list[str]:
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
        self.app = self.build()

    async def nContextualize(self, state: AgentState) -> dict:
        history = state.get("history") or []
        if not history:
            ents = extractEntities(state["query"])
            dangling = bool(re.search(r"\b(it|its|they|their|those|these)\b", state["query"], re.I)) and not ents
            return {"standalone_q": state["query"], "carried_entities": ents, "clarify": (f"Could you clarify what '{state['query'].strip()}' refers to?" if dangling else "")}

        last = next((t for t in reversed(history) if t.role == "assistant"), None)
        entities = last.carried_entities if last else []
        prompt = f"<entities>{', '.join(entities)}</entities>\n<query>{state['query']}</query>\nRewrite the query into a standalone question, resolving pronouns/ellipsis."
        with self.deps.tracer.span("contextualize"):
            rewrite, llm_result = await self.deps.small_llm.generateStructured(prompt, RewriteResult, timeout_s=state["budget"].callTimeoutS())
            state["budget"].charge(llm_result.totalTokens)

        clarify = "" if rewrite.resolved else (f"Could you clarify what '{state['query'].strip()}' refers to?")
        return {"standalone_q": rewrite.standalone_query or state["query"], "carried_entities": rewrite.carried_entities, "clarify": clarify}

    async def nClarify(self, state: AgentState) -> dict:
        self.deps.tracer.event("clarify", question=state["clarify"])
        ans = abstain(state["trace_id"], "needs_clarification")
        ans.answer_text = state["clarify"]
        return {"answer": ans}

    async def nClassify(self, state: AgentState) -> dict:
        with self.deps.tracer.span("classify"):
            route, llm_result = await self.deps.small_llm.generateStructured(f"<query>{state['standalone_q']}</query>", Route, timeout_s=state["budget"].callTimeoutS())
            state["budget"].charge(llm_result.totalTokens)

        self.deps.tracer.event("route", intent=route.intent, retrieve=route.needs_retrieval)
        return {"route": route}

    async def nRespond(self, state: AgentState) -> dict:
        system = "You are a document-grounded QA assistant. Reply briefly and conversationally. Never state facts about documents or the world here — invite a document question instead."
        with self.deps.tracer.span("respond"):
            llm_result = await self.deps.llm.generate(f"<query>{state['standalone_q']}</query>", system=system, timeout_s=state["budget"].callTimeoutS())

        cites = []
        if state["route"].history_answerable:
            last = next((t for t in reversed(state.get("history") or []) if t.role == "assistant"), None)
            cites = last.citations if last else []

        ans = Answer(answer_id="ans_" + uuid.uuid4().hex[:8], trace_id=state["trace_id"], status=AnswerStatus.ANSWERED, answer_text=llm_result.text)
        if cites:
            from ..contracts import Claim

            ans.claims = [Claim(claim_id="c1", text=llm_result.text, citations=cites, support=SupportLabel.SUPPORTED)]
        return {"answer": ans}

    async def nPlan(self, state: AgentState) -> dict:
        prompt = f"<query_id>{uuid.uuid4().hex[:8]}</query_id><trace_id>{state['trace_id']}</trace_id><query>{state['standalone_q']}</query>"
        with self.deps.tracer.span("plan"):
            plan, llm_result = await self.deps.small_llm.generateStructured(prompt, QueryPlan, timeout_s=state["budget"].callTimeoutS())
            state["budget"].charge(llm_result.totalTokens)

        plan = self.sanitizePlan(plan, state["standalone_q"])
        plan = await self.critiquePlan(plan, state["budget"])

        self.deps.tracer.event("plan", steps=len(plan.sub_steps), merge=plan.merge)
        return {"plan": plan}

    async def critiquePlan(self, plan: QueryPlan, budget) -> QueryPlan:
        if len(plan.sub_steps) <= 1:
            return plan

        listing = "\n".join(f"{s.step_id}: tool={s.tool.value} q={s.query}" for s in plan.sub_steps)
        try:
            critique, meta = await self.deps.small_llm.generateStructured(f"Review this retrieval plan for redundant steps:\n{listing}", PlanCritique, timeout_s=budget.callTimeoutS())
            budget.charge(meta.totalTokens)
        except Exception:
            return plan

        drop = set(critique.redundant_step_ids)
        needed = {d for s in plan.sub_steps if s.step_id not in drop for d in s.depends_on}
        drop -= needed
        kept = [s for s in plan.sub_steps if s.step_id not in drop] or plan.sub_steps

        if len(kept) != len(plan.sub_steps):
            self.deps.tracer.event("plan.critique", dropped=sorted(drop))
        return plan.model_copy(update={"sub_steps": kept})

    def sanitizePlan(self, plan: QueryPlan, fallback_q: str) -> QueryPlan:
        steps = plan.sub_steps[:8]
        known = {st.step_id for st in steps}
        steps = [st.model_copy(update={"k": max(1, min(st.k, 50)), "depends_on": [d for d in st.depends_on if d in known and d != st.step_id]}) for st in steps]

        if not steps:
            steps = [SubStep(step_id="s1", tool=Strategy.HYBRID, query=fallback_q, k=self.deps.settings.retrieval.top_k)]
        return plan.model_copy(update={"sub_steps": steps})

    async def nRetrieve(self, state: AgentState) -> dict:
        with self.deps.tracer.span("retrieve", attempt=self.deps.settings.agent.max_iters - state["budget"].iters_left):
            evidence, comps, gaps = await self.executor.run(state["plan"], tenant_id=state["tenant_id"], budget=state["budget"])

        prior = state.get("computations") or []
        return {"evidence": evidence, "computations": comps or prior, "gaps": gaps}

    async def nGrade(self, state: AgentState) -> dict:
        comps = state.get("computations") or []
        if state["evidence"].scored and any(c.result is not None for c in comps):
            grade = Grade(verdict=GradeVerdict.SUFFICIENT, max_relevance=1.0, rationale="tool result")
            self.deps.tracer.event("grade", verdict=grade.verdict, relevance=1.0, via="tool")
            return {"grade": grade}

        step = SubStep(step_id="grade", tool=Strategy.HYBRID, query=state["standalone_q"])
        with self.deps.tracer.span("grade"):
            grade = await self.deps.grader.grade(step, state["evidence"].scored, budget=state["budget"])

        self.deps.tracer.event("grade", verdict=grade.verdict, relevance=grade.max_relevance)
        return {"grade": grade}

    async def nReformulate(self, state: AgentState) -> dict:
        state["budget"].consumeIter()

        grade = state.get("grade")
        verdict = grade.verdict if grade else GradeVerdict.AMBIGUOUS
        missing_keyword = grade.missing_slots[0] if grade and grade.missing_slots else ""
        hyde = ""
        if verdict == GradeVerdict.IRRELEVANT:
            hyde = await self.hyde(state["standalone_q"], state["budget"])

        new_steps = []
        for st in state["plan"].sub_steps:
            tool = _REFORMULATE.get(st.tool, st.tool)
            q = st.query
            if missing_keyword and missing_keyword not in q.lower():
                q = f"{q} {missing_keyword}"
            if hyde:
                q = f"{q} {hyde}"
            new_steps.append(st.model_copy(update={"tool": tool, "query": q}))

        self.deps.tracer.event("reformulate", iters_left=state["budget"].iters_left, verdict=verdict, hyde=bool(hyde))
        return {"plan": state["plan"].model_copy(update={"sub_steps": new_steps})}

    async def hyde(self, query: str, budget) -> str:
        prompt = f"<query>{query}</query>\nWrite a one-sentence hypothetical answer to this question as it might appear in a document. Output only that sentence."

        try:
            res = await self.deps.small_llm.generate(prompt, max_tokens=80, timeout_s=budget.callTimeoutS())
            budget.charge(res.totalTokens)
            return res.text.strip()[:200]
        except Exception:
            return ""

    async def nGenerate(self, state: AgentState) -> dict:
        with self.deps.tracer.span("generate"):
            draft = await self.generator.generate(state["standalone_q"], state["evidence"], intent=state["route"].intent, trace_id=state["trace_id"], budget=state["budget"], computations=state.get("computations"))
        return {"draft": draft}

    async def nVerify(self, state: AgentState) -> dict:
        with self.deps.tracer.span("verify", claims=len(state["draft"].claims)):
            claims = await self.verifier.verify(state["draft"].claims, state["evidence"], budget=state["budget"])

        supported = sum(1 for c in claims if c.support == SupportLabel.SUPPORTED)
        useful = self.useful(state["standalone_q"], claims, state.get("computations") or [])
        self.deps.tracer.event("verify", supported=supported, total=len(claims), useful=useful)
        return {"claims": claims, "useful": useful}

    @staticmethod
    def useful(query: str, claims: list, computations: list) -> bool:
        if computations:
            return True

        q = contentTokens(query)
        if not q:
            return True

        ans = set()
        for c in claims:
            if c.support == SupportLabel.SUPPORTED:
                ans |= contentTokens(c.text)

        return bool(q & ans)

    async def nFinalize(self, state: AgentState) -> dict:
        ans = buildAnswer(state["standalone_q"], state["route"].intent, state.get("claims") or [], trace_id=state["trace_id"], computations=state.get("computations"), gaps=state.get("gaps"))
        draft = state.get("draft")
        if draft is not None and draft.degraded:
            ans.degraded = True
        return {"answer": ans}

    async def nAbstain(self, state: AgentState) -> dict:
        reason = self.abstainReason(state)
        self.deps.tracer.event("abstain", reason=reason)
        return {"answer": abstain(state["trace_id"], reason, gaps=state.get("gaps"))}

    def routeContextualize(self, state: AgentState) -> str:
        return "clarify" if state.get("clarify") else "classify"

    def routeClassify(self, state: AgentState) -> str:
        r: Route = state["route"]
        return "respond" if (r.intent == Intent.CHITCHAT or r.history_answerable) else "plan"

    def budgetOk(self, state: AgentState) -> bool:
        return state["budget"].iters_left > 0 and not state["budget"].exceeded()

    def routeGrade(self, state: AgentState) -> str:
        if state["grade"].verdict == GradeVerdict.SUFFICIENT:
            return "generate"
        return "reformulate" if self.budgetOk(state) else "abstain"

    def routeVerify(self, state: AgentState) -> str:
        grounded = any(c.support == SupportLabel.SUPPORTED for c in (state.get("claims") or []))
        if grounded and state.get("useful", True):
            return "finalize"
        return "reformulate" if self.budgetOk(state) else "abstain"

    def abstainReason(self, state: AgentState) -> str:
        if any(c.support == SupportLabel.CONTRADICTED for c in (state.get("claims") or [])):
            return "contradicted"

        grade = state.get("grade")
        evidence = state.get("evidence")
        no_evidence = evidence is None or not evidence.scored or (grade is not None and grade.verdict == GradeVerdict.IRRELEVANT)
        if no_evidence:
            return "no_evidence"
        return "budget_abstain"

    def build(self):
        g = StateGraph(AgentState)

        g.add_node("contextualize", self.nContextualize)
        g.add_node("clarify", self.nClarify)
        g.add_node("classify", self.nClassify)
        g.add_node("respond", self.nRespond)
        g.add_node("plan", self.nPlan)
        g.add_node("retrieve", self.nRetrieve)
        g.add_node("grade", self.nGrade)
        g.add_node("reformulate", self.nReformulate)
        g.add_node("generate", self.nGenerate)
        g.add_node("verify", self.nVerify)
        g.add_node("finalize", self.nFinalize)
        g.add_node("abstain", self.nAbstain)

        g.set_entry_point("contextualize")
        g.add_conditional_edges("contextualize", self.routeContextualize, {"clarify": "clarify", "classify": "classify"})
        g.add_edge("clarify", END)
        g.add_conditional_edges("classify", self.routeClassify, {"respond": "respond", "plan": "plan"})
        g.add_edge("respond", END)
        g.add_edge("plan", "retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", self.routeGrade, {"generate": "generate", "reformulate": "reformulate", "abstain": "abstain"})
        g.add_edge("reformulate", "retrieve")
        g.add_edge("generate", "verify")
        g.add_conditional_edges("verify", self.routeVerify, {"finalize": "finalize", "reformulate": "reformulate", "abstain": "abstain"})
        g.add_edge("finalize", END)
        g.add_edge("abstain", END)

        return g.compile()
