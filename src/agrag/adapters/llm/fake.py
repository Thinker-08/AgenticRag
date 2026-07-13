from __future__ import annotations

import re
from typing import Sequence, Type, TypeVar

from pydantic import BaseModel

from ...interfaces.types import LLMResult
from ...promptfmt import extractTag, parseEvidenceBlocks, sentences

T = TypeVar("T", bound=BaseModel)

_WORD = re.compile(r"[a-z0-9]+")
_AGG = ("how many", "list all", "count", "number of", "enumerate", "which of")
_SUM = ("summarize", "summary", "overview", "themes", "risk factors")
_CMP = (" vs ", "versus", "compare", "difference between", "compared to")
_CHIT = ("thanks", "thank you", "hello", "hi ", "what can you do", "who are you")


def toks(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def queryOf(prompt: str) -> str:
    for tag in ("query", "question", "standalone_query"):
        v = extractTag(prompt, tag)
        if v:
            return v

    return prompt.strip().splitlines()[-1] if prompt.strip() else ""


class FakeLLM:
    def __init__(self, model: str = "fake") -> None:
        self.name = model
        self.model = model

    async def generate(self, prompt: str, *, system: str | None = None, max_tokens: int = 512, temperature: float = 0.0, images: Sequence[bytes] | None = None, timeout_s: float | None = None) -> LLMResult:
        q = queryOf(prompt)
        low = q.lower()

        if "context" in (system or "").lower() and "situate" in prompt.lower():
            text = f"This passage concerns: {q[:80]}."
        elif any(w in low for w in _CHIT):
            text = "I answer questions strictly from the uploaded documents, with citations. Ask about their content."
        elif "reformulate" in (system or "").lower() or "rewrite" in prompt.lower():
            text = q + " (exact terms, figures)"
        else:
            text = q

        return LLMResult(text=text, prompt_tokens=len(toks(prompt)), completion_tokens=len(toks(text)), model=self.model)

    async def generateStructured(self, prompt: str, schema: Type[T], *, system: str | None = None, max_tokens: int = 512, temperature: float = 0.0, images: Sequence[bytes] | None = None, timeout_s: float | None = None) -> tuple[T, LLMResult]:
        name = schema.__name__
        handler = getattr(self, f"mk{name.capitalize()}", None)
        obj = handler(prompt, schema) if handler else self.mkDefault(prompt, schema)

        result = LLMResult(text=obj.model_dump_json(), prompt_tokens=len(toks(prompt)), completion_tokens=len(toks(obj.model_dump_json())), model=self.model)
        return obj, result

    @staticmethod
    def intent(q: str) -> str:
        low = q.lower()

        if any(w in low for w in _CHIT) and len(low) < 40:
            return "chitchat"
        if any(w in low for w in _CMP):
            return "comparison"
        if any(w in low for w in _AGG):
            return "aggregation"
        if any(w in low for w in _SUM):
            return "summarization"
        if ("who" in low or "which" in low) and ("largest" in low or "highest" in low or "then" in low):
            return "multi_hop"
        return "factoid"

    def mkRoute(self, prompt: str, schema: Type[T]) -> T:
        intent = self.intent(queryOf(prompt))
        return schema(intent=intent, needs_retrieval=intent != "chitchat", history_answerable=False, rationale="heuristic-fake")

    def mkQueryplan(self, prompt: str, schema: Type[T]) -> T:
        q = queryOf(prompt)
        low = q.lower()
        qid = extractTag(prompt, "query_id") or "q"
        steps: list[dict] = []

        if any(w in low for w in _CMP):
            parts = re.split(r"\s+vs\.?\s+|\s+versus\s+|\s+compared to\s+|\s+and\s+", q, maxsplit=1, flags=re.I)
            a = parts[0]
            b = parts[1] if len(parts) > 1 else q
            steps = [{"step_id": "s1", "tool": "hybrid", "query": a, "k": 8, "depends_on": []}, {"step_id": "s2", "tool": "hybrid", "query": b, "k": 8, "depends_on": []}, {"step_id": "s3", "tool": "code", "query": f"compute comparison of s1 vs s2 for: {q}", "k": 4, "depends_on": ["s1", "s2"]}]
            merge = "compare"
        elif any(w in low for w in _AGG):
            steps = [{"step_id": "s1", "tool": "metadata_filter", "query": q, "k": 50, "depends_on": []}]
            merge = "aggregate"
        elif any(w in low for w in _SUM):
            steps = [{"step_id": "s1", "tool": "doc_summary", "query": q, "k": 8, "depends_on": []}]
            merge = "concat"
        else:
            steps = [{"step_id": "s1", "tool": "hybrid", "query": q, "k": 8, "depends_on": []}]
            merge = "concat"

        return schema(query_id=qid, trace_id=extractTag(prompt, "trace_id"), intent=self.intent(q), sub_steps=steps, merge=merge)

    def mkGrade(self, prompt: str, schema: Type[T]) -> T:
        blocks = parseEvidenceBlocks(prompt)
        q = set(toks(queryOf(prompt)))

        best = 0.0
        for b in blocks:
            overlap = len(q & set(toks(b["text"]))) / (len(q) or 1)
            best = max(best, overlap)

        verdict = "SUFFICIENT" if best >= 0.3 else ("AMBIGUOUS" if blocks else "IRRELEVANT")
        return schema(verdict=verdict, max_relevance=round(best, 3), covered_slots=[], missing_slots=[], rationale="heuristic-fake")

    def mkJudgement(self, prompt: str, schema: Type[T]) -> T:
        evidence = extractTag(prompt, "evidence")
        claim = extractTag(prompt, "claim")
        num = re.compile(r"-?\d[\d,]*\.?\d*")
        claim_nums = {n.replace(",", "").rstrip(".") for n in num.findall(claim)}
        ev_nums = {n.replace(",", "").rstrip(".") for n in num.findall(evidence)}

        ctoks = set(toks(claim))
        etoks = set(toks(evidence))
        overlap = len(ctoks & etoks) / (len(ctoks) or 1)

        supported = overlap >= 0.6 and claim_nums.issubset(ev_nums)
        return schema(supported=supported, rationale="heuristic-judge")

    def mkExtracteditems(self, prompt: str, schema: Type[T]) -> T:
        blocks = parseEvidenceBlocks(prompt)
        text = "\n".join(b["text"] for b in blocks) or prompt

        items: list[str] = []
        for m in re.finditer(r"^\s*(?:[-*•]|\d+[.)])\s+(.+)$", text, re.MULTILINE):
            items.append(m.group(1).strip())
        for m in re.finditer(r"(?:Segment|Item|Name|Subsidiary):\s*([^.;\n]+)", text, re.IGNORECASE):
            items.append(m.group(1).strip())

        if not items:
            m = re.search(r":\s*([A-Z][^.\n]{3,120}(?:,[^.\n]+)+)", text)
            if m:
                items = [p.strip() for p in re.split(r",|\band\b", m.group(1)) if p.strip()]

        return schema(items=[i for i in items if i][:200])

    def mkRewriteresult(self, prompt: str, schema: Type[T]) -> T:
        q = queryOf(prompt)
        entities = [e.strip() for e in extractTag(prompt, "entities").split(",") if e.strip()]
        pronoun = bool(re.search(r"\b(it|its|that|this|they|their|those|these)\b", q, re.I))
        standalone = q
        resolved = True

        if pronoun and entities:
            standalone = f"{q} (regarding {', '.join(entities)})"
        elif pronoun and not entities:
            resolved = False

        fields = schema.model_fields
        data = {}
        if "standalone_query" in fields:
            data["standalone_query"] = standalone
        if "carried_entities" in fields:
            data["carried_entities"] = entities
        if "resolved" in fields:
            data["resolved"] = resolved

        return schema(**data)

    def mkPlancritique(self, prompt: str, schema: Type[T]) -> T:
        seen: set[tuple[str, str]] = set()
        redundant: list[str] = []
        for m in re.finditer(r"(\S+): tool=(\S+) q=(.+)", prompt):
            sid, tool, q = m.group(1), m.group(2), m.group(3).strip().lower()
            key = (tool, q)
            if key in seen:
                redundant.append(sid)
            else:
                seen.add(key)

        return schema(redundant_step_ids=redundant, rationale="heuristic-critique")

    def mkDraft(self, prompt: str, schema: Type[T]) -> T:
        from ...contracts import AnswerFormat, Citation, DraftClaim

        blocks = parseEvidenceBlocks(prompt)
        q = queryOf(prompt)
        qtoks = set(toks(q))
        claims: list[DraftClaim] = []

        if blocks:
            scored = []
            for b in blocks:
                for sent in sentences(b["text"]) or [b["text"]]:
                    ov = len(qtoks & set(toks(sent))) / (len(qtoks) or 1)
                    scored.append((ov, b, sent))
            scored.sort(key=lambda t: t[0], reverse=True)

            seen: set[str] = set()
            for ov, b, sent in scored:
                if ov <= 0 or sent in seen or len(claims) >= 2:
                    continue
                seen.add(sent)
                start = b["text"].find(sent)
                span = (start, start + len(sent)) if start >= 0 else (0, len(sent))
                claims.append(DraftClaim(text=sent, citations=[Citation(chunk_id=b["chunk_id"], doc_id=b["doc_id"], page_no=b["page"], char_span=span, quote=sent)]))

        answer_text = " ".join(c.text for c in claims) if claims else ""
        return schema(answer_text=answer_text, format=AnswerFormat.PROSE, claims=claims, computations=[])

    def mkSelfquery(self, prompt: str, schema: Type[T]) -> T:
        return schema(semantic_query=queryOf(prompt), filters={})

    def mkCodeplan(self, prompt: str, schema: Type[T]) -> T:
        blocks = parseEvidenceBlocks(prompt)
        q = queryOf(prompt).lower()
        num_re = re.compile(r"([A-Za-z][\w %/]*?)[:\s]\s*(-?\d[\d,]*\.?\d*)")

        found: list[tuple[str, float, str]] = []
        for b in blocks:
            for label, raw in num_re.findall(b["text"]):
                try:
                    found.append((label.strip()[:24].replace(" ", "_") or "v", float(raw.replace(",", "")), b["chunk_id"]))
                except ValueError:
                    continue

        inputs, code, template = [], "", ""
        if len(found) >= 2:
            (label_a, value_a, chunk_a), (label_b, value_b, chunk_b) = found[0], found[1]
            var_a, var_b = f"a_{len(label_a)}", f"b_{len(label_b)}"
            inputs = [{"name": var_a, "value": value_a, "source_chunk_id": chunk_a}, {"name": var_b, "value": value_b, "source_chunk_id": chunk_b}]
            if any(w in q for w in ("change", "growth", "compare", "vs", "versus", "difference")):
                code = f"result = ({var_a} - {var_b}) / {var_b} * 100"
                template = "The change is {result}%."
            else:
                code = f"result = {var_a} + {var_b}"
                template = "The total is {result}."
        elif len(found) == 1:
            inputs = [{"name": "a", "value": found[0][1], "source_chunk_id": found[0][2]}]
            code = "result = a"
            template = "The value is {result}."

        return schema(inputs=inputs, code=code, claim_template=template)

    def mkDefault(self, prompt: str, schema: Type[T]) -> T:
        data = {}
        for fname, field in schema.model_fields.items():
            if field.is_required():
                ann = field.annotation
                data[fname] = "" if ann in (str, str | None) else (0 if ann in (int, float) else None)

        try:
            return schema(**data)
        except Exception:
            return schema.model_construct(**data)
