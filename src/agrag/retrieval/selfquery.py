from __future__ import annotations

from pydantic import BaseModel, Field

from ..interfaces import LLM

_ALLOWED_FIELDS = {
    "doc_type",
    "fiscal_year",
    "fiscal_quarter",
    "page_no",
    "lang",
    "currency",
    "kind",
}
_ALLOWED_OPS = {"$in", "$nin", "$gte", "$lte", "$gt", "$lt", "$ne"}


class SelfQuery(BaseModel):
    semantic_query: str
    filters: dict = Field(default_factory=dict)


def sanitize(filters: dict) -> dict:
    clean: dict = {}
    for field, cond in (filters or {}).items():
        if field not in _ALLOWED_FIELDS:
            continue
        if isinstance(cond, dict):
            clean[field] = {op: v for op, v in cond.items() if op in _ALLOWED_OPS}
        else:
            clean[field] = cond
    return {k: v for k, v in clean.items() if v not in (None, {}, [])}


async def self_query(llm: LLM, query: str, *, timeout_s: float | None = None) -> SelfQuery:
    prompt = (
        "Split the question into a semantic search string and structured filters over the allowed "
        f"fields {sorted(_ALLOWED_FIELDS)} using operators {sorted(_ALLOWED_OPS)}.\n<query>{query}</query>"
    )
    try:
        sq, _ = await llm.generate_structured(
            prompt, SelfQuery, temperature=0.0, timeout_s=timeout_s
        )
    except Exception:
        return SelfQuery(semantic_query=query, filters={})
    return SelfQuery(semantic_query=sq.semantic_query or query, filters=sanitize(sq.filters))
