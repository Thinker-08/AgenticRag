from __future__ import annotations

import hashlib
import re
from typing import Awaitable, Callable

import numpy as np

from ..contracts import Answer, AnswerStatus
from ..deps import Deps

_NUM = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)|\d[\d,]*\.?\d+|\d+")
_WS = re.compile(r"\s+")


def normQuery(q: str) -> str:
    return _WS.sub(" ", q.strip().lower())


def salient(q: str) -> set[str]:
    return {m.group().replace(",", "") for m in _NUM.finditer(q)}


def historyHash(history: list) -> str:
    joined = "|".join(getattr(t, "content", "") for t in history)
    return hashlib.blake2b(joined.encode(), digest_size=8).hexdigest()


class AnswerCache:
    def __init__(self, deps: Deps) -> None:
        self.deps = deps
        cfg = deps.settings.cache
        self.enabled = cfg.answers_enabled
        self.semantic_enabled = cfg.semantic_enabled
        self.threshold = cfg.semantic_threshold
        self.ttl = cfg.answer_ttl_s
        self.max_entries = cfg.semantic_max_entries

    async def docsetVersion(self, tenant_id: str) -> str:
        from ..contracts import JobState

        docs = sorted((d.doc_id, d.content_hash) for d in await self.deps.docstore.listDocs(tenant_id) if d.status == JobState.READY)
        return hashlib.blake2b(repr(docs).encode(), digest_size=8).hexdigest()

    def exactKey(self, tenant_id: str, version: str, q: str, history_hash: str) -> str:
        digest = hashlib.blake2b(f"{normQuery(q)}|{history_hash}".encode(), digest_size=12).hexdigest()
        return f"ans:{tenant_id}:{version}:{digest}"

    def semIndexKey(self, tenant_id: str, version: str) -> str:
        return f"ansidx:{tenant_id}:{version}"

    async def get(self, tenant_id: str, query: str, history: list) -> Answer | None:
        if not self.enabled:
            return None

        version = await self.docsetVersion(tenant_id)
        hh = historyHash(history)
        exact = await self.deps.cache.get(self.exactKey(tenant_id, version, query, hh))
        if exact is not None:
            ans = Answer.model_validate(exact)
            ans.from_cache = True
            return ans

        if not self.semantic_enabled or history:
            return None
        return await self.semanticGet(tenant_id, version, query)

    async def semanticGet(self, tenant_id: str, version: str, query: str) -> Answer | None:
        index = await self.deps.cache.get(self.semIndexKey(tenant_id, version)) or []
        if not index:
            return None

        qvec = np.asarray(self.deps.embedding.encodeQueries([query]).dense[0], dtype=np.float32)
        qn = float(np.linalg.norm(qvec)) or 1.0
        qsal = salient(query)
        best, best_sim = None, 0.0

        for entry in index:
            vec = np.asarray(entry["vec"], dtype=np.float32)
            sim = float(np.dot(qvec, vec) / (qn * (float(np.linalg.norm(vec)) or 1.0)))
            if sim >= self.threshold and sim > best_sim and set(entry["salient"]) == qsal:
                best, best_sim = entry, sim

        if best is None:
            return None
        cached = await self.deps.cache.get(best["key"])
        if cached is None:
            return None
        ans = Answer.model_validate(cached)
        ans.from_cache = True
        self.deps.tracer.event("cache.semantic_hit", sim=round(best_sim, 4))
        return ans

    async def resolve(self, tenant_id: str, query: str, history: list, compute: Callable[[], Awaitable[Answer]]) -> Answer:
        if not self.enabled:
            return await compute()
        hit = await self.get(tenant_id, query, history)
        if hit is not None:
            return hit

        version = await self.docsetVersion(tenant_id)
        key = self.exactKey(tenant_id, version, query, historyHash(history))

        async def computePayload():
            answer = await compute()
            return answer.model_dump(mode="json")

        payload = await self.deps.cache.getOrCompute(key, computePayload, ttl_s=self.ttl)
        answer = Answer.model_validate(payload)
        if answer.status == AnswerStatus.ABSTAINED or answer.degraded:
            await self.deps.cache.invalidate(key)
        elif self.semantic_enabled and not history:
            await self.semanticPut(tenant_id, version, query, key)

        return answer

    async def semanticPut(self, tenant_id: str, version: str, query: str, key: str) -> None:
        idx_key = self.semIndexKey(tenant_id, version)
        index = await self.deps.cache.get(idx_key) or []
        vec = self.deps.embedding.encodeQueries([query]).dense[0]

        index = [e for e in index if e["key"] != key]
        index.append({"key": key, "vec": vec, "salient": sorted(salient(query))})

        await self.deps.cache.set(idx_key, index[-self.max_entries :], ttl_s=self.ttl)
