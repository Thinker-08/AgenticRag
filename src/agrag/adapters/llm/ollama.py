from __future__ import annotations

import base64
from typing import Any, Sequence, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ...config import ReliabilityConfig
from ...contracts import Budget
from ...interfaces.types import LLMResult
from ...reliability import CircuitBreaker, CircuitOpen, SlotPool, retryAsync

T = TypeVar("T", bound=BaseModel)

_RETRIABLE = (httpx.TransportError, httpx.HTTPStatusError, httpx.TimeoutException)


class OllamaLLM:
    name: str

    def __init__(self, model: str, host: str, default_max_tokens: int = 1024, *, num_ctx: int = 8192, reliability: ReliabilityConfig | None = None, slots: int = 4, breaker: CircuitBreaker | None = None) -> None:
        self.model = model
        self.name = model
        self.host = host.rstrip("/")
        self.default_max_tokens = default_max_tokens
        self.num_ctx = num_ctx

        self._client = httpx.AsyncClient(base_url=self.host, timeout=httpx.Timeout(300.0))
        self._rel = reliability or ReliabilityConfig()
        self._slots = SlotPool(slots, wait_fraction=self._rel.slot_wait_fraction)
        self._breaker = breaker or CircuitBreaker(failures=self._rel.breaker_failures, window_s=self._rel.breaker_window_s, cooldown_s=self._rel.breaker_cooldown_s)

    def payload(self, prompt: str, system: str | None, max_tokens: int, temperature: float, images: Sequence[bytes] | None) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": self.num_ctx}}
        if system is not None:
            payload["system"] = system
        if images:
            payload["images"] = [base64.b64encode(img).decode("ascii") for img in images]

        return payload

    async def rawPost(self, payload: dict[str, Any], timeout_s: float | None) -> LLMResult:
        kwargs: dict[str, Any] = {"timeout": timeout_s} if timeout_s is not None else {}
        resp = await self._client.post("/api/generate", json=payload, **kwargs)
        resp.raise_for_status()
        data = resp.json()

        return LLMResult(text=data.get("response", ""), prompt_tokens=data.get("prompt_eval_count", 0), completion_tokens=data.get("eval_count", 0), model=self.model)

    async def post(self, payload: dict[str, Any], timeout_s: float | None) -> LLMResult:
        if not self._breaker.allow():
            raise CircuitOpen(f"{self.model} breaker open")

        budget = Budget.start(timeout_s, 10**9) if timeout_s is not None else None
        try:
            async with self._slots.acquire(budget):
                result = await retryAsync(lambda: self.rawPost(payload, timeout_s), retriable=_RETRIABLE, max_retries=self._rel.max_retries, base_s=self._rel.backoff_base_s, cap_s=self._rel.backoff_cap_s, budget=budget)
            self._breaker.recordSuccess()
            return result
        except Exception:
            self._breaker.recordFailure()
            raise

    async def generate(self, prompt: str, *, system: str | None = None, max_tokens: int = 512, temperature: float = 0.0, images: Sequence[bytes] | None = None, timeout_s: float | None = None) -> LLMResult:
        payload = self.payload(prompt, system, max_tokens, temperature, images)
        return await self.post(payload, timeout_s)

    async def generateStructured(self, prompt: str, schema: Type[T], *, system: str | None = None, max_tokens: int = 512, temperature: float = 0.0, images: Sequence[bytes] | None = None, timeout_s: float | None = None) -> tuple[T, LLMResult]:
        fmt = schema.model_json_schema()
        payload = self.payload(prompt, system, max_tokens, temperature, images)
        payload["format"] = fmt
        result = await self.post(payload, timeout_s)

        try:
            return schema.model_validate_json(result.text), result
        except ValidationError as exc:
            repair = f"{prompt}\n\nThe previous response failed schema validation:\n{exc}\nReturn ONLY valid JSON that conforms to the required schema."
            payload = self.payload(repair, system, max_tokens, temperature, images)
            payload["format"] = fmt
            result = await self.post(payload, timeout_s)
            return schema.model_validate_json(result.text), result
