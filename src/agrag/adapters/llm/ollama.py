"""Ollama LLM adapter (full mode): grammar-constrained output behind reliability primitives.

Every call is guarded by a circuit breaker (fail fast to the fallback tier when the server is
unhealthy), deadline-gated retries with jitter on transient faults, and a GPU-slot semaphore that
sheds to Backpressure rather than relocating the queue into the GPU (07 §1.2, §2.2-§2.5).
"""

from __future__ import annotations

import base64
from typing import Any, Sequence, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ...config import ReliabilityConfig
from ...contracts import Budget
from ...interfaces.types import LLMResult
from ...reliability import CircuitBreaker, CircuitOpen, SlotPool, retry_async

T = TypeVar("T", bound=BaseModel)

_RETRIABLE = (httpx.TransportError, httpx.HTTPStatusError, httpx.TimeoutException)


class OllamaLLM:
    """Generate / structured-generate against a local Ollama server."""

    name: str

    def __init__(
        self,
        model: str,
        host: str,
        default_max_tokens: int = 1024,
        *,
        reliability: ReliabilityConfig | None = None,
        slots: int = 4,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self.model = model
        self.name = model
        self.host = host.rstrip("/")
        self.default_max_tokens = default_max_tokens
        self._client = httpx.AsyncClient(base_url=self.host, timeout=httpx.Timeout(300.0))
        self._rel = reliability or ReliabilityConfig()
        self._slots = SlotPool(slots, wait_fraction=self._rel.slot_wait_fraction)
        self._breaker = breaker or CircuitBreaker(
            failures=self._rel.breaker_failures,
            window_s=self._rel.breaker_window_s,
            cooldown_s=self._rel.breaker_cooldown_s,
        )

    def _payload(
        self,
        prompt: str,
        system: str | None,
        max_tokens: int,
        temperature: float,
        images: Sequence[bytes] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system is not None:
            payload["system"] = system
        if images:
            payload["images"] = [base64.b64encode(img).decode("ascii") for img in images]
        return payload

    async def _raw_post(self, payload: dict[str, Any], timeout_s: float | None) -> LLMResult:
        kwargs: dict[str, Any] = {"timeout": timeout_s} if timeout_s is not None else {}
        resp = await self._client.post("/api/generate", json=payload, **kwargs)
        resp.raise_for_status()
        data = resp.json()
        return LLMResult(
            text=data.get("response", ""),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            model=self.model,
        )

    async def _post(self, payload: dict[str, Any], timeout_s: float | None) -> LLMResult:
        if not self._breaker.allow():
            raise CircuitOpen(f"{self.model} breaker open")
        budget = Budget.start(timeout_s, 10**9) if timeout_s is not None else None
        try:
            async with self._slots.acquire(budget):
                result = await retry_async(
                    lambda: self._raw_post(payload, timeout_s),
                    retriable=_RETRIABLE,
                    max_retries=self._rel.max_retries,
                    base_s=self._rel.backoff_base_s,
                    cap_s=self._rel.backoff_cap_s,
                    budget=budget,
                )
            self._breaker.record_success()
            return result
        except Exception:
            self._breaker.record_failure()
            raise

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        images: Sequence[bytes] | None = None,
        timeout_s: float | None = None,
    ) -> LLMResult:
        payload = self._payload(prompt, system, max_tokens, temperature, images)
        return await self._post(payload, timeout_s)

    async def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        images: Sequence[bytes] | None = None,
        timeout_s: float | None = None,
    ) -> tuple[T, LLMResult]:
        fmt = schema.model_json_schema()
        payload = self._payload(prompt, system, max_tokens, temperature, images)
        payload["format"] = fmt
        result = await self._post(payload, timeout_s)
        try:
            return schema.model_validate_json(result.text), result
        except ValidationError as exc:
            repair = (
                f"{prompt}\n\nThe previous response failed schema validation:\n{exc}\n"
                "Return ONLY valid JSON that conforms to the required schema."
            )
            payload = self._payload(repair, system, max_tokens, temperature, images)
            payload["format"] = fmt
            result = await self._post(payload, timeout_s)
            return schema.model_validate_json(result.text), result
