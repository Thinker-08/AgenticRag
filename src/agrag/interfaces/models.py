"""LLM and EmbeddingModel — the model-serving seams (C1, C3)."""

from __future__ import annotations

from typing import Protocol, Sequence, Type, TypeVar, runtime_checkable

from pydantic import BaseModel

from .types import EmbeddingResult, LLMResult

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLM(Protocol):
    """Generate / classify / reformulate. Structured output is grammar-constrained, not parsed-and-prayed."""

    name: str

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        images: Sequence[bytes] | None = None,
        timeout_s: float | None = None,
    ) -> LLMResult: ...

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
        """Emit an instance of `schema` under constrained decoding; the envelope is valid by construction."""
        ...


@runtime_checkable
class EmbeddingModel(Protocol):
    """Map text to dense (+ optional learned-sparse) vectors under a versioned contract (C3)."""

    model: str
    version: str
    dim: int

    def encode_documents(self, texts: Sequence[str]) -> EmbeddingResult: ...

    def encode_queries(self, texts: Sequence[str]) -> EmbeddingResult: ...
