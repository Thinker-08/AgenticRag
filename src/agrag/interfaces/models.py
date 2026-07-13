from __future__ import annotations

from typing import Protocol, Sequence, Type, TypeVar, runtime_checkable

from pydantic import BaseModel

from .types import EmbeddingResult, LLMResult

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLM(Protocol):
    name: str

    async def generate(self, prompt: str, *, system: str | None = None, max_tokens: int = 512, temperature: float = 0.0, images: Sequence[bytes] | None = None, timeout_s: float | None = None) -> LLMResult: ...

    async def generateStructured(self, prompt: str, schema: Type[T], *, system: str | None = None, max_tokens: int = 512, temperature: float = 0.0, images: Sequence[bytes] | None = None, timeout_s: float | None = None) -> tuple[T, LLMResult]: ...


@runtime_checkable
class EmbeddingModel(Protocol):
    model: str
    version: str
    dim: int

    def encodeDocuments(self, texts: Sequence[str]) -> EmbeddingResult: ...

    def encodeQueries(self, texts: Sequence[str]) -> EmbeddingResult: ...
