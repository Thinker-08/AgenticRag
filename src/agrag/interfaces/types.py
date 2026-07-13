from __future__ import annotations

from dataclasses import dataclass, field

from ..contracts import Chunk, SupportLabel

SparseVector = dict[int, float]


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class EmbeddingResult:
    dense: list[list[float]]
    sparse: list[SparseVector] = field(default_factory=list)
    model: str = ""
    version: str = ""


@dataclass
class VectorRecord:
    chunk: Chunk
    dense: list[float]
    sparse: SparseVector | None = None


@dataclass
class ToolResult:
    ok: bool
    result: object = None
    stdout: str = ""
    stderr: str = ""
    run_id: str = ""
    error: str = ""
    wall_ms: float = 0.0


@dataclass
class VerdictResult:
    label: SupportLabel
    score: float
    verifier: str = "NLI"
