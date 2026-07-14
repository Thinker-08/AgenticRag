from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str = "fake"
    model: str = "gemma4:12b"
    small_model: str = "gemma3:4b"
    host: str = "http://localhost:11434"
    max_tokens: int = 1024
    temperature: float = 0.0


class EmbeddingConfig(BaseModel):
    provider: str = "hash"
    model: str = "BAAI/bge-m3"
    version: str = "1.5"
    dim: int = 1024
    device: str = "cpu"


class VectorStoreConfig(BaseModel):
    provider: str = "memory"
    host: str = "http://localhost:6333"
    collection: str = "agrag_chunks"
    quantize: str = "none"
    hnsw_m: int = 32
    ef_construction: int = 256
    ef_search: int = 128


class LexicalConfig(BaseModel):
    provider: str = "bm25"
    k1: float = 1.5
    b: float = 0.75


class RerankerConfig(BaseModel):
    provider: str = "identity"
    model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"


class VerifierConfig(BaseModel):
    provider: str = "lexical"
    model: str = "cross-encoder/nli-deberta-v3-base"
    tau_entail: float = 0.85
    tau_contra: float = 0.10
    judge_gray_zone: bool = True


class CacheConfig(BaseModel):
    provider: str = "memory"
    host: str = "redis://localhost:6379/0"
    answers_enabled: bool = True
    answer_ttl_s: int = 3600
    semantic_enabled: bool = True
    semantic_threshold: float = 0.97
    semantic_max_entries: int = 128


class TracerConfig(BaseModel):
    provider: str = "logging"
    host: str = "http://localhost:3000"


class ParserConfig(BaseModel):
    provider: str = "pymupdf"
    text_ratio_threshold: float = 0.1
    max_pages: int = 3000
    max_upload_mb: int = 100
    parse_timeout_s: float = 120.0
    max_inflate_ratio: float = 200.0


class ChunkerConfig(BaseModel):
    provider: str = "hierarchical"
    child_size: int = 320
    parent_size: int = 1500
    overlap: int = 64


class RetrievalConfig(BaseModel):
    over_fetch: int = 100
    rrf_k: int = 60
    dedupe_threshold: float = 0.9
    top_k: int = 8
    bm25_weight: float = 1.0
    dense_weight: float = 1.0


class SandboxConfig(BaseModel):
    provider: str = "subprocess"
    timeout_s: float = 2.0
    max_mem_mb: int = 256


class AgentConfig(BaseModel):
    wall_clock_s: float = 30.0
    token_budget: int = 20000
    max_iters: int = 3
    grade_relevance_floor: float = 0.7
    slot_concurrency: int = 4
    max_scan_chunks: int = 500


class ReliabilityConfig(BaseModel):
    max_retries: int = 2
    backoff_base_s: float = 0.2
    backoff_cap_s: float = 2.0
    breaker_failures: int = 5
    breaker_window_s: float = 30.0
    breaker_cooldown_s: float = 15.0
    slot_wait_fraction: float = 0.3


class ServingConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    default_tenant: str = "default"


class Settings(BaseModel):
    mode: str = "local"
    agent_mode: str = "agentic"
    data_dir: str = "./data"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    lexical: LexicalConfig = Field(default_factory=LexicalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    tracer: TracerConfig = Field(default_factory=TracerConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    reliability: ReliabilityConfig = Field(default_factory=ReliabilityConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)

    def isBaseline(self) -> bool:
        return self.agent_mode == "baseline"


def deepMerge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deepMerge(out[k], v)
        else:
            out[k] = v

    return out


def envOverrides() -> dict[str, Any]:
    out: dict[str, Any] = {}
    top = {f.upper() for f in Settings.model_fields}

    for key, val in os.environ.items():
        if not key.startswith("AGRAG_"):
            continue
        rest = key[len("AGRAG_") :]
        section = next((s for s in top if rest == s or rest.startswith(s + "_")), None)
        if rest in top:
            out[rest.lower()] = val
        elif section:
            field = rest[len(section) + 1 :].lower()
            out.setdefault(section.lower(), {})[field] = coerce(val)

    return out


def coerce(v: str) -> Any:
    low = v.lower()

    if low in ("true", "false"):
        return low == "true"

    try:
        return int(v)
    except ValueError:
        pass

    try:
        return float(v)
    except ValueError:
        return v


def loadSettings(path: str | Path | None = None) -> Settings:
    data: dict[str, Any] = {}
    cfg_path = Path(path) if path else Path(os.getenv("AGRAG_CONFIG", "config/default.yaml"))
    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text()) or {}

    env = envOverrides()
    data = deepMerge(data, env)

    return Settings(**data)
