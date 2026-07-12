"""`Deps` — the assembled dependency bundle threaded through every service (C21).

Modules depend on this bundle of *interfaces*, never on concrete adapters. The container
(`agrag.container`) is the single place that picks concretes from `Settings` feature flags.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .interfaces import (
    LLM,
    Cache,
    Chunker,
    DocStore,
    EmbeddingModel,
    Grader,
    LexicalIndex,
    Parser,
    Reranker,
    Retriever,
    ToolRunner,
    Tracer,
    VectorStore,
    Verifier,
)


@dataclass
class Deps:
    settings: Settings
    llm: LLM
    small_llm: LLM
    embedding: EmbeddingModel
    vectorstore: VectorStore
    lexical: LexicalIndex
    docstore: DocStore
    reranker: Reranker
    grader: Grader
    verifier: Verifier
    cache: Cache
    tracer: Tracer
    parser: Parser
    chunker: Chunker
    toolrunner: ToolRunner
    retriever: Retriever
