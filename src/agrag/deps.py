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
    SessionStore,
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
    sessions: SessionStore
