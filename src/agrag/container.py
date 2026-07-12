"""The composition root: build a `Deps` bundle and top-level apps from `Settings`.

Concrete adapters are imported lazily inside factories so that selecting the `local`
provider set never imports torch/qdrant/redis, and a missing optional dependency only
errors when its provider is actually chosen.
"""

from __future__ import annotations

from .config import Settings, load_settings
from .deps import Deps
from .interfaces import EmbeddingModel, LLM


def _tracer(settings: Settings):
    if settings.tracer.provider == "logging":
        from .adapters.tracer.logging import LoggingTracer

        return LoggingTracer()
    if settings.tracer.provider == "langfuse":
        from .adapters.tracer.langfuse import LangfuseTracer

        return LangfuseTracer(host=settings.tracer.host)
    if settings.tracer.provider == "otel":
        from .adapters.tracer.otel import OtelTracer

        return OtelTracer(host=settings.tracer.host)
    raise ValueError(f"unknown tracer provider {settings.tracer.provider}")


def _llm(settings: Settings, model: str) -> LLM:
    if settings.llm.provider == "fake":
        from .adapters.llm.fake import FakeLLM

        return FakeLLM(model=model)
    if settings.llm.provider == "ollama":
        from .adapters.llm.ollama import OllamaLLM

        return OllamaLLM(
            model=model,
            host=settings.llm.host,
            default_max_tokens=settings.llm.max_tokens,
            reliability=settings.reliability,
            slots=settings.agent.slot_concurrency,
        )
    raise ValueError(f"unknown llm provider {settings.llm.provider}")


def _embedding(settings: Settings) -> EmbeddingModel:
    if settings.embedding.provider == "hash":
        from .adapters.embedding.hash import HashEmbedding

        return HashEmbedding(dim=settings.embedding.dim, version=settings.embedding.version)
    if settings.embedding.provider == "bge_m3":
        from .adapters.embedding.bge_m3 import BgeM3Embedding

        return BgeM3Embedding(
            model=settings.embedding.model,
            version=settings.embedding.version,
            device=settings.embedding.device,
        )
    raise ValueError(f"unknown embedding provider {settings.embedding.provider}")


def _vectorstore(settings: Settings):
    if settings.vectorstore.provider == "memory":
        from .adapters.vectorstore.memory import MemoryVectorStore

        return MemoryVectorStore()
    if settings.vectorstore.provider == "qdrant":
        from .adapters.vectorstore.qdrant import QdrantVectorStore

        return QdrantVectorStore(
            host=settings.vectorstore.host,
            collection=settings.vectorstore.collection,
            dim=settings.embedding.dim,
            cfg=settings.vectorstore,
        )
    raise ValueError(f"unknown vectorstore provider {settings.vectorstore.provider}")


def _lexical(settings: Settings):
    from .adapters.lexical.bm25 import Bm25Index

    return Bm25Index(k1=settings.lexical.k1, b=settings.lexical.b)


def _docstore(settings: Settings):
    if settings.cache.provider == "redis":
        from .adapters.docstore.redis import RedisDocStore

        return RedisDocStore(host=settings.cache.host)
    from .adapters.docstore.memory import MemoryDocStore

    return MemoryDocStore()


def _sessions(settings: Settings):
    if settings.cache.provider == "redis":
        from .adapters.session.redis import RedisSessionStore

        return RedisSessionStore(host=settings.cache.host)
    from .adapters.session.memory import MemorySessionStore

    return MemorySessionStore()


def _reranker(settings: Settings):
    if settings.reranker.provider == "identity":
        from .adapters.reranker.identity import IdentityReranker

        return IdentityReranker()
    if settings.reranker.provider == "bge":
        from .adapters.reranker.bge import BgeReranker

        return BgeReranker(model=settings.reranker.model, device=settings.reranker.device)
    raise ValueError(f"unknown reranker provider {settings.reranker.provider}")


def _verifier(settings: Settings):
    if settings.verifier.provider == "lexical":
        from .adapters.verifier.lexical import LexicalVerifier

        return LexicalVerifier()
    if settings.verifier.provider == "nli":
        from .adapters.verifier.nli import NliVerifier

        return NliVerifier(model=settings.verifier.model)
    raise ValueError(f"unknown verifier provider {settings.verifier.provider}")


def _cache(settings: Settings):
    if settings.cache.provider == "memory":
        from .adapters.cache.memory import MemoryCache

        return MemoryCache()
    if settings.cache.provider == "redis":
        from .adapters.cache.redis import RedisCache

        return RedisCache(host=settings.cache.host)
    raise ValueError(f"unknown cache provider {settings.cache.provider}")


def _parser(settings: Settings, vision_llm: LLM):
    if settings.parser.provider == "text":
        from .adapters.parser.text import TextParser

        return TextParser()
    from .adapters.parser.pymupdf import PymupdfParser

    return PymupdfParser(cfg=settings.parser, vision_llm=vision_llm)


def _chunker(settings: Settings):
    from .ingestion.chunker import build_chunker

    return build_chunker(settings.chunker)


def _sandbox(settings: Settings):
    from .tools.sandbox import build_sandbox

    return build_sandbox(settings.sandbox)


def _grader(settings: Settings, small_llm: LLM):
    from .agent.grader import HeuristicGrader

    return HeuristicGrader(relevance_floor=settings.agent.grade_relevance_floor)


def build_deps(settings: Settings | None = None) -> Deps:
    settings = settings or load_settings()
    llm = _llm(settings, settings.llm.model)
    small_llm = _llm(settings, settings.llm.small_model)
    embedding = _embedding(settings)
    vectorstore = _vectorstore(settings)
    lexical = _lexical(settings)
    reranker = _reranker(settings)

    from .retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever(
        embedding=embedding,
        vectorstore=vectorstore,
        lexical=lexical,
        reranker=reranker,
        cfg=settings.retrieval,
    )
    return Deps(
        settings=settings,
        llm=llm,
        small_llm=small_llm,
        embedding=embedding,
        vectorstore=vectorstore,
        lexical=lexical,
        docstore=_docstore(settings),
        reranker=reranker,
        grader=_grader(settings, small_llm),
        verifier=_verifier(settings),
        cache=_cache(settings),
        tracer=_tracer(settings),
        parser=_parser(settings, llm),
        chunker=_chunker(settings),
        toolrunner=_sandbox(settings),
        retriever=retriever,
        sessions=_sessions(settings),
    )


def build_app(settings: Settings | None = None):
    """Return the query-plane app selected by `agent_mode` (agentic FSM or vanilla control)."""
    deps = build_deps(settings)
    if deps.settings.is_baseline():
        from .baseline.vanilla import BaselineRAG

        return BaselineRAG(deps)
    from .agent.app import AgentApp

    return AgentApp(deps)


def build_ingestion(settings: Settings | None = None):
    deps = build_deps(settings)
    from .ingestion.service import IngestionService

    return IngestionService(deps)
