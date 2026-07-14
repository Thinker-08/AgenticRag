from __future__ import annotations

from .config import Settings, loadSettings
from .deps import Deps
from .interfaces import EmbeddingModel, LLM


def makeTracer(settings: Settings):
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


def makeLlm(settings: Settings, model: str) -> LLM:
    if settings.llm.provider == "fake":
        from .adapters.llm.fake import FakeLLM

        return FakeLLM(model=model)

    if settings.llm.provider == "ollama":
        from .adapters.llm.ollama import OllamaLLM

        return OllamaLLM(model=model, host=settings.llm.host, default_max_tokens=settings.llm.max_tokens, num_ctx=settings.llm.num_ctx, reliability=settings.reliability, slots=settings.agent.slot_concurrency)

    raise ValueError(f"unknown llm provider {settings.llm.provider}")


def makeEmbedding(settings: Settings) -> EmbeddingModel:
    if settings.embedding.provider == "hash":
        from .adapters.embedding.hash import HashEmbedding

        return HashEmbedding(dim=settings.embedding.dim, version=settings.embedding.version)

    if settings.embedding.provider == "bge_m3":
        from .adapters.embedding.bge_m3 import BgeM3Embedding

        return BgeM3Embedding(model=settings.embedding.model, version=settings.embedding.version, device=settings.embedding.device)

    raise ValueError(f"unknown embedding provider {settings.embedding.provider}")


def makeVectorstore(settings: Settings):
    if settings.vectorstore.provider == "memory":
        from .adapters.vectorstore.memory import MemoryVectorStore

        return MemoryVectorStore()

    if settings.vectorstore.provider == "qdrant":
        from .adapters.vectorstore.qdrant import QdrantVectorStore

        return QdrantVectorStore(host=settings.vectorstore.host, collection=settings.vectorstore.collection, dim=settings.embedding.dim, cfg=settings.vectorstore)

    raise ValueError(f"unknown vectorstore provider {settings.vectorstore.provider}")


def makeLexical(settings: Settings):
    from .adapters.lexical.bm25 import Bm25Index

    return Bm25Index(k1=settings.lexical.k1, b=settings.lexical.b)


def makeDocstore(settings: Settings):
    if settings.cache.provider == "redis":
        from .adapters.docstore.redis import RedisDocStore

        return RedisDocStore(host=settings.cache.host)

    from .adapters.docstore.memory import MemoryDocStore

    return MemoryDocStore()


def makeSessions(settings: Settings):
    if settings.cache.provider == "redis":
        from .adapters.session.redis import RedisSessionStore

        return RedisSessionStore(host=settings.cache.host)

    from .adapters.session.memory import MemorySessionStore

    return MemorySessionStore()


def makeReranker(settings: Settings):
    if settings.reranker.provider == "identity":
        from .adapters.reranker.identity import IdentityReranker

        return IdentityReranker()

    if settings.reranker.provider == "bge":
        from .adapters.reranker.bge import BgeReranker

        return BgeReranker(model=settings.reranker.model, device=settings.reranker.device)

    raise ValueError(f"unknown reranker provider {settings.reranker.provider}")


def makeVerifier(settings: Settings):
    if settings.verifier.provider == "lexical":
        from .adapters.verifier.lexical import LexicalVerifier

        return LexicalVerifier()

    if settings.verifier.provider == "nli":
        from .adapters.verifier.nli import NliVerifier

        return NliVerifier(model=settings.verifier.model)

    raise ValueError(f"unknown verifier provider {settings.verifier.provider}")


def makeCache(settings: Settings):
    if settings.cache.provider == "memory":
        from .adapters.cache.memory import MemoryCache

        return MemoryCache()

    if settings.cache.provider == "redis":
        from .adapters.cache.redis import RedisCache

        return RedisCache(host=settings.cache.host)

    raise ValueError(f"unknown cache provider {settings.cache.provider}")


def makeParser(settings: Settings, vision_llm: LLM):
    from .adapters.parser.pymupdf import PymupdfParser

    return PymupdfParser(cfg=settings.parser, vision_llm=vision_llm)


def makeChunker(settings: Settings, embedder=None):
    from .ingestion.chunker import buildChunker

    return buildChunker(settings.chunker, embedder=embedder)


def makeSandbox(settings: Settings):
    from .tools.sandbox import buildSandbox

    return buildSandbox(settings.sandbox)


def makeGrader(settings: Settings, small_llm: LLM):
    from .agent.grader import HeuristicGrader

    return HeuristicGrader(relevance_floor=settings.agent.grade_relevance_floor)


def buildDeps(settings: Settings | None = None) -> Deps:
    settings = settings or loadSettings()
    llm = makeLlm(settings, settings.llm.model)
    small_llm = makeLlm(settings, settings.llm.small_model)
    embedding = makeEmbedding(settings)
    vectorstore = makeVectorstore(settings)
    lexical = makeLexical(settings)
    reranker = makeReranker(settings)

    from .retrieval.hybrid import HybridRetriever

    cache = makeCache(settings)
    retriever = HybridRetriever(embedding=embedding, vectorstore=vectorstore, lexical=lexical, reranker=reranker, cfg=settings.retrieval, cache=cache if settings.cache.answers_enabled else None)

    return Deps(settings=settings, llm=llm, small_llm=small_llm, embedding=embedding, vectorstore=vectorstore, lexical=lexical, docstore=makeDocstore(settings), reranker=reranker, grader=makeGrader(settings, small_llm), verifier=makeVerifier(settings), cache=cache, tracer=makeTracer(settings), parser=makeParser(settings, llm), chunker=makeChunker(settings, embedder=embedding), toolrunner=makeSandbox(settings), retriever=retriever, sessions=makeSessions(settings))


def buildApp(settings: Settings | None = None):
    deps = buildDeps(settings)
    if deps.settings.isBaseline():
        from .baseline.vanilla import BaselineRAG

        return BaselineRAG(deps)
    from .agent.app import AgentApp

    return AgentApp(deps)


def buildIngestion(settings: Settings | None = None):
    deps = buildDeps(settings)
    from .ingestion.service import IngestionService

    return IngestionService(deps)
