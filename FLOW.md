# FLOW — how agrag is built and how a request travels through it

A reading guide for the codebase. It follows execution from process start-up, through the
two runtime paths (ingest a document, ask a question), pointing at the exact file and line
where each step lives. Read it next to the source with your editor open.

The one architectural idea to hold in mind: **every layer depends on an *interface*
(`src/agrag/interfaces/`), never on a concrete library.** A single file — `container.py` —
picks the concrete implementation for each interface from config flags and bundles them into
a `Deps` object that is threaded everywhere. That indirection is what lets the identical code
run in **local mode** (deterministic fake LLM, in-memory stores, no GPU) or **full mode**
(Ollama + Qdrant + Redis + BGE + Langfuse) with only a config change.

---

## 1 · The two planes

```
                         ┌─────────────────────────────┐
   PDF / text  ─────────▶│  OFFLINE INGESTION PLANE     │
                         │  parse → chunk → contextual- │
                         │  ize → embed → index         │
                         └──────────────┬──────────────┘
                                        │ writes
                                        ▼
                         ┌─────────────────────────────┐
                         │  SHARED HYBRID INDEX          │
                         │  vector store + BM25 + docs   │
                         └──────────────┬──────────────┘
                                        │ reads
                                        ▼
   question   ─────────▶ ┌─────────────────────────────┐ ─────────▶ grounded answer
                         │  ONLINE SERVING PLANE         │            (cited) or abstain
                         │  agentic FSM  /  vanilla ctrl │
                         └─────────────────────────────┘
```

The two planes share one thing: the index. They meet nowhere else. Both are assembled from
the same `Deps` bundle.

---

## 2 · Start-up: how the object graph is assembled

Entry point is the ASGI server command in the [Dockerfile](Dockerfile) (or `make run`):

```
uvicorn agrag.serving.app:app
```

Loading that module runs its last line — `app = create_app()` — which triggers the whole
wiring chain:

| Step | Where | What happens |
|---|---|---|
| Module load | [src/agrag/serving/app.py:84](src/agrag/serving/app.py#L84) | `app = create_app()` runs at import time |
| Build the app | [src/agrag/serving/app.py:35](src/agrag/serving/app.py#L35) | `create_app()` orchestrates everything below |
| Load config | [src/agrag/config.py:180](src/agrag/config.py#L180) | `load_settings()` reads the YAML, then applies `AGRAG_*` env overrides |
| **Composition root** | [src/agrag/container.py:122](src/agrag/container.py#L122) | `build_deps()` picks a concrete adapter per interface from the flags |
| The bundle | [src/agrag/deps.py:30](src/agrag/deps.py#L30) | `Deps` dataclass — every interface, passed by reference to all services |
| Offline plane | [src/agrag/ingestion/service.py:24](src/agrag/ingestion/service.py#L24) | `IngestionService(deps)` |
| Online plane | [src/agrag/serving/app.py:76](src/agrag/serving/app.py#L76) | `_build_query_app()` → `AgentApp` or `BaselineRAG`, chosen by `settings.is_baseline()` |
| Routes | [src/agrag/serving/app.py:41](src/agrag/serving/app.py#L41)+ | `/health`, `/ingest`, `/docs/{tenant}/{doc}`, `/ask` |

### The composition root in detail

[container.py](src/agrag/container.py) is the **only** file that names concrete classes. Each
factory is a small `if provider == ...` switch that lazily imports the chosen implementation —
so selecting the local providers never imports torch/qdrant/redis, and a missing optional
dependency only errors if you actually select that provider.

```
build_deps()                       container.py:122
  ├─ _llm(s, model)                :25   → FakeLLM        | OllamaLLM
  ├─ _embedding(s)                 :35   → HashEmbedding  | BgeM3Embedding
  ├─ _vectorstore(s)               :45   → MemoryVectorStore | QdrantVectorStore
  ├─ _lexical(s)                   :56   → Bm25Index
  ├─ _reranker(s)                  :69   → IdentityReranker | BgeReranker
  ├─ _verifier(s)                  :79   → LexicalVerifier  | NliVerifier
  ├─ _cache(s)                     :89   → MemoryCache      | RedisCache
  ├─ _tracer(s)                    :15   → LoggingTracer    | LangfuseTracer
  ├─ _parser(s, llm)               :99   → TextParser       | PymupdfParser
  ├─ HybridRetriever(...)          :131  → composes embedding+vectorstore+lexical+reranker
  └─ returns Deps(...)             :136
```

**Read these three files first** — they explain how the whole thing is built:
`config.py` → `container.py` → `deps.py`, plus a glance at `interfaces/` for the contracts.

---

## 3 · Ingest flow (offline plane)

Trigger: `POST /ingest` ([serving/app.py:46](src/agrag/serving/app.py#L46)) or `agrag ingest <path>`
([cli.py:20](src/agrag/cli.py#L20)). Both call into `IngestionService`.

```
IngestionService.submit(bytes)                      ingestion/service.py:40
  │  content_hash = sha256(bytes)                    ← idempotency + cache key (C10)
  │  already READY for this hash?  → return, no work  service.py:43
  │  already in-flight?            → return handle    service.py:45
  │  oversized?                    → quarantine        service.py:49  (PDF-bomb guard)
  └─▶ enqueue job → _run(job, bytes)                 service.py:95
        │
        ├─ parse         parser.parse(bytes)          service.py:105  → ParsedDoc (pages, tables, layout)
        ├─ chunk         chunker.split(parsed)        service.py:113  → parent/child hierarchical chunks
        ├─ contextualize contextualize(chunks, ...)   ingestion/stages.py:40
        │                  small LLM writes a 1-2 sentence "situating" blurb per chunk (cached)
        ├─ embed+index   embed_and_index(chunks, ...) ingestion/stages.py:55
        │                  embed each chunk → upsert to vector store + BM25 + doc store
        └─ mark READY, supersede prior versions       service.py:125, :138  (blue/green re-index, C17/C20)
```

Key contracts to notice: chunks are **frozen** (immutable) — each stage returns updated copies
via `model_copy`, never mutates ([stages.py:3](src/agrag/ingestion/stages.py#L3)). `content_hash`
is the idempotency and cache-invalidation key throughout.

---

## 4 · Ask flow — the agentic FSM (the heart of the project)

Trigger: `POST /ask` ([serving/app.py:69](src/agrag/serving/app.py#L69)) or `agrag ask "..."`
([cli.py:30](src/agrag/cli.py#L30)).

```
AgentApp.answer(query, history)                      agent/app.py:26
  │  build a Budget (wall-clock + tokens + max_iters) agent/app.py:18   ← the hard stop
  └─▶ AgentGraph FSM (LangGraph state machine)        agent/graph.py
```

Every method named `n_*` in [agent/graph.py](src/agrag/agent/graph.py) is a **state**; the
`route_*` methods are the **transitions**; `_build()` at the bottom wires them into the graph.

```
   n_contextualize   graph.py:45    rewrite "what about it?" into a standalone question using history
        │
   n_classify        graph.py:58    intent = factoid | aggregation | comparison | ... ; retrieve or not?
        │
        ├── chit-chat / history-answerable ──▶ n_respond   graph.py:66  → DONE (no retrieval)
        │
   n_plan            graph.py:83    decompose into a DAG of sub-steps + strategy per step
        │                           (sanitized: clamp k, cap fan-out, drop bad deps  graph.py:93)
        ▼
   n_retrieve        graph.py:106   PlanExecutor runs the DAG (see §5)
        ▼
   n_grade           graph.py:113   Corrective-RAG: is the evidence good enough?
        │
        ├── SUFFICIENT ─────────────▶ n_generate
        ├── weak, budget left ──────▶ n_reformulate ─┐  graph.py:120  (change strategy, add missing keyword)
        └── budget exhausted ───────▶ n_abstain      │
                                          ▲          │
                                          └──────────┘  (reformulate loops back to n_retrieve)
        ▼
   n_generate        graph.py:132   LLM writes a draft; every claim must cite a chunk_id + verbatim quote
        ▼
   n_verify          graph.py:139   groundedness check (entailment) per claim — Self-RAG
        │
        ├── grounded ───────────────▶ n_finalize   graph.py:146  → DONE (answer + citations)
        ├── ungrounded, budget left ▶ n_reformulate → n_retrieve
        └── ungrounded, no budget ──▶ n_abstain     graph.py:151  → "not stated in the document"
```

The two corrective cycles — `Grade→Reformulate→Retrieve` (fix retrieval) and
`Verify→Reformulate→Retrieve` (fix an ungrounded draft) — share **one** iteration counter and
budget, so the machine provably terminates. It can only exit three ways: sufficient evidence,
iteration cap hit, or deadline exceeded.

Supporting pieces:
- Answer generation + citation prompt: [grounding/generator.py:26](src/agrag/grounding/generator.py#L26)
- Groundedness verification: [grounding/verify.py:33](src/agrag/grounding/verify.py#L33)
  (the gray zone between "entailed" and "contradicted" deliberately abstains — bias to safety)

---

## 5 · Inside retrieval

`n_retrieve` delegates to the plan executor, which fans out the DAG concurrently:

```
PlanExecutor.run(plan)                               agent/plan_exec.py:46
  ├─ _layers(steps)          plan_exec.py:27   topological layers of the sub-step DAG
  ├─ independent steps run concurrently under a GPU-slot semaphore (C7/C8)
  ├─ CODE steps              plan_exec.py:112  offload table arithmetic to the sandbox (never the LLM)
  └─ each retrieval step ──▶ HybridRetriever.retrieve()   retrieval/hybrid.py
```

```
HybridRetriever.retrieve(query, strategy, k)         retrieval/hybrid.py
  ├─ dense ANN search  ─┐
  ├─ BM25 lexical      ─┴─▶ fuse with RRF (retrieval/rrf.py)
  ├─ dedupe near-duplicates (retrieval/dedupe.py, MinHash/LSH)
  └─ cross-encoder rerank → top-k
```

---

## 6 · The baseline control

[baseline/vanilla.py:36](src/agrag/baseline/vanilla.py#L36) is deliberately the *simplest possible*
RAG: `retrieve top-k → one grounded prompt → answer`. No routing, no grading, no verification, no
abstention. It sits behind the **same** interface as `AgentApp`, so it swaps in via one config flag
(`agent_mode: baseline`). Its entire purpose is to be the control the agentic path is measured
against — every capability is reported as a *delta over this baseline*.

---

## 7 · Local vs full mode

| | Local (`config/default.yaml`) | Full (`config/full.yaml`) |
|---|---|---|
| LLM | `FakeLLM` (deterministic) | Ollama + `gemma3:12b` / `gemma3:4b` |
| Embeddings | `HashEmbedding` (hashed) | BGE-M3 dense+sparse |
| Vector store | in-memory | Qdrant |
| Cache / doc store | in-memory | Redis |
| Verifier | lexical | NLI cross-encoder |
| Tracer | logging | Langfuse |
| Needs GPU / services? | No | Yes |

Same code path (`build_deps` → app) in both — only the concrete adapters differ.

---

## 8 · Suggested study order (and how to run it while reading)

Read in this order:

1. `config.py` · `container.py` · `deps.py` — how the object graph is assembled.
2. `interfaces/` — the contracts (`LLM`, `VectorStore`, `Retriever`, `Verifier`, …).
3. `cli.py` — the simplest driver of both flows.
4. `ingestion/service.py` + `ingestion/stages.py` — the offline plane.
5. `agent/graph.py` + `agent/plan_exec.py` — the agentic loop.
6. `baseline/vanilla.py` — the control, to see exactly what the agent adds.

Run these in **local mode** (no GPU, no services, fully deterministic) and watch the trace logs
as you read — they narrate each state the FSM enters:

```bash
make install
agrag ingest data/sample.txt
agrag ask "What was Acme's FY2023 revenue?" --path data/sample.txt
agrag eval        # runs BOTH the baseline and the agent end-to-end, prints the metric delta
```

`agrag eval` is the most instructive single command: it exercises ingestion, both query paths,
and the full metric suite in one run.
