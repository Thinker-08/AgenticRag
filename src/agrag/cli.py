"""agrag CLI: ingest a document, ask a question, run the eval delta, or serve the API."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .config import load_settings
from .container import build_deps
from .ingestion.service import IngestionService


def _settings(args):
    return load_settings(args.config)


async def _ingest(args) -> None:
    deps = build_deps(_settings(args))
    ingestion = IngestionService(deps)
    path = Path(args.path)
    data = path.read_bytes()
    doc = await ingestion.ingest(data, tenant_id=args.tenant, filename=path.name)
    print(json.dumps({"doc_id": doc.doc_id, "status": doc.status, "pages": doc.page_count,
                      "chunks": await deps.vectorstore.count(args.tenant), "error": doc.error}, indent=2))


async def _ask(args) -> None:
    from .agent.app import AgentApp
    from .baseline.vanilla import BaselineRAG
    settings = _settings(args)
    deps = build_deps(settings)
    app = BaselineRAG(deps) if settings.is_baseline() else AgentApp(deps)
    if args.path:
        ingestion = IngestionService(deps)
        doc_path = Path(args.path)
        await ingestion.ingest(doc_path.read_bytes(), tenant_id=args.tenant, filename=doc_path.name)
    ans = await app.answer(args.query, tenant_id=args.tenant)
    print(f"\n[{ans.status.value}] {ans.answer_text}\n")
    for c in ans.claims:
        for cit in c.citations:
            print(f"  · {cit.chunk_id} (p{cit.page_no}): \"{cit.quote[:80]}\"")
    for comp in ans.computations:
        print(f"  Σ {comp.code} = {comp.result}  [{comp.sandbox_run_id}]")
    if ans.gaps:
        print("  gaps:", "; ".join(ans.gaps))


async def _eval(args) -> None:
    from .eval.golden import load_corpus, load_golden
    from .eval.harness import EvalHarness
    corpus = load_corpus(args.corpus)
    golden = load_golden(args.golden)
    settings = _settings(args)
    # the control is vanilla by construction: fixed-size chunks, single-shot (page 13, step 1)
    baseline_settings = settings.model_copy(update={
        "agent_mode": "baseline",
        "chunker": settings.chunker.model_copy(update={"provider": "recursive"}),
    })
    agentic_settings = settings.model_copy(update={"agent_mode": "agentic"})
    result = await EvalHarness(agentic_settings, corpus=corpus).compare(baseline_settings, agentic_settings, golden, corpus=corpus, tenant_id=args.tenant)
    out = {
        "baseline": result["baseline"].aggregate,
        "agentic": result["agentic"].aggregate,
        "delta": result["delta"],
    }
    print(json.dumps(out, indent=2, default=str))


def _serve(args) -> None:
    import uvicorn
    settings = _settings(args)
    uvicorn.run("agrag.serving.app:app", host=settings.serving.host, port=settings.serving.port)


def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None, help="config yaml (else $AGRAG_CONFIG or config/default.yaml)")
    common.add_argument("--tenant", default="default")

    parser = argparse.ArgumentParser(prog="agrag", parents=[common])
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest_parser = sub.add_parser("ingest", parents=[common], help="ingest a document")
    ingest_parser.add_argument("path")

    ask_parser = sub.add_parser("ask", parents=[common], help="ask a question")
    ask_parser.add_argument("query")
    ask_parser.add_argument("--path", help="ingest this doc first")

    eval_parser = sub.add_parser("eval", parents=[common], help="run the eval delta (baseline vs agent)")
    eval_parser.add_argument("--golden", default="data/golden/sample.jsonl")
    eval_parser.add_argument("--corpus", default="data/golden/sample_corpus.jsonl")
    sub.add_parser("serve", parents=[common], help="run the FastAPI server")

    args = parser.parse_args(argv)
    if args.cmd == "serve":
        _serve(args)
        return 0
    runner = {"ingest": _ingest, "ask": _ask, "eval": _eval}[args.cmd]
    asyncio.run(runner(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
