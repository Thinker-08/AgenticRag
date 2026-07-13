from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .config import loadSettings
from .container import buildDeps
from .ingestion.service import IngestionService


def makeSettings(args):
    return loadSettings(args.config)


async def ingest(args) -> None:
    deps = buildDeps(makeSettings(args))
    ingestion = IngestionService(deps)
    path = Path(args.path)
    data = path.read_bytes()

    doc = await ingestion.ingest(data, tenant_id=args.tenant, filename=path.name)

    print(json.dumps({"doc_id": doc.doc_id, "status": doc.status, "pages": doc.page_count, "chunks": await deps.vectorstore.count(args.tenant), "error": doc.error}, indent=2))


async def ask(args) -> None:
    from .agent.app import AgentApp
    from .baseline.vanilla import BaselineRAG

    settings = makeSettings(args)
    deps = buildDeps(settings)
    app = BaselineRAG(deps) if settings.isBaseline() else AgentApp(deps)

    if args.path:
        ingestion = IngestionService(deps)
        doc_path = Path(args.path)
        await ingestion.ingest(doc_path.read_bytes(), tenant_id=args.tenant, filename=doc_path.name)

    ans = await app.answer(args.query, tenant_id=args.tenant)
    print(f"\n[{ans.status.value}] {ans.answer_text}\n")

    for c in ans.claims:
        for cit in c.citations:
            print(f'  · {cit.chunk_id} (p{cit.page_no}): "{cit.quote[:80]}"')
    for comp in ans.computations:
        print(f"  Σ {comp.code} = {comp.result}  [{comp.sandbox_run_id}]")
    if ans.gaps:
        print("  gaps:", "; ".join(ans.gaps))


async def eval(args) -> None:
    from .eval.golden import loadCorpus, loadGolden
    from .eval.harness import EvalHarness

    corpus = loadCorpus(args.corpus)
    golden = loadGolden(args.golden)
    settings = makeSettings(args)
    baseline_settings = settings.model_copy(update={"agent_mode": "baseline", "chunker": settings.chunker.model_copy(update={"provider": "recursive"})})
    agentic_settings = settings.model_copy(update={"agent_mode": "agentic"})

    result = await EvalHarness(agentic_settings, corpus=corpus).compare(baseline_settings, agentic_settings, golden, corpus=corpus, tenant_id=args.tenant)
    out = {"baseline": result["baseline"].aggregate, "agentic": result["agentic"].aggregate, "delta": result["delta"]}

    if args.gate:
        from .eval.gate import promote

        control = loadControl(args.control) or result["baseline"].aggregate
        try:
            out["gate"] = promote("agentic", result["agentic"].aggregate, control)
        except AssertionError as exc:
            print(json.dumps(out, indent=2, default=str))
            print(f"\nGATE FAILED: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

    print(json.dumps(out, indent=2, default=str))


def loadControl(path: str | None) -> dict | None:
    if not path:
        return None

    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


async def calibrate(args) -> None:
    from .eval.calibration import cohensKappa, sweepTau

    rows = [json.loads(ln) for ln in Path(args.labels).read_text().splitlines() if ln.strip()]
    scored = [(float(r["entail_score"]), bool(r["is_grounded"])) for r in rows if "entail_score" in r]
    best_tau, curve = sweepTau(scored, beta=args.beta) if scored else (None, [])
    out: dict = {"best_tau": best_tau, "curve": [c.__dict__ for c in curve]}

    judged = [(bool(r["judge"]), bool(r["human"])) for r in rows if "judge" in r and "human" in r]
    if judged:
        out["cohens_kappa"] = round(cohensKappa([j for j, _ in judged], [h for _, h in judged]), 4)
        out["kappa_gate_ok"] = out["cohens_kappa"] >= args.kappa_min

    print(json.dumps(out, indent=2))


async def recall(args) -> None:
    from .eval.golden import loadCorpus
    from .eval.harness import EvalHarness
    from .eval.recall import measureAnnRecall

    settings = makeSettings(args)
    harness = EvalHarness(settings, corpus=loadCorpus(args.corpus))

    await harness.ingestCorpus(harness.corpus, tenant_id=args.tenant)

    print(json.dumps(await measureAnnRecall(harness.deps, tenant_id=args.tenant, k=args.k), indent=2))


def serve(args) -> None:
    import uvicorn

    settings = makeSettings(args)
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
    eval_parser.add_argument("--gate", action="store_true", help="exit 1 if agentic regresses the control")
    eval_parser.add_argument("--control", default=None, help="optional frozen-floors JSON; default gates vs the live baseline re-run (C27)")

    cal_parser = sub.add_parser("calibrate", parents=[common], help="sweep TAU + judge kappa (C28)")
    cal_parser.add_argument("--labels", required=True, help="JSONL of labeled entail/judge/human rows")
    cal_parser.add_argument("--beta", type=float, default=0.5)
    cal_parser.add_argument("--kappa-min", type=float, default=0.6)

    recall_parser = sub.add_parser("recall", parents=[common], help="measure ANN recall@k (C1)")
    recall_parser.add_argument("--corpus", default="data/golden/sample_corpus.jsonl")
    recall_parser.add_argument("--k", type=int, default=10)

    sub.add_parser("serve", parents=[common], help="run the FastAPI server")

    args = parser.parse_args(argv)
    if args.cmd == "serve":
        serve(args)
        return 0

    runner = {"ingest": ingest, "ask": ask, "eval": eval, "calibrate": calibrate, "recall": recall}[args.cmd]
    asyncio.run(runner(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
