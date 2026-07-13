from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel, Field

from ..config import Settings, loadSettings
from ..container import buildDeps
from ..contracts import Answer, AnswerStatus
from ..deps import Deps
from .golden import GoldenCorpusDoc, GoldenItem
from .metrics import citationAccuracy, contextPrecision, correctRefusal, exactMatch, faithfulness, metricsOver, overAbstention, recallAtK, tokenF1


class EvalItemResult(BaseModel):
    item: GoldenItem
    answer: Answer
    metrics: dict[str, float] = Field(default_factory=dict)


class EvalReport(BaseModel):
    label: str = ""
    items: list[EvalItemResult] = Field(default_factory=list)
    aggregate: dict[str, float] = Field(default_factory=dict)


class EvalHarness:
    def __init__(self, deps_or_settings: Deps | Settings | None = None, *, corpus: Iterable[GoldenCorpusDoc] | None = None) -> None:
        if isinstance(deps_or_settings, Deps):
            self.deps = deps_or_settings
        else:
            self.deps = buildDeps(deps_or_settings or loadSettings())

        self.corpus: list[GoldenCorpusDoc] = list(corpus or [])
        self._app = None
        self._ingestion = None
        self._ingested: set[tuple[str, str]] = set()

    def getApp(self):
        if self._app is None:
            if self.deps.settings.isBaseline():
                from ..baseline.vanilla import BaselineRAG

                self._app = BaselineRAG(self.deps)
            else:
                from ..agent.app import AgentApp

                self._app = AgentApp(self.deps)
        return self._app

    def getIngestion(self):
        if self._ingestion is None:
            from ..ingestion.service import IngestionService

            self._ingestion = IngestionService(self.deps)
        return self._ingestion

    async def ingestCorpus(self, items: Iterable[GoldenCorpusDoc], *, tenant_id: str = "default") -> list:
        ingestion = self.getIngestion()
        docs = []

        for cd in items:
            key = (tenant_id, cd.doc_id)
            if key in self._ingested:
                continue
            doc = await ingestion.ingestText(cd.text, tenant_id=tenant_id, filename=cd.filename or cd.doc_id, doc_id=cd.doc_id)
            self._ingested.add(key)
            docs.append(doc)

        return docs

    def score(self, answer: Answer, item: GoldenItem, k: int) -> dict[str, float]:
        metrics: dict[str, float] = {"abstained": float(answer.status == AnswerStatus.ABSTAINED)}
        cited_ids = [c.chunk_id for c in answer.sources()]

        if item.answerable:
            if item.gold_answer:
                metrics["token_f1"] = tokenF1(answer.answer_text, item.gold_answer)
                metrics["exact_match"] = exactMatch(answer.answer_text, item.gold_answer)
            metrics["over_abstention"] = float(overAbstention(answer, item))
        else:
            metrics["correct_refusal"] = float(correctRefusal(answer, item))

        if item.gold_chunk_ids:
            metrics["recall_at_k"] = recallAtK(cited_ids, item.gold_chunk_ids, k)
            metrics["context_precision"] = contextPrecision(cited_ids, item.gold_chunk_ids)

        if answer.status != AnswerStatus.ABSTAINED:
            metrics["faithfulness"] = faithfulness(answer)
            metrics["citation_accuracy"] = citationAccuracy(answer)

        return metrics

    async def run(self, golden_items: Iterable[GoldenItem], *, tenant_id: str = "default", k: int | None = None, label: str = "") -> EvalReport:
        app = self.getApp()
        k = k or self.deps.settings.retrieval.top_k
        results: list[EvalItemResult] = []

        for item in golden_items:
            answer = await app.answer(item.question, tenant_id=tenant_id)
            metrics = self.score(answer, item, k)
            results.append(EvalItemResult(item=item, answer=answer, metrics=metrics))

        aggregate = metricsOver([r.metrics for r in results])
        return EvalReport(label=label or self.deps.settings.agent_mode, items=results, aggregate=aggregate)

    async def compare(self, baseline_settings: Settings, agentic_settings: Settings, golden: Iterable[GoldenItem], *, corpus: Iterable[GoldenCorpusDoc] | None = None, tenant_id: str = "default") -> dict:
        corpus = list(corpus if corpus is not None else self.corpus)
        golden = list(golden)

        baseline = EvalHarness(baseline_settings, corpus=corpus)
        await baseline.ingestCorpus(corpus, tenant_id=tenant_id)
        base_report = await baseline.run(golden, tenant_id=tenant_id, label="baseline")

        agentic = EvalHarness(agentic_settings, corpus=corpus)
        await agentic.ingestCorpus(corpus, tenant_id=tenant_id)
        ag_report = await agentic.run(golden, tenant_id=tenant_id, label="agentic")

        keys = set(base_report.aggregate) | set(ag_report.aggregate)
        delta = {key: round(ag_report.aggregate.get(key, 0.0) - base_report.aggregate.get(key, 0.0), 4) for key in sorted(keys)}
        return {"baseline": base_report, "agentic": ag_report, "delta": delta}
