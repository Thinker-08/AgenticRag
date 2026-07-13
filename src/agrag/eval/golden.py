from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class GoldenItem(BaseModel):
    model_config = {"frozen": True}

    question: str
    gold_answer: str = ""
    gold_chunk_ids: list[str] = Field(default_factory=list)
    answerable: bool = True
    dataset: str = ""
    intent: str = ""
    corpus_doc: str = ""


class GoldenCorpusDoc(BaseModel):
    model_config = {"frozen": True}

    doc_id: str
    filename: str = ""
    text: str


def readJsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def loadGolden(path: str | Path) -> list[GoldenItem]:
    return [GoldenItem(**row) for row in readJsonl(Path(path))]


def loadCorpusFile(f: Path) -> list[GoldenCorpusDoc]:
    if f.suffix == ".jsonl":
        return [GoldenCorpusDoc(**row) for row in readJsonl(f)]
    if f.suffix == ".txt":
        return [GoldenCorpusDoc(doc_id=f.stem, filename=f.name, text=f.read_text())]
    return []


def loadCorpus(dir_or_file: str | Path) -> list[GoldenCorpusDoc]:
    p = Path(dir_or_file)
    if p.is_dir():
        docs: list[GoldenCorpusDoc] = []
        for f in sorted(p.iterdir()):
            docs.extend(loadCorpusFile(f))
        return docs
    return loadCorpusFile(p)
