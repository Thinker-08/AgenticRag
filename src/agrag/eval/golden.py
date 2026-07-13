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


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def load_golden(path: str | Path) -> list[GoldenItem]:
    return [GoldenItem(**row) for row in _read_jsonl(Path(path))]


def _load_corpus_file(f: Path) -> list[GoldenCorpusDoc]:
    if f.suffix == ".jsonl":
        return [GoldenCorpusDoc(**row) for row in _read_jsonl(f)]
    if f.suffix == ".txt":
        return [GoldenCorpusDoc(doc_id=f.stem, filename=f.name, text=f.read_text())]
    return []


def load_corpus(dir_or_file: str | Path) -> list[GoldenCorpusDoc]:
    p = Path(dir_or_file)
    if p.is_dir():
        docs: list[GoldenCorpusDoc] = []
        for f in sorted(p.iterdir()):
            docs.extend(_load_corpus_file(f))
        return docs
    return _load_corpus_file(p)
