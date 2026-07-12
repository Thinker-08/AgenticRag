"""`Chunk` — the atomic unit of retrieval and the API between the two planes (02 §4.1)."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ChunkKind(str, Enum):
    PROSE = "prose"
    TABLE = "table"
    FIGURE = "figure"
    EQUATION = "equation"
    LIST = "list"
    CODE = "code"
    SUMMARY = "summary"       # RAPTOR node


class Chunk(BaseModel):
    model_config = {"frozen": True}

    chunk_id: str
    doc_id: str
    tenant_id: str                                    # HARD isolation key, enforced at query layer (C31)
    page_no: int
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    section_breadcrumb: list[str] = Field(default_factory=list)
    kind: ChunkKind = ChunkKind.PROSE
    atomic: bool = False
    text: str                                         # human/LLM-facing rendering; normalized source of truth for spans
    linearized_text: str = ""                         # flattened form for embedding + BM25
    context_blurb: str = ""                           # contextual-retrieval situating sentence
    parent_id: Optional[str] = None                   # small-to-big
    linked_block: Optional[str] = None                # FK to structured table/figure object
    embedding_model: str = ""                         # --- CONTRACT (C3)
    embedding_version: str = ""                       # --- CONTRACT (C3)
    content_hash: str = ""                            # sha256(normalized span) --- idempotency/invalidation
    lang: str = "en"                                  # BCP-47
    extra_metadata: dict = Field(default_factory=dict)  # dates, doc_type, fiscal_year, numeric spans

    def embed_input(self) -> str:
        body = self.linearized_text or self.text
        return f"{self.context_blurb}\n\n{body}".strip() if self.context_blurb else body

    def embed_cache_key(self) -> str:
        return f"{self.content_hash}:{self.embedding_version}"
