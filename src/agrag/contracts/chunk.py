from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ChunkKind(str, Enum):
    PROSE = "prose"
    FIGURE = "figure"
    EQUATION = "equation"
    LIST = "list"
    CODE = "code"


class Chunk(BaseModel):
    model_config = {"frozen": True}

    chunk_id: str
    doc_id: str
    tenant_id: str
    page_no: int
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    section_breadcrumb: list[str] = Field(default_factory=list)
    kind: ChunkKind = ChunkKind.PROSE
    atomic: bool = False
    text: str
    linearized_text: str = ""
    context_blurb: str = ""
    parent_id: Optional[str] = None
    embedding_model: str = ""
    embedding_version: str = ""
    content_hash: str = ""
    lang: str = "en"
    extra_metadata: dict = Field(default_factory=dict)

    def embedInput(self) -> str:
        body = self.linearized_text or self.text
        return f"{self.context_blurb}\n\n{body}".strip() if self.context_blurb else body
