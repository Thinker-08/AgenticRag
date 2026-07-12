"""Intermediate ingestion artifacts passed between chain-of-responsibility stages (03)."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class BlockType(str, Enum):
    TITLE = "title"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    EQUATION = "equation"
    HEADER = "header"
    FOOTER = "footer"


class ParseTier(str, Enum):
    DIGITAL = "digital"     # PyMuPDF text layer
    OCR = "ocr"             # Tesseract/PaddleOCR
    VISION = "vision"       # multimodal Gemma


class Table(BaseModel):
    block_id: str
    page: int
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    title: str = ""
    n_header_rows: int = 1
    grid: list[list[str]] = Field(default_factory=list)
    merged_cells: list[dict] = Field(default_factory=list)
    units: str = ""
    checksum_ok: bool = True

    def linearize(self) -> str:
        if not self.grid:
            return self.title
        header = self.grid[: self.n_header_rows]
        cols = [" ".join(h[c] for h in header if c < len(h)) for c in range(len(self.grid[0]))]
        lines = [f"Table: {self.title}." if self.title else "Table."]
        lines.append("Columns: " + "; ".join(c.strip() for c in cols) + ".")
        for row in self.grid[self.n_header_rows:]:
            cells = [f"{cols[i].strip()} {v}" for i, v in enumerate(row) if i < len(cols)]
            lines.append(" — ".join(cells) + ".")
        return "\n".join(lines)


class Block(BaseModel):
    block_id: str
    page: int
    type: BlockType = BlockType.PARAGRAPH
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    text: str = ""
    reading_order: int = 0
    lang: str = "en"
    table: Optional[Table] = None
    image_ref: Optional[str] = None       # crop retained for answer-time vision
    breadcrumb: list[str] = Field(default_factory=list)


class Page(BaseModel):
    page_no: int
    text_ratio: float = 1.0
    tier: ParseTier = ParseTier.DIGITAL
    rotation: int = 0
    width: float = 0.0
    height: float = 0.0
    image_hash: str = ""
    blocks: list[Block] = Field(default_factory=list)


class ParsedDoc(BaseModel):
    doc_id: str
    tenant_id: str
    content_hash: str
    filename: str = ""
    page_count: int = 0
    pages: list[Page] = Field(default_factory=list)
    doc_summary: str = ""                  # cached prompt prefix for contextualize
    extra_metadata: dict = Field(default_factory=dict)
