from __future__ import annotations

import re
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
    DIGITAL = "digital"
    OCR = "ocr"
    VISION = "vision"


_NUM_STRIP = re.compile(r"[$€£,\s]")


def cellNum(raw: str) -> float | None:
    s = _NUM_STRIP.sub("", raw.strip())
    if not s or s in ("-", "—"):
        return None

    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").rstrip("%")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


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

        for row in self.grid[self.n_header_rows :]:
            cells = [f"{cols[i].strip()} {v}" for i, v in enumerate(row) if i < len(cols)]
            lines.append(" — ".join(cells) + ".")

        return "\n".join(lines)

    def validateChecksum(self) -> bool:
        rows = self.grid[self.n_header_rows :]
        headers = self.grid[: self.n_header_rows]
        total_idx = next((i for i in range(len(rows) - 1, -1, -1) if rows[i] and rows[i][0].strip().lower().startswith(("total", "sum"))), None)
        if total_idx is None or total_idx == 0:
            self.checksum_ok = True
            return True

        data, total_row = rows[:total_idx], rows[total_idx]
        ok = True
        for c in range(1, max(len(r) for r in self.grid)):
            header_text = " ".join(h[c] for h in headers if c < len(h))
            if "%" in header_text:
                continue
            cells = [cellNum(r[c]) for r in data if c < len(r)]
            values = [v for v in cells if v is not None]
            if any(c < len(r) and r[c].strip().endswith("%") for r in data):
                continue
            total = cellNum(total_row[c]) if c < len(total_row) else None
            if total is None or len(values) < 2:
                continue
            if abs(sum(values) - total) > max(0.6, abs(total) * 0.015):
                ok = False
                break

        self.checksum_ok = ok
        return ok


class Block(BaseModel):
    block_id: str
    page: int
    type: BlockType = BlockType.PARAGRAPH
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    text: str = ""
    reading_order: int = 0
    lang: str = "en"
    table: Optional[Table] = None
    image_ref: Optional[str] = None
    breadcrumb: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)


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
    doc_summary: str = ""
    extra_metadata: dict = Field(default_factory=dict)
