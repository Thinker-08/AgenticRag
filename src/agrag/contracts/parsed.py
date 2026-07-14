from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class BlockType(str, Enum):
    TITLE = "title"
    PARAGRAPH = "paragraph"
    LIST = "list"
    FIGURE = "figure"
    CAPTION = "caption"
    EQUATION = "equation"
    HEADER = "header"
    FOOTER = "footer"


class ParseTier(str, Enum):
    DIGITAL = "digital"
    OCR = "ocr"
    VISION = "vision"


class Block(BaseModel):
    block_id: str
    page: int
    type: BlockType = BlockType.PARAGRAPH
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    text: str = ""
    reading_order: int = 0
    lang: str = "en"
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
    doc_title: str = ""
    extra_metadata: dict = Field(default_factory=dict)
