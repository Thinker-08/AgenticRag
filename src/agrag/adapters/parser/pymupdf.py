from __future__ import annotations

import asyncio
import importlib
import re
import unicodedata

from ...config import ParserConfig
from ...contracts import Block, BlockType, Page, ParsedDoc, ParseTier
from ...interfaces import LLM
from ...security.sanitize import neutralizeTemplateTokens
from .mathdetect import looksLikeEquation

_OCR_DPI = 200
_VISION_DPI = 150
_VISION_MAX_TOKENS = 4096
_MAX_RENDER_PX = 40_000_000
_OCR_MIN_CHARS = 24
_LANG_MIN_CHARS = 20
_VISION_PROMPT = "Transcribe all readable text from this scanned document page in natural reading order. Return only the transcription, with no commentary."


def tryImport(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


class PymupdfParser:
    def __init__(self, cfg: ParserConfig, vision_llm: LLM) -> None:
        self._cfg = cfg
        self._vision_llm = vision_llm

        try:
            import fitz
        except ImportError as e:
            raise ImportError("PymupdfParser needs the 'pdf' extra: pip install -e '.[pdf]'") from e
        self._fitz = fitz

        self._pytesseract = tryImport("pytesseract")
        self._Image = tryImport("PIL.Image")
        if self._Image is None:
            self._pytesseract = None
        self._ftfy = tryImport("ftfy")
        self._lingua = tryImport("lingua")
        self._lang_detector = None

    async def parse(self, data: bytes, *, doc_id: str, tenant_id: str, filename: str = "") -> ParsedDoc:
        doc = await asyncio.to_thread(self._fitz.open, stream=data, filetype="pdf")
        try:
            if doc.needs_pass:
                raise ValueError(f"cannot parse encrypted/password-protected PDF {filename or doc_id!r}")
            page_count = doc.page_count
            if page_count > self._cfg.max_pages:
                raise ValueError(f"PDF has {page_count} pages, exceeds max_pages={self._cfg.max_pages} (suspected PDF-bomb)")

            pages: list[Page] = []
            extracted_chars = 0
            for i in range(page_count):
                page, png = await asyncio.to_thread(self.processPage, doc, i, doc_id)
                if png is not None:
                    page = await self.visionPage(page, png, i, doc_id)
                self.orderBlocks(page)
                pages.append(page)

                extracted_chars += sum(len(b.text) for b in page.blocks)
                if len(data) and extracted_chars / len(data) > self._cfg.max_inflate_ratio:
                    raise ValueError(f"decompressed/input ratio exceeds max_inflate_ratio={self._cfg.max_inflate_ratio} (suspected inflate bomb)")

            summary = self.docSummary(pages, filename)
        finally:
            doc.close()

        return ParsedDoc(doc_id=doc_id, tenant_id=tenant_id, content_hash="", filename=filename, page_count=len(pages), pages=pages, doc_summary=summary)

    def processPage(self, doc, i: int, doc_id: str) -> tuple[Page, bytes | None]:
        page = doc[i]
        rect = page.rect
        w, h = float(rect.width), float(rect.height)
        area_scaled = max(1.0, (w * h) / 1000.0)
        raw = page.get_text("text") or ""
        ratio = len(raw.strip()) / area_scaled
        text_ratio = round(min(1.0, ratio), 4)
        base = dict(page_no=i + 1, text_ratio=text_ratio, rotation=int(page.rotation), width=w, height=h)

        if ratio >= self._cfg.text_ratio_threshold:
            blocks = self.digitalBlocks(page, i, doc_id, w)
            if blocks:
                return Page(tier=ParseTier.DIGITAL, blocks=blocks, **base), None

        if self._pytesseract is not None:
            try:
                ocr_text = self.clean(self.ocrPage(page))
            except Exception:
                ocr_text = ""
            if len(ocr_text.strip()) >= _OCR_MIN_CHARS:
                block = self.singleBlock(doc_id, i, ocr_text, (0.0, 0.0, w, h))
                return Page(tier=ParseTier.OCR, blocks=[block], **base), None

        png = self.renderPng(page)
        return Page(tier=ParseTier.VISION, blocks=[], **base), png

    def digitalBlocks(self, page, i: int, doc_id: str, width: float) -> list[Block]:
        raw_blocks = page.get_text("blocks")
        text_blocks = [b for b in raw_blocks if len(b) >= 7 and b[6] == 0 and (b[4] or "").strip()]
        text_blocks.sort(key=lambda b: (self.colBucket(b[0], width), b[1]))

        blocks: list[Block] = []
        for order, b in enumerate(text_blocks):
            clean = self.clean(b[4])
            if not clean.strip():
                continue
            btype = BlockType.EQUATION if looksLikeEquation(clean) else BlockType.PARAGRAPH
            blocks.append(Block(block_id=f"{doc_id}:p{i + 1}:b{order}", page=i + 1, type=btype, bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])), text=clean, reading_order=order, lang=self.detectLang(clean)))

        return blocks

    def singleBlock(self, doc_id: str, i: int, text: str, bbox: tuple[float, float, float, float]) -> Block:
        return Block(block_id=f"{doc_id}:p{i + 1}:b0", page=i + 1, type=BlockType.PARAGRAPH, bbox=bbox, text=text, reading_order=0, lang=self.detectLang(text))

    def dpiFor(self, page, base_dpi: int) -> int:
        rect = page.rect
        pts = max(1.0, float(rect.width)) * max(1.0, float(rect.height))
        px = pts * (base_dpi / 72.0) ** 2

        if px > _MAX_RENDER_PX:
            return max(36, int(base_dpi * (_MAX_RENDER_PX / px) ** 0.5))
        return base_dpi

    def ocrPage(self, page) -> str:
        pix = page.get_pixmap(dpi=self.dpiFor(page, _OCR_DPI), alpha=False)
        img = self._Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return self._pytesseract.image_to_string(img) or ""

    def renderPng(self, page) -> bytes:
        pix = page.get_pixmap(dpi=self.dpiFor(page, _VISION_DPI), alpha=False)
        return pix.tobytes("png")

    async def visionPage(self, page: Page, png: bytes, i: int, doc_id: str) -> Page:
        if self._vision_llm is None:
            return page

        try:
            result = await self._vision_llm.generate(_VISION_PROMPT, images=[png], max_tokens=_VISION_MAX_TOKENS)
            text = self.clean(result.text)
        except Exception:
            text = ""

        if text.strip():
            page.blocks = [self.singleBlock(doc_id, i, text, (0.0, 0.0, page.width, page.height))]
        return page

    def orderBlocks(self, page: Page) -> None:
        page.blocks.sort(key=lambda b: (self.colBucket(b.bbox[0], page.width), b.bbox[1]))
        for order, b in enumerate(page.blocks):
            b.reading_order = order

    @staticmethod
    def colBucket(x0: float, width: float) -> int:
        return 0 if width <= 0 or x0 < width * 0.5 else 1

    def clean(self, text: str) -> str:
        if self._ftfy is not None:
            text = self._ftfy.fix_text(text)
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        return neutralizeTemplateTokens(text)

    def detectLang(self, text: str) -> str:
        if self._lingua is None or len(text.strip()) < _LANG_MIN_CHARS:
            return "en"

        if self._lang_detector is None:
            self._lang_detector = self._lingua.LanguageDetectorBuilder.from_all_languages().build()

        lang = self._lang_detector.detect_language_of(text)
        code = getattr(lang, "iso_code_639_1", None) if lang is not None else None
        return code.name.lower() if code is not None else "en"

    def docSummary(self, pages: list[Page], filename: str) -> str:
        title = ""
        for p in pages:
            for b in p.blocks:
                t = b.text.strip()
                if t and "\n" not in t and len(t) <= 120:
                    b.type = BlockType.TITLE
                    title = t
                    break
            if title:
                break

        parts = [x for x in (title, filename) if x]
        return " — ".join(parts)
