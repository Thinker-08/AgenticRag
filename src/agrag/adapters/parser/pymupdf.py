"""PyMuPDF parser — cascading digital -> OCR -> vision extraction by text-ratio (03 §1).

Digital text (with block geometry + reading order) is preferred; scanned pages fall through to
Tesseract OCR, and pages OCR can't read fall through to multimodal vision transcription. Tables come
from pdfplumber, recurring headers/footers are stripped, and text is NFKC/ftfy-normalized.
"""

from __future__ import annotations

import asyncio
import importlib
import io
import re
import unicodedata
from collections import defaultdict

from ...config import ParserConfig
from ...contracts import Block, BlockType, Page, ParsedDoc, ParseTier, Table
from ...interfaces import LLM

_OCR_DPI = 200
_VISION_DPI = 150
_OCR_MIN_CHARS = 24
_LANG_MIN_CHARS = 20
_BOILERPLATE_MIN_PAGES = 3
_BOILERPLATE_RATIO = 0.4
_VISION_PROMPT = (
    "Transcribe all readable text from this scanned document page in natural reading order. "
    "Return only the transcription, with no commentary."
)


def _try_import(name: str):
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
        self._pdfplumber = _try_import("pdfplumber")
        self._pytesseract = _try_import("pytesseract")
        self._Image = _try_import("PIL.Image")
        if self._Image is None:
            self._pytesseract = None
        self._ftfy = _try_import("ftfy")
        self._lingua = _try_import("lingua")
        self._lang_detector = None

    async def parse(
        self, data: bytes, *, doc_id: str, tenant_id: str, filename: str = ""
    ) -> ParsedDoc:
        doc = await asyncio.to_thread(self._fitz.open, stream=data, filetype="pdf")
        try:
            if doc.needs_pass:
                raise ValueError(
                    f"cannot parse encrypted/password-protected PDF {filename or doc_id!r}"
                )
            page_count = doc.page_count
            if page_count > self._cfg.max_pages:
                raise ValueError(
                    f"PDF has {page_count} pages, exceeds max_pages={self._cfg.max_pages} "
                    "(suspected PDF-bomb)"
                )

            tables_by_page = await asyncio.to_thread(self._extract_tables, data, doc_id)

            pages: list[Page] = []
            for i in range(page_count):
                page, png = await asyncio.to_thread(self._process_page, doc, i, doc_id)
                if png is not None:
                    page = await self._vision_page(page, png, i, doc_id)
                self._merge_tables(page, tables_by_page.get(i, []))
                self._order_blocks(page)
                pages.append(page)

            self._strip_boilerplate(pages)
            summary = self._doc_summary(pages, filename)
        finally:
            doc.close()

        return ParsedDoc(
            doc_id=doc_id,
            tenant_id=tenant_id,
            content_hash="",
            filename=filename,
            page_count=len(pages),
            pages=pages,
            doc_summary=summary,
        )

    def _process_page(self, doc, i: int, doc_id: str) -> tuple[Page, bytes | None]:
        page = doc[i]
        rect = page.rect
        w, h = float(rect.width), float(rect.height)
        area_scaled = max(1.0, (w * h) / 1000.0)
        raw = page.get_text("text") or ""
        ratio = len(raw.strip()) / area_scaled
        text_ratio = round(min(1.0, ratio), 4)
        base = dict(
            page_no=i + 1, text_ratio=text_ratio, rotation=int(page.rotation), width=w, height=h
        )

        if ratio >= self._cfg.text_ratio_threshold:
            blocks = self._digital_blocks(page, i, doc_id, w)
            return Page(tier=ParseTier.DIGITAL, blocks=blocks, **base), None

        if self._pytesseract is not None:
            ocr_text = self._clean(self._ocr_page(page))
            if len(ocr_text.strip()) >= _OCR_MIN_CHARS:
                block = self._single_block(doc_id, i, ocr_text, (0.0, 0.0, w, h))
                return Page(tier=ParseTier.OCR, blocks=[block], **base), None

        png = self._render_png(page)
        return Page(tier=ParseTier.VISION, blocks=[], **base), png

    def _digital_blocks(self, page, i: int, doc_id: str, width: float) -> list[Block]:
        raw_blocks = page.get_text("blocks")
        text_blocks = [b for b in raw_blocks if len(b) >= 7 and b[6] == 0 and (b[4] or "").strip()]
        text_blocks.sort(key=lambda b: (self._col_bucket(b[0], width), b[1]))
        blocks: list[Block] = []
        for order, b in enumerate(text_blocks):
            clean = self._clean(b[4])
            if not clean.strip():
                continue
            blocks.append(
                Block(
                    block_id=f"{doc_id}:p{i + 1}:b{order}",
                    page=i + 1,
                    type=BlockType.PARAGRAPH,
                    bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                    text=clean,
                    reading_order=order,
                    lang=self._detect_lang(clean),
                )
            )
        return blocks

    def _single_block(
        self, doc_id: str, i: int, text: str, bbox: tuple[float, float, float, float]
    ) -> Block:
        return Block(
            block_id=f"{doc_id}:p{i + 1}:b0",
            page=i + 1,
            type=BlockType.PARAGRAPH,
            bbox=bbox,
            text=text,
            reading_order=0,
            lang=self._detect_lang(text),
        )

    def _ocr_page(self, page) -> str:
        pix = page.get_pixmap(dpi=_OCR_DPI, alpha=False)
        img = self._Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return self._pytesseract.image_to_string(img) or ""

    def _render_png(self, page) -> bytes:
        pix = page.get_pixmap(dpi=_VISION_DPI, alpha=False)
        return pix.tobytes("png")

    async def _vision_page(self, page: Page, png: bytes, i: int, doc_id: str) -> Page:
        if self._vision_llm is None:
            return page
        try:
            result = await self._vision_llm.generate(_VISION_PROMPT, images=[png], max_tokens=2048)
            text = self._clean(result.text)
        except Exception:
            text = ""
        if text.strip():
            page.blocks = [self._single_block(doc_id, i, text, (0.0, 0.0, page.width, page.height))]
        return page

    def _extract_tables(self, data: bytes, doc_id: str) -> dict[int, list[Table]]:
        if self._pdfplumber is None:
            return {}
        out: dict[int, list[Table]] = {}
        try:
            with self._pdfplumber.open(io.BytesIO(data)) as pdf:
                for i, ppage in enumerate(pdf.pages):
                    tables: list[Table] = []
                    for j, found in enumerate(ppage.find_tables()):
                        grid = found.extract() or []
                        grid = [[(c or "").strip() for c in row] for row in grid if row]
                        if not grid:
                            continue
                        bbox = tuple(float(v) for v in found.bbox)
                        tables.append(
                            Table(
                                block_id=f"{doc_id}:p{i + 1}:tbl{j}",
                                page=i + 1,
                                bbox=bbox,
                                grid=grid,
                                n_header_rows=1,
                            )
                        )
                    if tables:
                        out[i] = tables
        except Exception:
            return {}
        return out

    def _merge_tables(self, page: Page, tables: list[Table]) -> None:
        for t in tables:
            page.blocks.append(
                Block(
                    block_id=t.block_id,
                    page=t.page,
                    type=BlockType.TABLE,
                    bbox=t.bbox,
                    text=t.linearize(),
                    table=t,
                )
            )

    def _order_blocks(self, page: Page) -> None:
        page.blocks.sort(key=lambda b: (self._col_bucket(b.bbox[0], page.width), b.bbox[1]))
        for order, b in enumerate(page.blocks):
            b.reading_order = order

    @staticmethod
    def _col_bucket(x0: float, width: float) -> int:
        return 0 if width <= 0 or x0 < width * 0.5 else 1

    def _strip_boilerplate(self, pages: list[Page]) -> None:
        if len(pages) < _BOILERPLATE_MIN_PAGES:
            return
        key_pages: dict[tuple, set[int]] = defaultdict(set)
        for p in pages:
            for b in p.blocks:
                if b.type == BlockType.TABLE:
                    continue
                key_pages[self._boiler_key(b)].add(p.page_no)
        n = len(pages)
        boiler = {k for k, ps in key_pages.items() if len(ps) / n > _BOILERPLATE_RATIO}
        if not boiler:
            return
        for p in pages:
            p.blocks = [
                b
                for b in p.blocks
                if b.type == BlockType.TABLE or self._boiler_key(b) not in boiler
            ]
            for order, b in enumerate(p.blocks):
                b.reading_order = order

    @staticmethod
    def _boiler_key(b: Block) -> tuple:
        norm = re.sub(r"\s+", " ", b.text.strip().lower())
        return (
            round(b.bbox[0] / 2.0),
            round(b.bbox[1] / 2.0),
            round(b.bbox[2] / 2.0),
            round(b.bbox[3] / 2.0),
            norm,
        )

    def _clean(self, text: str) -> str:
        if self._ftfy is not None:
            text = self._ftfy.fix_text(text)
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        return text

    def _detect_lang(self, text: str) -> str:
        if self._lingua is None or len(text.strip()) < _LANG_MIN_CHARS:
            return "en"
        if self._lang_detector is None:
            self._lang_detector = self._lingua.LanguageDetectorBuilder.from_all_languages().build()
        lang = self._lang_detector.detect_language_of(text)
        code = getattr(lang, "iso_code_639_1", None) if lang is not None else None
        return code.name.lower() if code is not None else "en"

    def _doc_summary(self, pages: list[Page], filename: str) -> str:
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
