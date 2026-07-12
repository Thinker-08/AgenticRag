"""Plain-text parser — the dependency-free local path (no PDF libs, no GPU).

Splits text into pages/blocks, recovers simple pipe/whitespace tables as structured Table objects,
and tracks section titles into breadcrumbs. Handles the local eval corpus and any .txt/.md upload.
"""

from __future__ import annotations

import re
import unicodedata

from ...contracts import Block, BlockType, Page, ParsedDoc, ParseTier, Table
from ...security.sanitize import neutralize_template_tokens
from .mathdetect import looks_like_equation

_TABLE_ROW = re.compile(r".+\|.+|.+\t.+|.+ {2,}.+")


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return neutralize_template_tokens(text)   # after NFKC so homoglyphs can't reassemble a token


def _split_row(line: str) -> list[str]:
    if "|" in line:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
    elif "\t" in line:
        cells = [c.strip() for c in line.split("\t")]
    else:
        cells = [c.strip() for c in re.split(r" {2,}", line.strip())]
    return [c for c in cells if c != ""] or [line.strip()]


def _is_table(block_lines: list[str]) -> bool:
    if len(block_lines) < 2:
        return False
    hits = sum(1 for ln in block_lines if _TABLE_ROW.match(ln) and "|" in ln or "\t" in ln)
    if hits >= max(2, len(block_lines) - 1):
        return True
    sep = sum(1 for ln in block_lines if re.search(r" {2,}\S", ln))
    return sep >= max(2, len(block_lines) - 1)


_BULLET = re.compile(r"^\s*(?:[-*•]|\(?\d{1,3}[.)])\s+\S")


def _is_list(block_lines: list[str]) -> bool:
    if len(block_lines) < 2:
        return False
    hits = sum(1 for ln in block_lines if _BULLET.match(ln))
    return hits >= max(2, len(block_lines) - 1)


def _is_title(line: str) -> bool:
    s = line.strip()
    # a heading has no digits, colon, or math operators (those signal data / an equation)
    if not (0 < len(s) <= 64) or any(ch.isdigit() for ch in s) or ":" in s or "=" in s:
        return False
    return not s.endswith((".", ",", ";")) and not re.search(r"[.!?]\s", s)


class TextParser:
    async def parse(
        self, data: bytes, *, doc_id: str, tenant_id: str, filename: str = ""
    ) -> ParsedDoc:
        if data[:5] == b"%PDF-":
            raise ImportError(
                "TextParser cannot read PDFs; install the 'pdf' extra to use PymupdfParser."
            )
        text = _clean(data.decode("utf-8", errors="replace"))
        raw_pages = text.split("\f") or [text]
        pages: list[Page] = []
        breadcrumb: list[str] = []
        order = 0
        summary = filename

        for pno, raw in enumerate(raw_pages, start=1):
            blocks: list[Block] = []
            for para in re.split(r"\n\s*\n", raw):
                lines = [ln for ln in para.splitlines() if ln.strip()]
                if not lines:
                    continue
                if _is_table(lines):
                    grid = [_split_row(ln) for ln in lines]
                    width = max(len(r) for r in grid)
                    grid = [r + [""] * (width - len(r)) for r in grid]
                    title = breadcrumb[-1] if breadcrumb else ""
                    tbl = Table(
                        block_id=f"{doc_id}:p{pno}:tbl{order}",
                        page=pno,
                        title=title,
                        n_header_rows=1,
                        grid=grid,
                    )
                    tbl.validate_checksum()
                    blocks.append(
                        Block(
                            block_id=tbl.block_id,
                            page=pno,
                            type=BlockType.TABLE,
                            text=tbl.linearize(),
                            reading_order=order,
                            table=tbl,
                            breadcrumb=list(breadcrumb),
                        )
                    )
                    order += 1
                elif _is_list(lines):
                    blocks.append(
                        Block(
                            block_id=f"{doc_id}:p{pno}:b{order}",
                            page=pno,
                            type=BlockType.LIST,
                            text="\n".join(ln.strip() for ln in lines),  # preserve line structure
                            reading_order=order,
                            breadcrumb=list(breadcrumb),
                        )
                    )
                    order += 1
                elif len(lines) == 1 and _is_title(lines[0]):
                    breadcrumb = [lines[0].strip()]
                    if not summary:
                        summary = lines[0].strip()
                    blocks.append(
                        Block(
                            block_id=f"{doc_id}:p{pno}:b{order}",
                            page=pno,
                            type=BlockType.TITLE,
                            text=lines[0].strip(),
                            reading_order=order,
                            breadcrumb=list(breadcrumb),
                        )
                    )
                    order += 1
                else:
                    body = " ".join(ln.strip() for ln in lines)
                    btype = BlockType.EQUATION if looks_like_equation(body) else BlockType.PARAGRAPH
                    blocks.append(
                        Block(
                            block_id=f"{doc_id}:p{pno}:b{order}",
                            page=pno,
                            type=btype,
                            text=body,
                            reading_order=order,
                            breadcrumb=list(breadcrumb),
                        )
                    )
                    order += 1
            pages.append(Page(page_no=pno, text_ratio=1.0, tier=ParseTier.DIGITAL, blocks=blocks))

        return ParsedDoc(
            doc_id=doc_id,
            tenant_id=tenant_id,
            content_hash="",
            filename=filename,
            page_count=len(pages),
            pages=pages,
            doc_summary=summary or filename,
        )
