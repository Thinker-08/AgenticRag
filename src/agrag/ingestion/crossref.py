from __future__ import annotations

import re

from ..contracts import Block, BlockType, ParsedDoc

_FOOTNOTE_DEF = re.compile(r"^\[(\d{1,2})\][.):]?\s+(\S.+)", re.DOTALL)
_FOOTNOTE_REF = re.compile(r"\[(\d{1,2})\]")
_FIG_TABLE_REF = re.compile(
    r"\b(?:see\s+)?(Figure|Fig\.?|Table|Exhibit)\s+(\d{1,3})\b", re.IGNORECASE
)
_SECTION_REF = re.compile(r"(?:§|\bsection\s+)(\d+(?:\.\d+)*)", re.IGNORECASE)
_CAPTION = re.compile(r"^(Figure|Fig\.?|Table|Exhibit)\s+(\d{1,3})\b", re.IGNORECASE)
_HEADING_NUM = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+\S")

_MAX_INLINE_CHARS = 300


def _norm_kind(word: str) -> str:
    w = word.lower().rstrip(".")
    return "figure" if w in ("figure", "fig") else w


def link_crossrefs(doc: ParsedDoc) -> None:
    blocks: list[Block] = [b for p in doc.pages for b in p.blocks]
    if not blocks:
        return

    footnote_defs: dict[str, Block] = {}
    targets: dict[str, str] = {}
    for b in blocks:
        text = b.text.strip()
        m = _FOOTNOTE_DEF.match(text)
        if m and len(text) <= 500 and b.type in (BlockType.PARAGRAPH, BlockType.FOOTER):
            footnote_defs.setdefault(m.group(1), b)
        m = _CAPTION.match(text)
        if m:
            targets.setdefault(f"{_norm_kind(m.group(1))} {m.group(2)}", b.block_id)
        m = _HEADING_NUM.match(text)
        if m and b.type == BlockType.TITLE:
            targets.setdefault(f"section {m.group(1)}", b.block_id)

    inlined: set[str] = set()
    for b in blocks:
        if b.type in (BlockType.FOOTER, BlockType.HEADER):
            continue
        links: list[str] = []

        for m in _FOOTNOTE_REF.finditer(b.text):
            num = m.group(1)
            fdef = footnote_defs.get(num)
            if fdef is None or fdef.block_id == b.block_id:
                continue
            links.append(fdef.block_id)
            if num not in inlined:
                note = _FOOTNOTE_DEF.match(fdef.text.strip()).group(2)[:_MAX_INLINE_CHARS]
                b.text = f"{b.text}\n[Footnote {num}] {note}"
                inlined.add(num)

        for m in _FIG_TABLE_REF.finditer(b.text):
            key = f"{_norm_kind(m.group(1))} {m.group(2)}"
            tid = targets.get(key)
            if tid and tid != b.block_id:
                links.append(tid)
        for m in _SECTION_REF.finditer(b.text):
            tid = targets.get(f"section {m.group(1)}")
            if tid and tid != b.block_id:
                links.append(tid)

        if links:
            b.links = sorted(set(links))

    for num in inlined:
        footnote_defs[num].type = BlockType.FOOTER
