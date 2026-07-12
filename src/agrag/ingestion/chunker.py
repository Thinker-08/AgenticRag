"""Chunkers (03 §4). Hierarchical parent-child is the default; recursive fixed-size is the baseline control.

Retrieve a small precise child, feed the larger parent to the generator (small-to-big). Tables/lists/
equations stay atomic so cells aren't split. Breadcrumbs are prepended so context survives isolation.
"""

from __future__ import annotations

from ..config import ChunkerConfig
from ..contracts import Block, BlockType, Chunk, ChunkKind, ParsedDoc
from .hashing import content_hash

_ATOMIC = {BlockType.TABLE, BlockType.LIST, BlockType.EQUATION, BlockType.FIGURE}
_KIND = {
    BlockType.TABLE: ChunkKind.TABLE,
    BlockType.LIST: ChunkKind.LIST,
    BlockType.EQUATION: ChunkKind.EQUATION,
    BlockType.FIGURE: ChunkKind.FIGURE,
}


def _mk_chunk(
    doc: ParsedDoc,
    cid: str,
    text: str,
    *,
    page: int,
    kind: ChunkKind,
    block: Block | None,
    parent_id: str | None,
    is_parent: bool,
    breadcrumb: list[str],
) -> Chunk:
    crumb = (" › ".join(breadcrumb) + "\n") if breadcrumb else ""
    linearized = crumb + text
    return Chunk(
        chunk_id=cid,
        doc_id=doc.doc_id,
        tenant_id=doc.tenant_id,
        page_no=page,
        bbox=block.bbox if block else (0, 0, 0, 0),
        section_breadcrumb=list(breadcrumb),
        kind=kind,
        atomic=kind in {ChunkKind.TABLE, ChunkKind.LIST, ChunkKind.EQUATION, ChunkKind.FIGURE},
        text=text,
        linearized_text=linearized,
        parent_id=parent_id,
        linked_block=block.block_id if block and block.table else None,
        content_hash=content_hash(text),
        lang=block.lang if block else "en",
        extra_metadata={**doc.extra_metadata, "is_parent": is_parent},
    )


def _split_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text] if text.strip() else []
    out, i = [], 0
    step = max(1, size - overlap)
    while i < len(words):
        out.append(" ".join(words[i : i + size]))
        i += step
    return out


class HierarchicalChunker:
    name = "hierarchical"

    def __init__(self, child_size: int = 320, parent_size: int = 1500, overlap: int = 64) -> None:
        self.child = child_size
        self.parent = parent_size
        self.overlap = overlap

    def split(self, doc: ParsedDoc) -> list[Chunk]:
        chunks: list[Chunk] = []
        n = 0
        section: list[Block] = []
        crumb: list[str] = []

        def flush() -> None:
            nonlocal n, section, crumb
            if not section:
                return
            page = section[0].page
            body = "\n".join(b.text for b in section)
            parent_id = f"{doc.doc_id}:p{page}:parent{n}"
            chunks.append(
                _mk_chunk(
                    doc,
                    parent_id,
                    body[: self.parent * 6],
                    page=page,
                    kind=ChunkKind.PROSE,
                    block=None,
                    parent_id=None,
                    is_parent=True,
                    breadcrumb=list(crumb),
                )
            )
            n += 1
            for piece in _split_words(body, self.child, self.overlap):
                cid = f"{doc.doc_id}:p{page}:c{n}"
                chunks.append(
                    _mk_chunk(
                        doc,
                        cid,
                        piece,
                        page=page,
                        kind=ChunkKind.PROSE,
                        block=None,
                        parent_id=parent_id,
                        is_parent=False,
                        breadcrumb=list(crumb),
                    )
                )
                n += 1
            section = []

        for page in doc.pages:
            for block in sorted(page.blocks, key=lambda b: b.reading_order):
                if block.type == BlockType.TITLE:
                    flush()
                    crumb = list(block.breadcrumb) or [block.text]
                    continue
                if block.type in (BlockType.HEADER, BlockType.FOOTER, BlockType.CAPTION):
                    continue
                if block.type in _ATOMIC:
                    flush()
                    cid = f"{doc.doc_id}:p{block.page}:c{n}"
                    chunks.append(
                        _mk_chunk(
                            doc,
                            cid,
                            block.text,
                            page=block.page,
                            kind=_KIND[block.type],
                            block=block,
                            parent_id=None,
                            is_parent=False,
                            breadcrumb=block.breadcrumb or crumb,
                        )
                    )
                    n += 1
                else:
                    section.append(block)
            flush()
        return chunks


class RecursiveChunker:
    """Fixed-size ~512 with ~15% overlap, structure-blind — the vanilla baseline control (page 13, step 1)."""

    name = "recursive"

    def __init__(self, size: int = 512, overlap: int = 77) -> None:
        self.size = size
        self.overlap = overlap

    def split(self, doc: ParsedDoc) -> list[Chunk]:
        chunks: list[Chunk] = []
        n = 0
        for page in doc.pages:
            text = "\n".join(b.text for b in sorted(page.blocks, key=lambda b: b.reading_order))
            for piece in _split_words(text, self.size, self.overlap):
                cid = f"{doc.doc_id}:p{page.page_no}:c{n}"
                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        doc_id=doc.doc_id,
                        tenant_id=doc.tenant_id,
                        page_no=page.page_no,
                        kind=ChunkKind.PROSE,
                        text=piece,
                        linearized_text=piece,
                        content_hash=content_hash(piece),
                        extra_metadata={**doc.extra_metadata, "is_parent": False},
                    )
                )
                n += 1
        return chunks


def build_chunker(cfg: ChunkerConfig):
    if cfg.provider == "recursive":
        return RecursiveChunker(size=512, overlap=77)
    return HierarchicalChunker(
        child_size=cfg.child_size, parent_size=cfg.parent_size, overlap=cfg.overlap
    )
