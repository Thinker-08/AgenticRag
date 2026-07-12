"""content_hash — one deterministic key doing three jobs: idempotency, invalidation, cache keys (C10/C17/C20)."""

from __future__ import annotations

import hashlib
import re
import unicodedata


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def normalize_for_hash(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def content_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(normalize_for_hash(text).encode()).hexdigest()


def merkle_diff(old_page_hashes: dict[int, str], new_page_hashes: dict[int, str]) -> set[int]:
    """Return the set of page numbers whose content changed — re-embed only these subtrees (C20)."""
    changed: set[int] = set()
    for page, h in new_page_hashes.items():
        if old_page_hashes.get(page) != h:
            changed.add(page)
    changed.update(set(old_page_hashes) - set(new_page_hashes))
    return changed
