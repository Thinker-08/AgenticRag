from __future__ import annotations

import hashlib
import re
import unicodedata


def sha256Bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def normalizeForHash(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def contentHash(text: str) -> str:
    return "sha256:" + hashlib.sha256(normalizeForHash(text).encode()).hexdigest()


def merkleDiff(old_page_hashes: dict[int, str], new_page_hashes: dict[int, str]) -> set[int]:
    changed: set[int] = set()
    for page, h in new_page_hashes.items():
        if old_page_hashes.get(page) != h:
            changed.add(page)

    changed.update(set(old_page_hashes) - set(new_page_hashes))
    return changed
