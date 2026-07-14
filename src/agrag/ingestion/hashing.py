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
