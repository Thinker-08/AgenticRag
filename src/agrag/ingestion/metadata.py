from __future__ import annotations

import re
from collections import Counter

from ..contracts import Chunk

_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_QUARTER = re.compile(r"\bQ([1-4])\b", re.IGNORECASE)
_CURRENCY = [(re.compile(r"\bUSD\b|\bUS\$|\$(?=\s*\d)|\bU\.S\. dollars?\b", re.I), "USD"), (re.compile(r"\bEUR\b|€"), "EUR"), (re.compile(r"\bGBP\b|£"), "GBP"), (re.compile(r"\bJPY\b|¥"), "JPY")]
_DOC_TYPES = [(re.compile(r"\bform\s+10-?K\b|\bannual report\b", re.I), "10-K"), (re.compile(r"\bform\s+10-?Q\b|\bquarterly report\b", re.I), "10-Q"), (re.compile(r"\bdatasheet\b|\bpart\s+number\b", re.I), "datasheet"), (re.compile(r"\bprospectus\b"), "prospectus")]


def currency(text: str) -> str | None:
    for pat, code in _CURRENCY:
        if pat.search(text):
            return code
    return None


def docMetadata(text: str) -> dict:
    meta: dict = {}
    for pat, dt in _DOC_TYPES:
        if pat.search(text):
            meta["doc_type"] = dt
            break

    cur = currency(text)
    if cur:
        meta["currency"] = cur

    years = _YEAR.findall(text)
    if years:
        meta["fiscal_year_dominant"] = int(Counter(years).most_common(1)[0][0])

    return meta


def enrichMetadata(chunks: list[Chunk], doc_text: str) -> list[Chunk]:
    dmeta = docMetadata(doc_text)
    out: list[Chunk] = []

    for c in chunks:
        extra = dict(c.extra_metadata)
        years = set(_YEAR.findall(c.text))
        if len(years) == 1:
            extra["fiscal_year"] = int(next(iter(years)))

        q = _QUARTER.search(c.text)
        if q:
            extra["fiscal_quarter"] = f"Q{q.group(1)}"

        cur = currency(c.text)
        if cur:
            extra["currency"] = cur

        for k, v in dmeta.items():
            extra.setdefault(k, v)
        out.append(c.model_copy(update={"extra_metadata": extra}) if extra != c.extra_metadata else c)
    return out
