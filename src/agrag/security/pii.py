"""Lightweight PII detection (08 threat 4): tag at ingest, scrub from telemetry.

Regex + checksum detectors for the high-signal classes. Policy is tag-and-restrict (redacting
inline destroys answerability); the tags ride chunk metadata so a deployment can filter or audit.
Phone/card patterns require separators or pass Luhn so financial figures and years never match.
"""

from __future__ import annotations

import re

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE = re.compile(r"(?<![\d-])(?:\+?1[-.\s])?(?:\(\d{3}\)\s?|\d{3}[-.\s])\d{3}[-.\s]\d{4}(?![\d-])")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _luhn_ok(digits: str) -> bool:
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def detect_pii(text: str) -> list[str]:
    found: set[str] = set()
    if _EMAIL.search(text):
        found.add("email")
    if _SSN.search(text):
        found.add("ssn")
    if _PHONE.search(text):
        found.add("phone")
    for m in _CARD.finditer(text):
        digits = re.sub(r"[ -]", "", m.group())
        if _luhn_ok(digits):
            found.add("credit_card")
            break
    return sorted(found)
