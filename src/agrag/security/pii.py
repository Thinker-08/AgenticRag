from __future__ import annotations

import re

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE = re.compile(r"(?<![\d-])(?:\+?1[-.\s])?(?:\(\d{3}\)\s?|\d{3}[-.\s])\d{3}[-.\s]\d{4}(?![\d-])")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def luhnOk(digits: str) -> bool:
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


def detectPii(text: str) -> list[str]:
    found: set[str] = set()
    if _EMAIL.search(text):
        found.add("email")
    if _SSN.search(text):
        found.add("ssn")
    if _PHONE.search(text):
        found.add("phone")
    for m in _CARD.finditer(text):
        digits = re.sub(r"[ -]", "", m.group())
        if luhnOk(digits):
            found.add("credit_card")
            break

    return sorted(found)


_MAX_ATTR = 200


def scrub(text: str) -> str:
    text = _EMAIL.sub("[email]", text)
    text = _SSN.sub("[ssn]", text)
    text = _PHONE.sub("[phone]", text)
    text = _CARD.sub(lambda m: "[card]" if luhnOk(re.sub(r"[ -]", "", m.group())) else m.group(), text)

    return text


def scrubAttrs(attrs: dict) -> dict:
    out: dict = {}
    for k, v in attrs.items():
        if isinstance(v, str):
            s = scrub(v)
            out[k] = s if len(s) <= _MAX_ATTR else s[:_MAX_ATTR] + "…"
        else:
            out[k] = v

    return out
