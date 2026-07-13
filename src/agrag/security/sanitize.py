from __future__ import annotations

import re

_ZW = "​"

_PATTERNS = [
    re.compile(r"<(\|im_(?:start|end)\|)>", re.IGNORECASE),
    re.compile(r"<(\|(?:system|user|assistant|end)\|)>", re.IGNORECASE),
    re.compile(r"<((?:start|end)_of_turn)>", re.IGNORECASE),
    re.compile(r"<(\|end(?:_of_text|oftext)\|)>", re.IGNORECASE),
    re.compile(r"\[(/?INST)\]"),
    re.compile(r"<<(/?SYS)>>"),
]


def neutralize_template_tokens(text: str) -> str:
    if "<" not in text and "[" not in text:
        return text
    for pat in _PATTERNS:
        text = pat.sub(lambda m: f"{m.group(0)[0]}{_ZW}{m.group(0)[1:]}", text)
    return text


_MARK = "⁣"
_MARK_RE = re.compile(r"(\s+)")


def datamark(text: str) -> str:
    return _MARK_RE.sub(lambda m: m.group(1) + _MARK, text)


def strip_datamarks(text: str) -> str:
    return text.replace(_MARK, "")
