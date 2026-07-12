"""Equation-block detection (03 stage 2): flag math so it chunks atomically, never mid-formula.

Heuristic: LaTeX markers, or a high density of math symbols with few plain words. Deliberately
conservative — a false EQUATION merely makes a block atomic; a false PARAGRAPH splits a formula.
"""

from __future__ import annotations

import re

_LATEX = re.compile(r"\\(?:frac|sum|int|sqrt|alpha|beta|sigma|cdot|times|left|right)\b|\$\$")
_MATH_CHARS = set("=+−-*/^_{}()[]<>≈≤≥±×÷√∆πΣ∫∂∞")
_WORD = re.compile(r"[A-Za-z]{3,}")
# an assignment whose right side carries an arithmetic operator: "margin = (rev - cost) / rev"
_ASSIGN = re.compile(r"^[A-Za-z_][\w ]{0,30}=\s*[^=].*[-+*/^()].*$")


def looks_like_equation(text: str) -> bool:
    s = text.strip()
    if len(s) < 6 or len(s) > 600 or "\n" in s:
        return False
    if _LATEX.search(s):
        return True
    if "=" not in s:
        return False
    if _ASSIGN.match(s) and len(_WORD.findall(s)) <= 8:
        return True
    mathy = sum(1 for ch in s if ch in _MATH_CHARS or ch.isdigit())
    words = len(_WORD.findall(s))
    return mathy / len(s) > 0.3 and words <= 4
