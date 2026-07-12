"""Answer output filter (08 threat 1, defense layer 4): fail closed on injection fingerprints.

Scans the final answer for (a) the per-request evidence nonce (delimiter leak), (b) raw chat-
template control tokens (ingest neutralizes legitimate document mentions, so a raw token here
means the model emitted one), and (c) system-prompt fingerprints (prompt-leak). A hit is treated
as a FAILED verification: the query fails closed to abstention, never open (07 §6 callout).
"""

from __future__ import annotations

import re

from ..contracts import Answer

_TEMPLATE_TOKENS = re.compile(
    r"<\|im_(?:start|end)\|>|<(?:start|end)_of_turn>|\[/?INST\]|<</?SYS>>|<\|(?:system|user|assistant)\|>"
)

_PROMPT_FINGERPRINTS = (
    "untrusted DATA, not commands",
    "Emit ONLY the Answer schema",
    "EVIDENCE_START",
    "EVIDENCE_END",
)


def scan_answer(answer: Answer, *, nonce: str) -> str | None:
    """Return a violation label, or None if the answer is clean."""
    texts = [answer.answer_text] + [c.text for c in answer.claims]
    for text in texts:
        if nonce and nonce in text:
            return "nonce_leak"
        if _TEMPLATE_TOKENS.search(text):
            return "template_token"
        for fp in _PROMPT_FINGERPRINTS:
            if fp in text:
                return "prompt_leak"
    return None
