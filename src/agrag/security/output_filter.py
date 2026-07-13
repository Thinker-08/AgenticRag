from __future__ import annotations

import re

from ..contracts import Answer

_TEMPLATE_TOKENS = re.compile(r"<\|im_(?:start|end)\|>|<(?:start|end)_of_turn>|\[/?INST\]|<</?SYS>>|<\|(?:system|user|assistant)\|>")

_PROMPT_FINGERPRINTS = ("untrusted DATA, not commands", "Emit ONLY the Answer schema", "EVIDENCE_START", "EVIDENCE_END")


def scanAnswer(answer: Answer, *, nonce: str) -> str | None:
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
