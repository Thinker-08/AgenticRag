"""Chat-template token neutralization (08 threat 1, defense layer 1).

A PDF can embed the literal control tokens of the serving chat template (`<start_of_turn>` for
Gemma, `<|im_start|>` for ChatML, `[INST]`/`<<SYS>>` for Llama) to forge a role boundary during
templating. We neutralize them by inserting a zero-width space so the tokenizer can never emit
the special token, while the text stays visually identical.

This runs at INGEST normalization time — after NFKC (so full-width homoglyphs can't reassemble a
token post-filter) and before chunking — so the stored chunk text, the prompt evidence, and the
quote the citation verifier checks are all the same string. Neutralizing at prompt time instead
would break the exact-substring quote check (06 §2).
"""

from __future__ import annotations

import re

_ZW = "​"

_PATTERNS = [
    re.compile(r"<(\|im_(?:start|end)\|)>", re.IGNORECASE),          # ChatML
    re.compile(r"<(\|(?:system|user|assistant|end)\|)>", re.IGNORECASE),
    re.compile(r"<((?:start|end)_of_turn)>", re.IGNORECASE),          # Gemma
    re.compile(r"<(\|end(?:_of_text|oftext)\|)>", re.IGNORECASE),     # Llama3 / GPT-style
    re.compile(r"\[(/?INST)\]"),                                      # Llama2
    re.compile(r"<<(/?SYS)>>"),
]


def neutralize_template_tokens(text: str) -> str:
    if "<" not in text and "[" not in text:
        return text
    for pat in _PATTERNS:
        text = pat.sub(lambda m: f"{m.group(0)[0]}{_ZW}{m.group(0)[1:]}", text)
    return text
