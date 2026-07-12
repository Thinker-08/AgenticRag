"""Groundedness verifier (06 §2-§3): the load-bearing faithfulness gate.

Three layers, cheapest first: structural (chunk_id in the tenant-scoped bundle) → lexical (quote is an
exact substring of the cited chunk) → semantic (NLI entailment). A specialized verifier adjudicates —
never the generator judging itself. Bias toward abstention: the gray zone resolves to UNSUPPORTED.
"""

from __future__ import annotations

import asyncio

from ..contracts import Budget, Claim, DraftClaim, Evidence, SupportLabel
from ..deps import Deps


def _quote_ok(quote: str, chunk_text: str, span: tuple[int, int]) -> bool:
    if not quote:
        return False
    if quote in chunk_text:
        return True
    s, e = span
    if 0 <= s < e <= len(chunk_text) and chunk_text[s:e].strip() == quote.strip():
        return True
    return False


class GroundednessVerifier:
    def __init__(self, deps: Deps) -> None:
        self.deps = deps
        self.tau_entail = deps.settings.verifier.tau_entail
        self.tau_contra = deps.settings.verifier.tau_contra

    async def verify_one(
        self, dc: DraftClaim, evidence: Evidence, *, budget: Budget, idx: int
    ) -> Claim:
        cites = [c for c in dc.citations]
        base = Claim(claim_id=f"c{idx}", text=dc.text, citations=cites)

        if not cites or not all(c.chunk_id in evidence.ids for c in cites):
            return base.model_copy(
                update={"support": SupportLabel.UNSUPPORTED, "verifier": "structural"}
            )

        for c in cites:
            chunk = evidence.by_id(c.chunk_id)
            if chunk is None or not _quote_ok(c.quote, chunk.text, c.char_span):
                return base.model_copy(
                    update={"support": SupportLabel.UNSUPPORTED, "verifier": "lexical"}
                )

        premise = "\n".join(evidence.by_id(c.chunk_id).text for c in cites)
        verdict = await self.deps.verifier.entail(premise, dc.text, budget=budget)
        score = verdict.score
        if verdict.label == SupportLabel.CONTRADICTED or score <= self.tau_contra:
            support = SupportLabel.CONTRADICTED
        elif score >= self.tau_entail:
            support = SupportLabel.SUPPORTED
        else:
            support = SupportLabel.UNSUPPORTED
        return base.model_copy(
            update={
                "support": support,
                "entail_score": round(score, 4),
                "verifier": verdict.verifier,
            }
        )

    async def verify(
        self, claims: list[DraftClaim], evidence: Evidence, *, budget: Budget
    ) -> list[Claim]:
        return list(
            await asyncio.gather(
                *(
                    self.verify_one(dc, evidence, budget=budget, idx=i)
                    for i, dc in enumerate(claims, start=1)
                )
            )
        )
