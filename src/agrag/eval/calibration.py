from __future__ import annotations

from dataclasses import dataclass


def cohensKappa(judge: list[bool], human: list[bool]) -> float:
    if len(judge) != len(human) or not judge:
        raise ValueError("judge and human label lists must be equal-length and non-empty")

    n = len(judge)
    agree = sum(1 for j, h in zip(judge, human) if j == h) / n
    pj = sum(judge) / n
    ph = sum(human) / n
    chance = pj * ph + (1 - pj) * (1 - ph)

    if chance >= 1.0:
        return 1.0
    return (agree - chance) / (1 - chance)


@dataclass
class TauPoint:
    tau: float
    correct_refusal: float
    over_abstention: float
    f_beta: float


def fbeta(tp: int, fp: int, fn: int, beta: float) -> float:
    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom else 0.0


def sweepTau(scored: list[tuple[float, bool]], *, grid: list[float] | None = None, beta: float = 0.5) -> tuple[float, list[TauPoint]]:
    grid = grid or [i / 20 for i in range(21)]
    curve: list[TauPoint] = []
    n_ans = sum(1 for _, g in scored if g)
    n_unans = len(scored) - n_ans

    for tau in grid:
        tp = sum(1 for s, g in scored if not g and s < tau)
        fp = sum(1 for s, g in scored if g and s < tau)
        fn = sum(1 for s, g in scored if not g and s >= tau)
        curve.append(TauPoint(tau=round(tau, 4), correct_refusal=(tp / n_unans) if n_unans else 0.0, over_abstention=(fp / n_ans) if n_ans else 0.0, f_beta=round(fbeta(tp, fp, fn, beta), 4)))

    best = max(curve, key=lambda p: (p.f_beta, p.tau))
    return best.tau, curve
