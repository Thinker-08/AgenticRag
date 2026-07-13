from __future__ import annotations

locked_floors: dict[str, float] = {
    "faithfulness": 0.74,
    "token_f1": 0.41,
    "recall_at_k": 0.61,
    "context_precision": 0.38,
    "citation_accuracy": 0.0,
    "correct_refusal": 0.12,
}

DEFAULT_TOL: dict[str, float] = {
    "faithfulness": 0.01,
    "token_f1": 0.01,
    "recall_at_k": 0.01,
    "context_precision": 0.01,
    "citation_accuracy": 0.0,
    "correct_refusal": 0.0,
}


def promote(
    phase_name: str,
    candidate: dict,
    control: dict,
    tol: dict | None = None,
) -> dict:
    tols = {**DEFAULT_TOL, **(tol or {})}
    for metric in locked_floors:
        cand = candidate.get(metric)
        if cand is None:
            continue
        floor = control.get(metric, locked_floors[metric])
        if cand < floor - tols.get(metric, 0.0):
            raise AssertionError(
                f"{phase_name} REGRESSED {metric}: {cand:.4f} < {floor:.4f} "
                f"- tol {tols.get(metric, 0.0):.4f}"
            )
    delta = {
        key: round(candidate[key] - control[key], 4)
        for key in candidate
        if key in control and isinstance(candidate[key], (int, float))
    }
    return {"status": "PROMOTE", "delta": delta}
