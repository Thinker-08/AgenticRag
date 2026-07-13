"""ANN recall measurement (C1, 04 §1): recall is empirical, never a spec-sheet number.

Ground truth = brute-force exact cosine kNN over the indexed vectors — so it needs NO human labels.
recall@k = |ANN_topk ∩ exact_topk| / k, averaged over sampled probe queries. Re-run after any
rebuild, quantization change, or embedding swap: a silent 0.98→0.82 drop ships confident wrong
answers with zero errors in the logs.
"""

from __future__ import annotations

import numpy as np

from ..deps import Deps


async def measure_ann_recall(
    deps: Deps,
    *,
    tenant_id: str = "default",
    k: int = 10,
    sample: int = 50,
) -> dict:
    chunks = await deps.docstore.list_chunks(tenant_id)
    if len(chunks) <= k:
        return {
            "recall_at_k": 1.0,
            "n_probes": 0,
            "corpus": len(chunks),
            "note": "corpus <= k; ANN is exact",
        }

    texts = [c.embed_input() for c in chunks]
    ids = [c.chunk_id for c in chunks]
    mat = np.asarray(deps.embedding.encode_documents(texts).dense, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    unit = mat / np.clip(norms, 1e-8, None)

    id_to_idx = {cid: i for i, cid in enumerate(ids)}
    step = max(1, len(chunks) // sample)
    probes = list(range(0, len(chunks), step))[:sample]
    qres = deps.embedding.encode_queries([texts[pi] for pi in probes])
    hits = 0
    for j, pi in enumerate(probes):
        exact = set(np.argsort(-(unit @ unit[pi]))[:k].tolist())
        ann = await deps.vectorstore.search(qres.dense[j], tenant_id=tenant_id, top_k=k)
        ann_idx = {id_to_idx[sc.chunk.chunk_id] for sc in ann if sc.chunk.chunk_id in id_to_idx}
        hits += len(exact & ann_idx)
    recall = hits / (len(probes) * k)
    return {"recall_at_k": round(recall, 4), "k": k, "n_probes": len(probes), "corpus": len(chunks)}
