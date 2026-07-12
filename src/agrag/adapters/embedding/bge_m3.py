"""BGE-M3 embedding adapter (full mode): dense + learned-sparse vectors."""

from __future__ import annotations

from typing import Sequence

from ...interfaces.types import EmbeddingResult


class BgeM3Embedding:
    """Encode text into 1024-d dense vectors plus BGE-M3 lexical (sparse) weights."""

    def __init__(self, model: str, version: str, device: str = "cpu") -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError(
                "BgeM3Embedding needs the 'ml' extra: pip install -e '.[ml]'"
            ) from exc
        self.model = model
        self.version = version
        self.dim = 1024
        self._encoder = BGEM3FlagModel(model, use_fp16=False, device=device)

    def _encode(self, texts: Sequence[str]) -> EmbeddingResult:
        out = self._encoder.encode(list(texts), return_dense=True, return_sparse=True)
        dense = out["dense_vecs"].tolist()
        sparse = [
            {int(term): float(weight) for term, weight in weights.items()}
            for weights in out["lexical_weights"]
        ]
        return EmbeddingResult(dense=dense, sparse=sparse, model=self.model, version=self.version)

    def encode_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> EmbeddingResult:
        return self._encode(texts)
