from __future__ import annotations

from typing import List

try:
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:  # optional dependency
    CrossEncoder = None  # type: ignore

from app.core.config import settings


class Reranker:
    def __init__(self):
        self.model = None
        name = getattr(settings, "RERANK_MODEL", None) or getattr(settings, "RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        if bool(getattr(settings, "RERANK_ENABLED", False)) and CrossEncoder is not None:
            try:
                self.model = CrossEncoder(name)
            except Exception:
                self.model = None

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        if not self.model or not documents:
            return [0.0] * len(documents)
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)
        return [float(x) for x in scores]


