from __future__ import annotations

from typing import List, Dict, Any
import re
import json
import numpy as np

from app.core.config import settings
from app.services.vector_index_manager import VectorIndexManager
from app.services.embedding_api import get_embedding_provider
from app.services.content.questions_service import QuestionsService
from app.services.reranking import Reranker

try:
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:  # optional dependency
    CrossEncoder = None  # type: ignore


class SemanticCache:
    def __init__(self, redis_url: str):
        import redis  # lazy import to keep optional
        self.r = redis.Redis.from_url(redis_url, decode_responses=True)
        self.ttl = int(getattr(settings, "SEMANTIC_CACHE_TTL_S", 900))

    def _key(self, query: str) -> str:
        return f"rag:cache:v1:{query.strip().lower()}"

    def get(self, query: str):
        try:
            raw = self.r.get(self._key(query))
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def set(self, query: str, value: Any) -> None:
        try:
            self.r.setex(self._key(query), self.ttl, json.dumps(value, ensure_ascii=False))
        except Exception:
            pass


class AdvancedRAGService:
    def __init__(self, enable_hybrid_search: bool = True, enable_reranking: bool = False):
        self.idx = VectorIndexManager(settings.REDIS_URL)
        self.embed = get_embedding_provider()
        self.qsvc = QuestionsService(settings.REDIS_URL)
        self.cache = SemanticCache(redis_url=settings.REDIS_URL)
        self.enable_hybrid_search = bool(enable_hybrid_search)
        self.enable_reranking = bool(enable_reranking and getattr(settings, "RERANK_ENABLED", False))
        self._reranker = Reranker() if self.enable_reranking else None

    # --- Public API ---
    def search(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        return self.hybrid_search(text, k=k) if self.enable_hybrid_search else self.vector_only_search(text, k=k)

    def hybrid_search(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        cached = self.cache.get(text)
        if cached:
            return cached

        vec = np.array([self.embed.embed_one(text)], dtype=np.float32)
        vector_ids, _ = self.idx.search(vec[0], k=max(5, k * 3))

        keyword_backend = getattr(settings, "HYBRID_KEYWORD_BACKEND", "none").lower()
        keyword_ids: List[int] = []
        if keyword_backend == "postgres":
            keyword_ids = self.keyword_search(text, k=max(5, k * 2))
        elif keyword_backend == "elastic":
            # Placeholder: integrate with Elasticsearch via official client
            keyword_ids = []

        all_ids = list({int(i) for i in (list(vector_ids) + list(keyword_ids))})
        results: List[Dict[str, Any]] = []
        for qid in all_ids:
            meta = self.qsvc.get(int(qid)) or {}
            if meta:
                results.append({"id": int(qid), "meta": meta, "text": meta.get("text", ""), "score": 0.0})

        if self.enable_reranking and self._reranker is not None and results:
            results = self._rerank(text, results, top_k=k)

        out = results[:k]
        self.cache.set(text, out)
        return out

    def vector_only_search(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        vec = np.array([self.embed.embed_one(text)], dtype=np.float32)
        ids, dists = self.idx.search(vec[0], k=k)
        out: List[Dict[str, Any]] = []
        for i, d in zip(ids, dists):
            meta = self.qsvc.get(int(i)) or {}
            out.append({"id": int(i), "dist": float(d), "meta": meta, "text": meta.get("text", ""), "score": 0.0})
        return out

    # --- Internal helpers ---
    def keyword_search(self, query: str, k: int) -> List[int]:
        # Simplified FTS over cached question texts in Redis
        tokens = self._preprocess_text(query)
        # naive scan (acceptable for MVP); replace with Postgres FTS/Elastic later
        try:
            keys = self.qsvc.r.keys("question:*")  # type: ignore[attr-defined]
        except Exception:
            keys = []
        scored: list[tuple[int, float]] = []
        for key in keys:
            try:
                qid = int(str(key).split(":")[-1])
                meta = self.qsvc.get(qid) or {}
                text = (meta.get("text") or "") + " " + (meta.get("title") or "")
                doc_tokens = self._preprocess_text(text)
                score = self._bm25_like(tokens, doc_tokens)
                if score > 0:
                    scored.append((qid, score))
            except Exception:
                continue
        scored.sort(key=lambda x: x[1], reverse=True)
        return [qid for qid, _ in scored[:k]]

    def _rerank(self, query: str, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        try:
            texts = [r.get("text", "") for r in results]
            scores = self._reranker.rerank(query, texts)  # type: ignore[union-attr]
            for i, s in enumerate(scores):
                results[i]["score"] = float(s)
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        except Exception:
            return results[:top_k]

    # --- Text utils ---
    def _preprocess_text(self, text: str) -> List[str]:
        text = re.sub(r"[^\w\s]", " ", (text or "").lower()).strip()
        tokens = [t for t in text.split() if len(t) > 2]
        return tokens

    def _bm25_like(self, q: List[str], d: List[str]) -> float:
        if not q or not d:
            return 0.0
        k1, b = 1.5, 0.75
        avg_len = 100.0
        score = 0.0
        for term in q:
            tf = d.count(term)
            if tf == 0:
                continue
            # naive idf approximation using doc length only (MVP)
            idf = 1.5
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len(d) / avg_len))
        return float(score)


