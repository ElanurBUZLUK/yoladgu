# app/services/index_backends/faiss_flat_index.py
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import faiss
except ImportError as e:
    raise ImportError("faiss-cpu gerekli: pip install faiss-cpu") from e

from .base import BaseIndexBackend

class FAISSFlatIndexBackend(BaseIndexBackend):
    """
    FAISS IndexFlat (L2 veya IP) — kesin (exact) ANN, küçük/orta veri için yüksek recall.
    Not: Silme/reindex için yeniden inşa gerekir.
    """
    def __init__(self, vector_size: int, metric: str = "ip"):
        super().__init__(vector_size)
        self.metric = metric  # "ip" (cosine için normalize edilmiş vektörlerle) veya "l2"
        self.index = None
        self.id_to_metadata: Dict[str, Dict[str, Any]] = {}
        self.string_to_int: Dict[str, int] = {}
        self.int_to_string: Dict[int, str] = {}
        self.next_id = 0
        self.stats = {"total_items": 0, "search_count": 0, "avg_search_time_ms": 0.0}

    async def initialize(self) -> bool:
        if self.metric == "ip":
            self.index = faiss.IndexFlatIP(self.vector_size)
        else:
            self.index = faiss.IndexFlatL2(self.vector_size)
        self._built = True
        return True

    def _add_ids(self, ids: List[str]) -> np.ndarray:
        int_ids = []
        for sid in ids:
            if sid not in self.string_to_int:
                self.string_to_int[sid] = self.next_id
                self.int_to_string[self.next_id] = sid
                self.next_id += 1
            int_ids.append(self.string_to_int[sid])
        return np.array(int_ids, dtype=np.int64)

    async def add_items(self, vectors: np.ndarray, ids: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> bool:
        if not self._validate_vector(vectors):
            raise ValueError(f"Vector dim must be {self.vector_size}")
        # cosine istiyorsak IP + normalize
        if self.metric == "ip":
            vectors = self._normalize_vector(vectors).astype(np.float32)
        else:
            vectors = vectors.astype(np.float32)

        int_ids = self._add_ids(ids)
        # IndexFlat ID’yi ayrı tutmadığı için, ID map’ini kendimiz yönetiyoruz
        self.index.add(vectors)
        if metadata:
            for i, sid in enumerate(ids):
                self.id_to_metadata[sid] = metadata[i] if i < len(metadata) else {}
        self.stats["total_items"] += vectors.shape[0]
        return True

    async def search(self, query_vector: np.ndarray, k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self._validate_vector(query_vector):
            raise ValueError(f"Query dim must be {self.vector_size}")
        if self.metric == "ip":
            query_vector = self._normalize_vector(query_vector).astype(np.float32)
        else:
            query_vector = query_vector.astype(np.float32)

        import time
        t0 = time.time()
        D, I = self.index.search(query_vector, k)
        took = (time.time() - t0) * 1000
        self.stats["search_count"] += 1
        sc = self.stats["search_count"]
        self.stats["avg_search_time_ms"] = (self.stats["avg_search_time_ms"]*(sc-1) + took)/sc

        # IndexFlat, sıra ekleme mantığına göre 0..N-1 döndürür; string ID’ye çevir
        results = []
        arr = I[0]
        sims_or_dists = D[0]
        for pos, idx in enumerate(arr):
            if idx < 0:  # no result
                continue
            # insertion order = idx, string id bul
            sid = self.int_to_string.get(idx)
            if sid is None:  # güvenlik
                continue
            md = self.id_to_metadata.get(sid, {})
            if self.metric == "ip":
                score = float(sims_or_dists[pos])      # similarity
                distance = 1.0 - score
            else:
                distance = float(sims_or_dists[pos])   # L2
                score = 1.0/(1.0+distance)
            results.append({"item_id": sid, "score": score, "distance": distance, "metadata": md})

        # basit payload filtre (in-memory)
        if filters:
            def ok(m):
                for k,v in filters.items():
                    if k not in m: return False
                    if isinstance(v, list):
                        if not any(x in (m[k] or []) for x in v): return False
                    else:
                        if m[k] != v: return False
                return True
            results = [r for r in results if ok(r["metadata"])]

        return results[:k]

    async def delete_items(self, ids: List[str]) -> bool:
        # IndexFlat delete yok; metadata’dan düş, gerekirse yeniden inşa stratejisi uygula
        for sid in ids:
            self.id_to_metadata.pop(sid, None)
        return True

    async def get_stats(self) -> Dict[str, Any]:
        return {**self.stats, "index_type": "FAISS-Flat", "metric": self.metric}
