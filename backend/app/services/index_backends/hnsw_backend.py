from typing import List, Tuple, Optional
import numpy as np, hnswlib, os
from app.core.config import settings
from .base import IndexBackend

class HNSWBackend(IndexBackend):
    def __init__(self, dim: int, space: str = "l2", M: int = 64, efC: int = 200, efS: int = 64):
        self.dim, self.space, self.M, self.efC, self.efS = dim, space, M, efC, efS
        self.index: Optional[hnswlib.Index] = None

    def build(self, embeddings: np.ndarray, ids: Optional[np.ndarray] = None) -> None:
        num = embeddings.shape[0]
        idx = hnswlib.Index(space=self.space, dim=self.dim)
        idx.init_index(max_elements=num, ef_construction=self.efC, M=self.M)
        if ids is None:
            ids = np.arange(num, dtype=np.int64)
        idx.add_items(embeddings.astype("float32"), ids.astype(np.int64))
        idx.set_ef(self.efS)
        self.index = idx

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.index is None: raise RuntimeError("index not built")
        self.index.save_index(path)

    def load(self, path: str) -> None:
        idx = hnswlib.Index(space=self.space, dim=self.dim)
        idx.load_index(path)
        idx.set_ef(self.efS)
        self.index = idx

    def search(self, query: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        if self.index is None: return [], []
        labels, dists = self.index.knn_query(query.reshape(1, -1).astype("float32"), k=k)
        return labels[0].tolist(), dists[0].tolist()

    def set_param(self, **kwargs) -> None:
        if "ef_search" in kwargs and self.index is not None:
            self.efS = int(kwargs["ef_search"])
            self.index.set_ef(self.efS)

    def stats(self) -> dict:
        return {"type":"hnsw","dim":self.dim,"M":self.M,"ef_construct":self.efC,"ef_search":self.efS}
