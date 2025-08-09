from typing import List, Tuple, Optional
import numpy as np, faiss, os
from .base import IndexBackend

class FaissIVFPQBackend(IndexBackend):
    def __init__(self, dim: int, nlist: int, m: int, nbits: int, nprobe: int):
        self.dim, self.nlist, self.m, self.nbits = dim, nlist, m, nbits
        self.nprobe = nprobe
        self.idx: Optional[faiss.IndexIVFPQ] = None

    def build(self, embeddings: np.ndarray, ids: Optional[np.ndarray] = None) -> None:
        quant = faiss.IndexFlatL2(self.dim)
        idx = faiss.IndexIVFPQ(quant, self.dim, self.nlist, self.m, self.nbits)
        idx.nprobe = self.nprobe
        idx.train(embeddings.astype("float32"))
        if ids is not None:
            idx.add_with_ids(embeddings.astype("float32"), ids.astype("int64"))
        else:
            idx.add(embeddings.astype("float32"))
        self.idx = idx

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.idx is None: raise RuntimeError("index not built")
        faiss.write_index(self.idx, path)

    def load(self, path: str) -> None:
        self.idx = faiss.read_index(path)
        self.idx.nprobe = self.nprobe

    def search(self, query: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        if self.idx is None: return [], []
        q = query.reshape(1, -1).astype("float32")
        D, I = self.idx.search(q, k)
        return I[0].tolist(), D[0].tolist()

    def set_param(self, **kwargs) -> None:
        if "nprobe" in kwargs and self.idx is not None:
            self.nprobe = int(kwargs["nprobe"])
            self.idx.nprobe = self.nprobe

    def stats(self) -> dict:
        return {
            "type": "faiss_ivfpq",
            "dim": self.dim,
            "nlist": self.nlist,
            "m": self.m,
            "nbits": self.nbits,
            "nprobe": self.nprobe,
            "ntotal": int(self.idx.ntotal) if self.idx else 0,
            "is_trained": bool(self.idx and self.idx.is_trained),
        }
