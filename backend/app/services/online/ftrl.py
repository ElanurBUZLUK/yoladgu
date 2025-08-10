import numpy as np
from typing import Dict, Optional
import json, redis

class FTRLService:
    def __init__(self, dim: int = 32, alpha: float = 0.1, beta: float = 1.0, l1: float = 0.0, l2: float = 1.0, redis_url: Optional[str] = None):
        self.dim = dim
        self.alpha, self.beta, self.l1, self.l2 = alpha, beta, l1, l2
        self.z = np.zeros(dim, dtype=np.float32)
        self.n = np.zeros(dim, dtype=np.float32)
        self._r = redis.Redis.from_url(redis_url, decode_responses=True) if redis_url else None
        self._load()

    def _feat(self, uf: Dict[str, float], qf: Dict[str, float]) -> np.ndarray:
        keys = sorted(list(uf.keys()) + list(qf.keys()))[:self.dim]
        x = np.zeros(self.dim, dtype=np.float32)
        for i,k in enumerate(keys):
            x[i] = float(uf.get(k, 0.0) if k in uf else qf.get(k, 0.0))
        return x

    def _weights(self) -> np.ndarray:
        w = np.zeros(self.dim, dtype=np.float32)
        for i in range(self.dim):
            if abs(self.z[i]) <= self.l1:
                w[i] = 0.0
            else:
                sign = -1.0 if self.z[i] < 0 else 1.0
                w[i] = (sign * self.l1 - self.z[i]) / ((self.beta + np.sqrt(self.n[i])) / self.alpha + self.l2)
        return w

    def _save(self) -> None:
        if not self._r:
            return
        try:
            self._r.set(f"ml:ftrl:{self.dim}:z", json.dumps(self.z.tolist()))
            self._r.set(f"ml:ftrl:{self.dim}:n", json.dumps(self.n.tolist()))
        except Exception:
            pass

    def _load(self) -> None:
        if not self._r:
            return
        try:
            rawz = self._r.get(f"ml:ftrl:{self.dim}:z")
            rawn = self._r.get(f"ml:ftrl:{self.dim}:n")
            if rawz and rawn:
                z = np.array(json.loads(rawz), dtype=np.float32)
                n = np.array(json.loads(rawn), dtype=np.float32)
                if z.shape == (self.dim,) and n.shape == (self.dim,):
                    self.z = z
                    self.n = n
        except Exception:
            pass

    def predict(self, student_id: int, uf: Dict[str, float], qf: Dict[str, float]) -> float:
        x = self._feat(uf, qf)
        w = self._weights()
        s = float(np.dot(w, x))
        # sigmoid
        return 1.0 / (1.0 + np.exp(-s))

    def update(self, label: int, uf: Dict[str, float], qf: Dict[str, float]):
        x = self._feat(uf, qf)
        p = 1.0 / (1.0 + np.exp(-np.dot(self._weights(), x)))
        g = (p - float(label)) * x
        for i in range(self.dim):
            sigma = (np.sqrt(self.n[i] + g[i]*g[i]) - np.sqrt(self.n[i])) / self.alpha
            self.z[i] += g[i] - sigma * self._weights()[i]
            self.n[i] += g[i]*g[i]
        self._save()
