import numpy as np
from typing import Dict, Optional
import json, redis

class LinUCBService:
    def __init__(self, feature_dim: int = 16, alpha: float = 0.2, redis_url: Optional[str] = None):
        self.d = feature_dim
        self.alpha = alpha
        self.A: Dict[int, np.ndarray] = {}
        self.b: Dict[int, np.ndarray] = {}
        self._r = redis.Redis.from_url(redis_url, decode_responses=True) if redis_url else None

    def _key_A(self, qid: int) -> str:
        return f"ml:linucb:{self.d}:{qid}:A"

    def _key_b(self, qid: int) -> str:
        return f"ml:linucb:{self.d}:{qid}:b"

    def _save(self, qid: int) -> None:
        if not self._r:
            return
        A = self.A.get(qid)
        b = self.b.get(qid)
        if A is None or b is None:
            return
        try:
            self._r.set(self._key_A(qid), json.dumps(A.tolist()))
            self._r.set(self._key_b(qid), json.dumps(b.reshape(-1).tolist()))
        except Exception:
            pass

    def _load(self, qid: int) -> bool:
        if not self._r:
            return False
        try:
            rawA = self._r.get(self._key_A(qid))
            rawb = self._r.get(self._key_b(qid))
            if not rawA or not rawb:
                return False
            A = np.array(json.loads(rawA), dtype=np.float32)
            b = np.array(json.loads(rawb), dtype=np.float32).reshape(-1, 1)
            if A.shape != (self.d, self.d) or b.shape != (self.d, 1):
                return False
            self.A[qid] = A
            self.b[qid] = b
            return True
        except Exception:
            return False

    def _ensure(self, qid: int):
        if qid in self.A:
            return
        if self._load(qid):
            return
        self.A[qid] = np.eye(self.d)
        self.b[qid] = np.zeros((self.d, 1))
        self._save(qid)

    def predict(self, user_features: Dict[str, float], question_features: Dict[str, float], qid: int) -> float:
        self._ensure(qid)
        x = self._feat(user_features, question_features).reshape(-1,1)
        A_inv = np.linalg.inv(self.A[qid])
        theta = A_inv @ self.b[qid]
        p = float((theta.T @ x) + self.alpha * np.sqrt((x.T @ A_inv @ x)))
        return p

    def update(self, user_features: Dict[str, float], question_features: Dict[str, float], qid: int, reward: float):
        self._ensure(qid)
        x = self._feat(user_features, question_features).reshape(-1,1)
        self.A[qid] += x @ x.T
        self.b[qid] += reward * x
        self._save(qid)

    def _feat(self, uf: Dict[str, float], qf: Dict[str, float]) -> np.ndarray:
        # simple fixed-order projection
        keys = sorted(list(uf.keys()) + list(qf.keys()))[:self.d]
        vec = np.zeros(self.d, dtype=np.float32)
        for i,k in enumerate(keys):
            vec[i] = float(uf.get(k, 0.0) if k in uf else qf.get(k, 0.0))
        return vec
