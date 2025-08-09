import numpy as np
from typing import Dict

class LinUCBService:
    def __init__(self, feature_dim: int = 16, alpha: float = 0.2):
        self.d = feature_dim
        self.alpha = alpha
        # For demo, per-question A, b are kept in-memory
        self.A = {}  # qid -> A (dxd)
        self.b = {}  # qid -> b (dx1)

    def _ensure(self, qid: int):
        if qid not in self.A:
            self.A[qid] = np.eye(self.d)
            self.b[qid] = np.zeros((self.d, 1))

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

    def _feat(self, uf: Dict[str, float], qf: Dict[str, float]) -> np.ndarray:
        # simple fixed-order projection
        keys = sorted(list(uf.keys()) + list(qf.keys()))[:self.d]
        vec = np.zeros(self.d, dtype=np.float32)
        for i,k in enumerate(keys):
            vec[i] = float(uf.get(k, 0.0) if k in uf else qf.get(k, 0.0))
        return vec
