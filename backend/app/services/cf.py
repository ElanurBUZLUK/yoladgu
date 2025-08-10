from typing import Dict, Tuple, Optional
import os
import numpy as np


class CFModel:
    def __init__(self):
        self.user_map: Dict[int, int] = {}
        self.item_map: Dict[int, int] = {}
        self.U: Optional[np.ndarray] = None
        self.V: Optional[np.ndarray] = None

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        data = np.load(path, allow_pickle=True)
        self.U = data["U"]
        self.V = data["V"]
        self.user_map = data["user_map"].item()
        self.item_map = data["item_map"].item()
        return True

    def score(self, user_id: int, question_id: int) -> float:
        if self.U is None or self.V is None:
            return 0.0
        ui = self.user_map.get(int(user_id))
        qi = self.item_map.get(int(question_id))
        if ui is None or qi is None:
            return 0.0
        s = float(self.U[ui] @ self.V[qi])
        # sigmoid to [0,1]
        return 1.0 / (1.0 + np.exp(-s))


