from __future__ import annotations

from typing import List, Dict, Any, Tuple
import numpy as np
from app.core.config import settings
from app.services.vector_index_manager import VectorIndexManager
from app.services.embedding_api import get_embedding_provider
from app.services.content.questions_service import QuestionsService


class RetrieverService:
    def __init__(self):
        self.idx = VectorIndexManager(settings.REDIS_URL)
        self.embed = get_embedding_provider()
        self.qsvc = QuestionsService(settings.REDIS_URL)

    def search(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        vec = np.array([self.embed.embed_one(text)], dtype=np.float32)
        ids, dists = self.idx.search(vec[0], k=k)
        out: List[Dict[str, Any]] = []
        for i, d in zip(ids, dists):
            meta = self.qsvc.get(int(i)) or {}
            out.append({"id": int(i), "dist": float(d), "meta": meta})
        return out


