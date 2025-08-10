import numpy as np
import hashlib
from typing import List
from app.core.config import settings

class EmbeddingService:
    def __init__(self, dim: int | None = None):
        self.dim = dim or settings.EMBED_DIM
        self.mode = settings.EMBEDDING_PROVIDER or "hash"
        self._sbert = None

    async def compute(self, text: str) -> list[float]:
        if self.mode == "sbert":
            return (await self._compute_sbert([text]))[0]
        else:
            h = hashlib.sha256(text.encode()).digest()
            rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
            vec = rng.normal(0, 1, size=self.dim).astype(np.float32)
            norm = np.linalg.norm(vec) + 1e-9
            return (vec / norm).tolist()

    def embed_sync(self, texts: List[str]) -> np.ndarray:
        # sync wrapper used by tools; choose provider
        if self.mode == "sbert":
            import asyncio
            return np.array(asyncio.get_event_loop().run_until_complete(self._compute_sbert(texts)), dtype=np.float32)
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
            vec = rng.normal(0, 1, size=self.dim).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            out.append(vec)
        return np.vstack(out)

    async def _compute_sbert(self, texts: List[str]) -> List[List[float]]:
        if self._sbert is None:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(settings.EMBEDDING_MODEL or "sentence-transformers/all-MiniLM-L6-v2")
        X = self._sbert.encode(texts, normalize_embeddings=True)
        return [x.tolist() for x in X]
