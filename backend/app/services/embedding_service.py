import numpy as np
import hashlib
from app.core.config import settings

class EmbeddingService:
    def __init__(self, dim: int | None = None):
        self.dim = dim or settings.EMBED_DIM

    async def compute(self, text: str) -> list[float]:
        # Deterministic "hash embedding" (demo): split SHA256 into floats
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
        vec = rng.normal(0, 1, size=self.dim).astype(np.float32)
        # L2 normalize
        norm = np.linalg.norm(vec) + 1e-9
        return (vec / norm).tolist()
