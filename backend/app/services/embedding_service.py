from __future__ import annotations
from typing import List, Optional
import os, asyncio, logging
import httpx

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Async embedding servisi (OpenAI).
    ENV:
      - OPENAI_API_KEY (zorunlu)
      - EMBEDDING_MODEL (default: text-embedding-3-large)
      - EMBEDDING_DIM (default: 1536)
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        dim: Optional[int] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.provider = provider.lower()
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        self.dim = int(dim or os.getenv("EMBEDDING_DIM", "1536"))
        self.timeout = timeout
        self.max_retries = max_retries

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.provider == "openai":
            return await self._embed_openai(texts)
        raise RuntimeError(f"Unknown embedding provider: {self.provider}")

    async def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": texts}

        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code in (429,) or resp.status_code >= 500:
                    await asyncio.sleep(min(1.0 * attempt, 4.0))
                    if attempt < self.max_retries:
                        continue
                resp.raise_for_status()
                data = resp.json()
                vectors = [item["embedding"] for item in data.get("data", [])]
                for v in vectors:
                    if len(v) != self.dim:
                        raise RuntimeError(f"Embedding dimension mismatch. expected={self.dim} got={len(v)}")
                return vectors
            except Exception as e:
                logger.warning("embed attempt %s failed: %s", attempt, e)
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(min(0.5 * attempt, 2.0))

embedding_service = EmbeddingService()
