import os
from typing import List, Sequence, Optional
from dataclasses import dataclass
from tenacity import retry, wait_exponential, stop_after_attempt
from .pii_masker import mask_text


PROVIDER_OPENAI = "openai"
PROVIDER_COHERE = "cohere"


@dataclass
class EmbeddingConfig:
    provider: str = os.getenv("EMBEDDING_PROVIDER", PROVIDER_OPENAI).lower()
    model_id: str = os.getenv("EMBEDDING_MODEL_ID", "text-embedding-3-small")
    pii_mask: bool = os.getenv("EMBEDDING_PII_MASK", "1") == "1"
    timeout_s: int = int(os.getenv("EMBEDDING_TIMEOUT_S", "20"))
    embedding_dim: Optional[int] = (int(os.getenv("EMBEDDING_DIM", "0")) or None)


class EmbeddingProvider:
    def __init__(self, cfg: Optional[EmbeddingConfig] = None) -> None:
        self.cfg = cfg or EmbeddingConfig()
        if self.cfg.provider == PROVIDER_OPENAI:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self._backend = self._openai_embed
        elif self.cfg.provider == PROVIDER_COHERE:
            import cohere
            self.client = cohere.Client(os.getenv("COHERE_API_KEY"))
            self._backend = self._cohere_embed
        else:
            raise ValueError(f"Unsupported EMBEDDING_PROVIDER={self.cfg.provider}")

    def _prep_inputs(self, texts: Sequence[str]) -> list:
        return [mask_text(t) for t in texts] if self.cfg.pii_mask else list(texts)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5))
    def _openai_embed(self, texts: Sequence[str]) -> List[List[float]]:
        inputs = self._prep_inputs(texts)
        resp = self.client.embeddings.create(
            model=self.cfg.model_id, input=inputs, timeout=self.cfg.timeout_s
        )
        return [d.embedding for d in resp.data]

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5))
    def _cohere_embed(self, texts: Sequence[str]) -> List[List[float]]:
        inputs = self._prep_inputs(texts)
        resp = self.client.embed(texts=inputs, model=self.cfg.model_id, input_type="search_document")
        return resp.embeddings

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return self._backend(texts)

    def embed_one(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]


_singleton: Optional[EmbeddingProvider] = None


def get_embedding_provider() -> EmbeddingProvider:
    global _singleton
    if _singleton is None:
        _singleton = EmbeddingProvider()
    return _singleton


