import os
import json
import time
import hashlib
from typing import List, Sequence, Optional, Tuple
from dataclasses import dataclass
from tenacity import retry, wait_exponential, stop_after_attempt
import redis
import structlog
from .pii_masker import mask_text
from app.core.config import settings
from app.services.embedding_service import EmbeddingService


PROVIDER_OPENAI = "openai"
PROVIDER_COHERE = "cohere"


@dataclass
class EmbeddingConfig:
    provider: str = os.getenv("EMBEDDING_PROVIDER", PROVIDER_OPENAI).lower()
    model_id: str = os.getenv("EMBEDDING_MODEL_ID", "text-embedding-3-small")
    pii_mask: bool = os.getenv("EMBEDDING_PII_MASK", "1") == "1"
    timeout_s: int = int(os.getenv("EMBEDDING_TIMEOUT_S", "20"))
    # Cache and batching
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "128"))
    redis_url: str = os.getenv("REDIS_URL", settings.REDIS_URL)
    cache_ttl_s: int = int(os.getenv("EMBEDDING_CACHE_TTL_S", "604800"))  # 7 days
    cache_prefix: str = os.getenv("EMBEDDING_CACHE_PREFIX", "embedding:v1")
    # Fallback
    enable_fallback: bool = os.getenv("EMBEDDING_ENABLE_FALLBACK", "1") == "1"
    fallback_mode: str = os.getenv("EMBEDDING_FALLBACK_MODE", "sbert")  # sbert|hash

    embedding_dim: Optional[int] = (int(os.getenv("EMBEDDING_DIM", "0")) or None)


class EmbeddingProvider:
    def __init__(self, cfg: Optional[EmbeddingConfig] = None) -> None:
        self.cfg = cfg or EmbeddingConfig()
        logger = structlog.get_logger()
        self.logger = logger  # prefer logger alias [[memory:5190708]]
        # Redis client for cache (optional)
        try:
            self.r = redis.Redis.from_url(self.cfg.redis_url, decode_responses=True) if self.cfg.redis_url else None
        except Exception:
            self.r = None
        self.logger.info("embedding_provider.init", provider=self.cfg.provider, model=self.cfg.model_id, redis=bool(self.r))
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
        # Local fallback engine (hash/sbert via EmbeddingService)
        self._local = EmbeddingService(dim=self.cfg.embedding_dim or settings.EMBED_DIM)

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

    # -------- Caching helpers --------
    def _normalize(self, text: str) -> str:
        return " ".join((text or "").strip().split())

    def _key_for(self, text: str) -> str:
        material = f"{self.cfg.provider}::{self.cfg.model_id}::{self._normalize(text)}"
        hid = hashlib.sha256(material.encode("utf-8")).hexdigest()
        return f"{self.cfg.cache_prefix}:{hid}"

    def _cache_get_many(self, keys: List[str]) -> List[Optional[List[float]]]:
        if not self.r:
            return [None] * len(keys)
        vals = self.r.mget(keys)
        out: List[Optional[List[float]]] = []
        for v in vals:
            if not v:
                out.append(None)
            else:
                try:
                    out.append(json.loads(v))
                except Exception:
                    out.append(None)
        return out

    def _cache_set_many(self, key_vec_pairs: List[Tuple[str, List[float]]]) -> None:
        if not self.r or not key_vec_pairs:
            return
        pipe = self.r.pipeline()
        for k, vec in key_vec_pairs:
            try:
                pipe.setex(k, self.cfg.cache_ttl_s, json.dumps(vec))
            except Exception:
                pass
        try:
            pipe.execute()
        except Exception:
            pass

    def _chunks(self, idxs: List[int], size: int) -> List[List[int]]:
        return [idxs[i:i+size] for i in range(0, len(idxs), size)]

    def _fallback_embed(self, texts: Sequence[str]) -> List[List[float]]:
        mode = self.cfg.fallback_mode if self.cfg.fallback_mode in ("sbert", "hash") else "hash"
        prev = self._local.mode
        self._local.mode = mode
        try:
            X = self._local.embed_sync(list(texts))
            return [v.astype(float).tolist() for v in X]
        finally:
            self._local.mode = prev

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        texts_list = list(texts)
        if not texts_list:
            return []
        keys = [self._key_for(t) for t in texts_list]
        cached = self._cache_get_many(keys)
        results: List[Optional[List[float]]] = cached[:]
        miss_idx = [i for i, v in enumerate(results) if v is None]

        if not miss_idx:
            self.logger.info("embedding.cache_hit_all", count=len(texts_list))
            # type: ignore - safe because no misses
            return [v for v in results if v is not None]

        bs = max(1, int(self.cfg.batch_size))
        staged: List[Tuple[str, List[float]]] = []
        for group in self._chunks(miss_idx, bs):
            inputs = [texts_list[i] for i in group]
            try:
                vecs = self._backend(inputs)
            except Exception as e:
                if self.cfg.enable_fallback:
                    self.logger.warning("embedding.api_error_fallback", error=str(e), n=len(inputs))
                    vecs = self._fallback_embed(inputs)
                else:
                    raise
            for i, v in zip(group, vecs):
                results[i] = v
                staged.append((keys[i], v))
            time.sleep(0)  # yield

        self._cache_set_many(staged)
        return [r if r is not None else [] for r in results]

    def embed_one(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]


_singleton: Optional[EmbeddingProvider] = None


def get_embedding_provider() -> EmbeddingProvider:
    global _singleton
    if _singleton is None:
        _singleton = EmbeddingProvider()
    return _singleton


