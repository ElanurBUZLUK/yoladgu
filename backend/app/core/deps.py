from functools import lru_cache
from app.services.embedding_service import EmbeddingService
from app.services.vector_index_manager import VectorIndexManager
from app.core.config import settings
from app.services.bandit.linucb import LinUCBService
from app.services.online.ftrl import FTRLService

@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(settings.EMBED_DIM)

@lru_cache(maxsize=1)
def get_index_manager() -> VectorIndexManager:
    return VectorIndexManager(settings.REDIS_URL)

@lru_cache(maxsize=1)
def get_linucb_service() -> LinUCBService:
    return LinUCBService(feature_dim=16, alpha=0.2)

@lru_cache(maxsize=1)
def get_ftrl_service() -> FTRLService:
    return FTRLService(dim=32, alpha=0.1, beta=1.0, l1=0.0, l2=1.0)
