from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "LearnAI"
    API_V1_STR: str = "/api/v1"
    ENV: str = "dev"

    JWT_SECRET: str = "supersecretjwt"
    JWT_ALG: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MIN: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    DATABASE_URL: str = "postgresql+asyncpg://learnai:learnai@localhost:55432/learnai"
    REDIS_URL: str = "redis://localhost:16379/0"

    VECTOR_BACKEND: str = "hnsw"  # hnsw | faiss | qdrant
    VECTOR_INDEX_DIR: str = "../data/indices"
    VECTOR_ACTIVE_KEY: str = "vector:index:active"
    VECTOR_BLUE_NAME: str = "blue"
    VECTOR_GREEN_NAME: str = "green"
    VECTOR_STORE_TOMBSTONE_KEY: str = "vector:tombstones"
    EMBED_DIM: int = 384
    EMBEDDING_MODEL: str | None = None  # e.g., "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_PROVIDER: str = "hash"    # "hash" | "sbert"

    # HNSW
    HNSW_SPACE: str = "l2"
    HNSW_M: int = 64
    HNSW_EF_CONSTRUCT: int = 200
    HNSW_EF_SEARCH: int = 64

    # FAISS
    FAISS_NLIST: int = 100
    FAISS_PQ_M: int = 48
    FAISS_PQ_BITS: int = 8
    FAISS_NPROBE: int = 8

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION_BLUE: str = "questions_blue"
    QDRANT_COLLECTION_GREEN: str = "questions_green"
    QDRANT_ALIAS_ACTIVE: str = "questions_active"

    # Ensemble weights
    W_CF: float = 0.25
    W_BANDIT: float = 0.35
    W_ONLINE: float = 0.40
    W_RETR: float = 0.0
    CF_MODEL_PATH: str | None = None

    # Feast (optional)
    FEAST_ENABLED: bool = False
    FEAST_REPO_PATH: str | None = None
    FEAST_REGISTRY_PATH: str | None = None
    FEAST_ONLINE_REDIS_URL: str | None = None

    # Adaptive rating params
    RATING_BETA: float = 0.0025
    K_STUDENT: float = 32.0
    K_QUESTION: float = 16.0
    ALPHA_TIME: float = 1.5
    TOLERANCE: float = 150.0
    EMA_TREF_ETA: float = 0.05
    REBUCKET_MIN_ATTEMPTS: int = 30
    REBUCKET_COOLDOWN_DAYS: int = 14
    REBUCKET_PCT: str = "25,60,85"
    EXPLORE_RATIO: float = 0.2

    model_config = SettingsConfigDict(env_file=os.getenv("ENV_FILE", ".env"), env_file_encoding="utf-8")

settings = Settings()
