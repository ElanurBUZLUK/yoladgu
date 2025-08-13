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
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_RECYCLE_S: int = 1800
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
    # CORS
    CORS_ALLOW_ORIGINS: str | None = None  # comma-separated origins, e.g., "https://app.example.com,https://admin.example.com"
    # Feature/Peer backends
    FEATURE_STORE_BACKEND: str = "db"     # "db" only for now
    FEATURE_WINDOW_DAYS: int = 30
    PEER_STORE_BACKEND: str = "memory"    # "memory" | "redis"
    PEER_STORE_PREFIX: str = "peer"

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
    W_PEER: float = 0.0
    CF_MODEL_PATH: str | None = None
    FEATURE_WINDOW_DAYS: int = 30
    FEATURE_CACHE_TTL_S: int = 600
    # LLM for explanations
    LLM_PROVIDER: str | None = None  # openai|cohere|anthropic|hf|google|none
    LLM_MODEL_ID: str | None = None  # e.g., gpt-4o-mini or gemini-1.5-flash
    EXPLAIN_CACHE_TTL_S: int = 3600
    EXPLAIN_MAX_TOKENS: int = 256
    EXPLAIN_TEMPERATURE: float = 0.2

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

    # Peer hardness settings
    PEER_MIN_NEIGHBORS: int = 5
    PEER_K_NEIGHBORS: int = 20
    PEER_LAMBDA_EASY: float = 0.5
    PEER_HALF_LIFE_DAYS: float = 14.0
    PEER_LOOKBACK_DAYS: int = 90
    PEER_GATE_ENABLE: bool = False
    PEER_GATE_P_MIN: float = 0.6
    PEER_GATE_P_MAX: float = 0.75
    PEER_STORE_BACKEND: str = "memory"  # memory | redis | postgres
    PEER_STORE_PREFIX: str = "peer"

    # RAG / Advanced Retrieval flags
    RERANK_ENABLED: bool = True
    RERANK_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_MODEL: str | None = None
    HYBRID_KEYWORD_BACKEND: str = "none"  # none | postgres | elastic
    KNOWLEDGE_GRAPH_ENABLED: bool = False
    SEMANTIC_CACHE_TTL_S: int = 900

    # MCP integration
    MCP_ENABLED: bool = False
    MCP_BASE_URL: str | None = None  # e.g., http://mcp.retriever.svc.cluster.local:7800

    # Model serving (TorchServe/Triton)
    SERVING_PROVIDER: str = "none"  # none | ts | triton
    TS_URL: str | None = None        # e.g., http://localhost:8080
    TS_MODEL_NAME: str | None = None # e.g., rec
    # KServe
    KS_URL: str | None = None        # e.g., http://kserve.default.svc.cluster.local
    KS_MODEL_NAME: str | None = None # e.g., rec
    KS_INFER_PATH: str = "/v2/models/{model}/infer"

    # Vector sharding
    VECTOR_SHARDS: int = 1

    # Archival (S3)
    S3_BUCKET: str | None = None
    AWS_REGION: str | None = None
    ARCHIVE_LOOKBACK_DAYS: int = 180
    ARCHIVE_BATCH_SIZE: int = 10000

    # MLflow
    MLFLOW_TRACKING_URI: str | None = None
    MLFLOW_EXPERIMENT: str | None = "default"

    # Vault
    VAULT_ADDR: str | None = None
    VAULT_TOKEN: str | None = None
    VAULT_KV_PATH: str | None = None  # e.g., kv/data/app

    model_config = SettingsConfigDict(env_file=os.getenv("ENV_FILE", ".env"), env_file_encoding="utf-8")

settings = Settings()
