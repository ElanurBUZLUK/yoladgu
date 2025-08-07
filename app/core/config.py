"""
Application Configuration
Uygulama konfigürasyonu
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import validator

class Settings(BaseSettings):
    """Application settings"""
    
    # ============================================================================
    # BASIC SETTINGS
    # ============================================================================
    
    PROJECT_NAME: str = "Yoladgu"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # ============================================================================
    # DATABASE SETTINGS
    # ============================================================================
    
    DATABASE_URL: str = "postgresql://yoladgu_user:yoladgu123@localhost/yoladgu"
    ASYNC_DATABASE_URL: str = "postgresql+asyncpg://yoladgu_user:yoladgu123@localhost/yoladgu"
    
    # ============================================================================
    # REDIS SETTINGS
    # ============================================================================
    
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_MAX_CONNECTIONS: int = 20
    
    # ============================================================================
    # NEO4J SETTINGS
    # ============================================================================
    
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    
    # ============================================================================
    # CORS SETTINGS
    # ============================================================================
    
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:4200"]
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # ============================================================================
    # AUTHENTICATION SETTINGS
    # ============================================================================
    
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # ============================================================================
    # ML/AI SETTINGS
    # ============================================================================
    
    # Embedding settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 50
    EMBEDDING_CACHE_TTL: int = 3600  # 1 saat
    
    # Recommendation settings
    RECOMMENDATION_CACHE_TTL: int = 300  # 5 dakika
    RECOMMENDATION_BATCH_SIZE: int = 100
    RECOMMENDATION_SIMILARITY_THRESHOLD: float = 0.7
    
    # Vector search settings
    VECTOR_SEARCH_CACHE_TTL: int = 600  # 10 dakika
    VECTOR_SEARCH_SIMILARITY_THRESHOLD: float = 0.5
    VECTOR_SEARCH_BATCH_SIZE: int = 30
    
    # FAISS settings
    FAISS_INDEX_PATH: str = "data/faiss.index"
    FAISS_INDEX_DIR: str = "/data/faiss"
    FAISS_DIM: int = 384
    FAISS_NLIST: int = 128
    FAISS_M: int = 64  # PQ alt-vokabül sayısı
    FAISS_NBITS: int = 8
    FAISS_NPROBE: int = 10
    FAISS_PQ_M: int = 64
    FAISS_PQ_BITS: int = 8
    
    # ML Model settings
    CF_N_FACTORS: int = 50
    CF_N_NEIGHBORS: int = 100
    BANDIT_ALPHA: float = 0.1
    BANDIT_LR: float = 0.01
    ONLINE_HIDDEN_SIZE: int = 64
    ONLINE_LR: float = 0.001
    
    # Neo4j settings
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    
    # Vector store settings
    VECTOR_STORE_CACHE_TTL: int = 7200  # 2 saat
    VECTOR_STORE_TOMBSTONE_KEY: str = "deleted_vectors"
    VECTOR_STORE_EMBEDDING_PREFIX: str = "embedding:question"
    
    # ============================================================================
    # OBSERVABILITY SETTINGS
    # ============================================================================
    
    # OpenTelemetry settings
    ENABLE_TRACING: bool = True
    JAEGER_HOST: str = "localhost"
    JAEGER_PORT: int = 6831
    
    # Prometheus settings
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 8000
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # ============================================================================
    # AUTO SCALING SETTINGS
    # ============================================================================
    
    ENABLE_AUTO_SCALING: bool = False
    AUTO_SCALING_INTERVAL: int = 30  # saniye
    AUTO_SCALING_COOLDOWN: int = 60  # saniye
    
    # CPU thresholds
    CPU_THRESHOLD_HIGH: float = 80.0
    CPU_THRESHOLD_LOW: float = 20.0
    
    # Memory thresholds
    MEMORY_THRESHOLD_HIGH: float = 85.0
    MEMORY_THRESHOLD_LOW: float = 30.0
    
    # Connection pool thresholds
    CONNECTION_THRESHOLD_HIGH: float = 0.8
    CONNECTION_THRESHOLD_LOW: float = 0.3
    
    # Scaling limits
    MIN_CONNECTIONS: int = 5
    MAX_CONNECTIONS: int = 50
    SCALING_FACTOR: float = 1.5
    
    # ============================================================================
    # BATCH MANAGEMENT SETTINGS
    # ============================================================================
    
    ENABLE_DYNAMIC_BATCH: bool = True
    BATCH_ADJUSTMENT_COOLDOWN: int = 30  # saniye
    
    # Batch size limits
    EMBEDDING_MIN_BATCH_SIZE: int = 10
    EMBEDDING_MAX_BATCH_SIZE: int = 100
    RECOMMENDATION_MIN_BATCH_SIZE: int = 5
    RECOMMENDATION_MAX_BATCH_SIZE: int = 50
    VECTOR_SEARCH_MIN_BATCH_SIZE: int = 5
    VECTOR_SEARCH_MAX_BATCH_SIZE: int = 30
    
    # Target latencies
    EMBEDDING_TARGET_LATENCY: float = 2.0  # saniye
    RECOMMENDATION_TARGET_LATENCY: float = 1.0  # saniye
    VECTOR_SEARCH_TARGET_LATENCY: float = 0.5  # saniye
    
    # Target throughputs
    EMBEDDING_TARGET_THROUGHPUT: float = 50.0  # items/s
    RECOMMENDATION_TARGET_THROUGHPUT: float = 20.0  # items/s
    VECTOR_SEARCH_TARGET_THROUGHPUT: float = 100.0  # items/s
    
    # ============================================================================
    # FALLBACK SETTINGS
    # ============================================================================
    
    ENABLE_FALLBACK_CHAINS: bool = True
    FALLBACK_HISTORY_SIZE: int = 100
    
    # Fallback timeouts
    CACHE_FALLBACK_TIMEOUT: float = 1.0  # saniye
    BASIC_SEARCH_FALLBACK_TIMEOUT: float = 2.0  # saniye
    RECENT_ITEMS_FALLBACK_TIMEOUT: float = 1.0  # saniye
    STATIC_RESPONSE_FALLBACK_TIMEOUT: float = 0.5  # saniye
    
    # ============================================================================
    # RATE LIMITING SETTINGS
    # ============================================================================
    
    ENABLE_RATE_LIMITING: bool = True
    RATE_LIMIT_PER_MINUTE: int = 100
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # ============================================================================
    # CACHE SETTINGS
    # ============================================================================
    
    ENABLE_CACHING: bool = True
    CACHE_DEFAULT_TTL: int = 300  # 5 dakika
    CACHE_MAX_SIZE: int = 1000
    
    # ============================================================================
    # FEATURE FLAGS
    # ============================================================================
    
    FEATURE_ADVANCED_METRICS: bool = True
    FEATURE_AUTO_SCALING: bool = False
    FEATURE_DYNAMIC_TTL: bool = True
    FEATURE_FALLBACK_CHAINS: bool = True
    FEATURE_BATCH_OPTIMIZATION: bool = True
    
    # ============================================================================
    # EXTERNAL SERVICES
    # ============================================================================
    
    # OpenAI settings
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OPENAI_MAX_TOKENS: int = 1000
    
    # Anthropic settings
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-sonnet-20240229"
    ANTHROPIC_MAX_TOKENS: int = 1000
    
    # Google settings
    GOOGLE_API_KEY: Optional[str] = None
    GOOGLE_API_VERSION: str = "v1beta"
    
    # HuggingFace settings
    HUGGINGFACE_API_TOKEN: Optional[str] = None
    HUGGINGFACE_MODEL: str = "microsoft/DialoGPT-medium"
    
    # ============================================================================
    # DATABASE SETTINGS (Additional)
    # ============================================================================
    
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "yoladgu_user"
    POSTGRES_PASSWORD: str = "yoladgu123"
    POSTGRES_DB: str = "yoladgu"
    
    # Async database settings
    ASYNCPG_MIN_SIZE: int = 5
    ASYNCPG_MAX_SIZE: int = 20
    
    # ============================================================================
    # ML/AI SETTINGS (Additional)
    # ============================================================================
    
    MODEL_CACHE_DIR: str = "./models"
    LEARNING_RATE: str = "0.01"
    EMBEDDING_DIM: str = "384"
    
    # LLM settings
    LLM_MODEL: str = "gemini-1.5-pro"
    LLM_MAX_TOKENS: str = "1000"
    LLM_TEMPERATURE: str = "0.7"
    LLM_PROVIDER: str = "huggingface"
    
    # Feature flags
    USE_NEO4J: str = "true"
    USE_EMBEDDING: str = "true"
    USE_DIVERSITY_FILTER: str = "true"
    USE_DLQ: str = "true"
    USE_ENSEMBLE_SCORING: str = "true"
    USE_PROMETHEUS_HISTOGRAM: str = "true"
    
    # Cache settings
    CACHE_TTL: str = "300"
    MAX_RECOMMENDATIONS: str = "20"
    SIMILARITY_THRESHOLD: str = "0.6"
    MAX_DIFFICULTY_GAP: str = "2"
    
    # Environment
    ENVIRONMENT: str = "development"
    
    # ============================================================================
    # VALIDATORS
    # ============================================================================
    
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    @validator("ALLOWED_HOSTS", pre=True)
    def assemble_allowed_hosts(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
    }


# Global settings instance
settings = Settings()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_database_url() -> str:
    """Get database URL from settings"""
    return settings.DATABASE_URL

def get_async_database_url() -> str:
    """Get async database URL from settings"""
    return settings.ASYNC_DATABASE_URL

def get_redis_url() -> str:
    """Get Redis URL from settings"""
    return settings.REDIS_URL

def get_neo4j_uri() -> str:
    """Get Neo4j URI from settings"""
    return settings.NEO4J_URI

def is_development() -> bool:
    """Check if environment is development"""
    return settings.ENVIRONMENT.lower() == "development"

def is_production() -> bool:
    """Check if environment is production"""
    return settings.ENVIRONMENT.lower() == "production"

def is_testing() -> bool:
    """Check if environment is testing"""
    return settings.ENVIRONMENT.lower() == "testing"

def is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature is enabled"""
    feature_map = {
        "advanced_metrics": settings.FEATURE_ADVANCED_METRICS,
        "auto_scaling": settings.FEATURE_AUTO_SCALING,
        "dynamic_ttl": settings.FEATURE_DYNAMIC_TTL,
        "fallback_chains": settings.FEATURE_FALLBACK_CHAINS,
        "batch_optimization": settings.FEATURE_BATCH_OPTIMIZATION,
    }
    return feature_map.get(feature_name, False)

def get_feature_config(feature_name: str) -> dict:
    """Get feature-specific configuration"""
    configs = {
        "embedding": {
            "model": settings.EMBEDDING_MODEL,
            "dimension": settings.EMBEDDING_DIMENSION,
            "batch_size": settings.EMBEDDING_BATCH_SIZE,
            "cache_ttl": settings.EMBEDDING_CACHE_TTL,
        },
        "vector_search": {
            "cache_ttl": settings.VECTOR_SEARCH_CACHE_TTL,
            "similarity_threshold": settings.VECTOR_SEARCH_SIMILARITY_THRESHOLD,
            "batch_size": settings.VECTOR_SEARCH_BATCH_SIZE,
        },
        "recommendation": {
            "cache_ttl": settings.RECOMMENDATION_CACHE_TTL,
            "batch_size": settings.RECOMMENDATION_BATCH_SIZE,
            "similarity_threshold": settings.RECOMMENDATION_SIMILARITY_THRESHOLD,
        },
    }
    return configs.get(feature_name, {})
