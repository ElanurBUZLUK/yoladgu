from pydantic_settings import BaseSettings
from typing import List, Optional
from enum import Enum
import os


class Environment(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class LLMProvider(str, Enum):
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT35 = "openai_gpt35"
    ANTHROPIC_CLAUDE_OPUS = "anthropic_claude_opus"
    ANTHROPIC_CLAUDE_SONNET = "anthropic_claude_sonnet"
    ANTHROPIC_CLAUDE_HAIKU = "anthropic_claude_haiku"
    LOCAL_MODEL = "local_model"


class Settings(BaseSettings):
    # App
    app_name: str = "Adaptive Question System"
    version: str = "1.0.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    
    # Database
    database_url: str = "postgresql://user:password@localhost:5432/adaptive_learning"
    test_database_url: Optional[str] = None
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    encryption_key: str = "your-encryption-key-change-in-production"
    
    # LLM Configuration
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    primary_llm_provider: str = "gpt4"
    secondary_llm_provider: str = "claude_haiku"
    daily_llm_budget: float = 100.0
    
    # LLM Fallback Settings
    enable_llm_fallback: bool = True
    fallback_to_templates: bool = True
    max_retry_attempts: int = 3
    
    # Embedding Configuration
    embedding_model: str = "text-embedding-3-small"
    local_embedding_model: Optional[str] = None  # e.g., "bge-small-en"
    pgvector_dim: int = 1536
    embedding_cache_ttl: int = 3600  # 1 hour
    embedding_batch_size: int = 100
    embedding_rate_limit_delay: float = 0.1  # 100ms
    
    # CORS
    cors_origins: str = "http://localhost:3000"
    
    # File Upload
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    upload_dir: str = "./uploads"
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    
    # Monitoring
    prometheus_enabled: bool = True
    
    # JWT Settings
    jwt_secret: str = "your-jwt-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    
    # MCP Settings
    mcp_server_url: str = "http://localhost:3001"
    mcp_timeout: int = 30
    
    # Vector Database Settings
    pgvector_enabled: bool = True
    vector_similarity_threshold: float = 0.7
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIM", "1536"))
    vector_batch_size: int = int(os.getenv("VECTOR_BATCH_SIZE", "100"))
    vector_namespace_default: str = "default"
    vector_slot_default: int = 1
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def cors_origins_list(self) -> List[str]:
        if isinstance(self.cors_origins, str):
            return [origin.strip() for origin in self.cors_origins.split(",")]
        return self.cors_origins


settings = Settings()