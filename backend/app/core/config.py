"""
Application configuration using Pydantic Settings.
"""

from typing import List, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    APP_NAME: str = "Adaptive Question Recommendation System"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    API_BASE_URL: str = "http://localhost:8000"
    
    # Security
    SECRET_KEY: str = "test-secret-key"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./test.db"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL_SECONDS: int = 3600
    
    # Vector Database
    VECTOR_DB_URL: str = "http://localhost:6333"
    VECTOR_DB_COLLECTION: str = "questions"
    
    # Search Engine
    SEARCH_ENGINE_URL: str = "http://localhost:9200"
    
    # LLM Service
    LLM_API_URL: Optional[str] = None
    LLM_API_KEY: Optional[str] = None
    LLM_MODEL: str = "llama3-70b"
    LLM_MAX_TOKENS: int = 2048
    LLM_TEMPERATURE: float = 0.7
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 60
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:4200"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # Bandit Configuration
    BANDIT_ALPHA: float = 1.0
    BANDIT_LAMBDA: float = 0.01
    EXPLORATION_RATIO: float = 0.2
    MIN_SUCCESS_RATE: float = 0.6
    MIN_COVERAGE: float = 0.8
    
    # IRT Configuration
    IRT_LEARNING_RATE: float = 0.1
    IRT_MAX_ITERATIONS: int = 100
    IRT_CONVERGENCE_THRESHOLD: float = 0.001
    
    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()