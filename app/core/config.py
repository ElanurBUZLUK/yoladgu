from typing import Dict, List, Optional

from pydantic import AnyHttpUrl, ConfigDict, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    PROJECT_NAME: str = "Question Recommendation System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"

    # Security Configuration - Must be set in environment variables
    SECRET_KEY: str  # Required: Strong secret key for JWT tokens
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    # PostgreSQL Configuration - Must be set in environment variables
    POSTGRES_SERVER: str  # Required: PostgreSQL server host
    POSTGRES_USER: str  # Required: PostgreSQL username
    POSTGRES_PASSWORD: str  # Required: PostgreSQL password
    POSTGRES_DB: str  # Required: PostgreSQL database name
    DATABASE_URL: Optional[str] = None  # Auto-generated from above

    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None  # Set if Redis requires auth

    # Neo4j Configuration - Must be set in environment variables
    NEO4J_URI: str  # Required: Neo4j connection URI
    NEO4J_USER: str  # Required: Neo4j username
    NEO4J_PASSWORD: str  # Required: Neo4j password

    # ML Model Configuration
    MODEL_CACHE_DIR: str = "./models"
    RECOMMENDATION_BATCH_SIZE: int = 100
    LEARNING_RATE: float = 0.01

    # Embedding Configuration
    EMBEDDING_MODEL: str = "paraphrase-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    EMBEDDING_BATCH_SIZE: int = 50

    # === PERFORMANCE TUNING ===
    # Vector Store Performance
    HNSW_M: int = 16  # HNSW index connectivity
    HNSW_EF_CONSTRUCTION: int = 200  # Index build quality
    HNSW_EF_SEARCH: int = 100  # Search quality vs speed

    # Connection Pool Settings
    ASYNCPG_MIN_SIZE: int = 5
    ASYNCPG_MAX_SIZE: int = 20
    ASYNCPG_TIMEOUT: int = 30

    # Cache Settings
    REDIS_MAX_CONNECTIONS: int = 20
    CACHE_DEFAULT_TTL: int = 300  # 5 minutes
    CACHE_EMBEDDING_TTL: int = 3600  # 1 hour
    CACHE_SEARCH_TTL: int = 600  # 10 minutes

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_BURST_MULTIPLIER: float = 1.5

    # Async Performance
    MAX_WORKERS: int = 4
    BATCH_SIZE_DEFAULT: int = 100
    BATCH_SIZE_LARGE: int = 500

    # Memory Management
    SEARCH_RESULT_LIMIT: int = 100
    CLEANUP_INTERVAL_HOURS: int = 6

    # LLM Configuration - Set in environment variables for production
    OPENAI_API_KEY: Optional[str] = None  # Required for OpenAI provider
    HUGGINGFACE_API_TOKEN: Optional[str] = None  # Required for Hugging Face models
    HUGGINGFACE_MODEL: str = "microsoft/DialoGPT-medium"
    LLM_PROVIDER: str = "huggingface"  # "openai" or "huggingface"
    LLM_MODEL: str = "gpt-3.5-turbo"  # For OpenAI
    LLM_MAX_TOKENS: int = 1000
    LLM_TEMPERATURE: float = 0.7

    # Ensemble Scoring Configuration
    ENSEMBLE_WEIGHTS: Dict[str, float] = {
        "river_score": 0.35,
        "embedding_similarity": 0.25,
        "skill_mastery": 0.20,
        "difficulty_match": 0.15,
        "neo4j_similarity": 0.05,
    }

    # Feature Flags
    USE_NEO4J: bool = True
    USE_EMBEDDING: bool = True
    USE_DIVERSITY_FILTER: bool = True
    USE_DLQ: bool = True
    USE_ENSEMBLE_SCORING: bool = True
    USE_PROMETHEUS_HISTOGRAM: bool = True

    # Performance Configuration
    CACHE_TTL: int = 300  # 5 minutes
    MAX_RECOMMENDATIONS: int = 20
    SIMILARITY_THRESHOLD: float = 0.6
    MAX_DIFFICULTY_GAP: int = 2

    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v, info):
        if v:
            return v
        values = info.data
        # Build DATABASE_URL from individual components if not provided
        postgres_user = values.get("POSTGRES_USER")
        postgres_password = values.get("POSTGRES_PASSWORD")
        postgres_server = values.get("POSTGRES_SERVER")
        postgres_db = values.get("POSTGRES_DB")

        if not all([postgres_user, postgres_password, postgres_server, postgres_db]):
            raise ValueError(
                "DATABASE_URL not provided and required PostgreSQL environment variables missing. "
                "Please set: POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_SERVER, POSTGRES_DB"
            )

        return f"postgresql://{postgres_user}:{postgres_password}@{postgres_server}/{postgres_db}"

    @property
    def redis_url(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"


settings = Settings()
