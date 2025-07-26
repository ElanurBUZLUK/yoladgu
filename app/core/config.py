from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, field_validator, ConfigDict
from typing import List, Optional
import os

class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", case_sensitive=True, extra="ignore")
    
    PROJECT_NAME: str = "Question Recommendation System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    # PostgreSQL Configuration
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "kullanici"
    POSTGRES_PASSWORD: str = "sifre"
    POSTGRES_DB: str = "veritabani"
    DATABASE_URL: str = "postgresql+psycopg2://kullanici:sifre@localhost:5432/veritabani"

    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    # Neo4j Configuration
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # ML Model Configuration
    MODEL_CACHE_DIR: str = "./models"
    RECOMMENDATION_BATCH_SIZE: int = 100
    LEARNING_RATE: float = 0.01

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v, info):
        if v:
            return v
        values = info.data
        return f"postgresql://{values.get('POSTGRES_USER', 'kullanici')}:" \
               f"{values.get('POSTGRES_PASSWORD', 'sifre')}@" \
               f"{values.get('POSTGRES_SERVER', 'localhost')}/" \
               f"{values.get('POSTGRES_DB', 'veritabani')}"

    @property
    def redis_url(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

settings = Settings() 