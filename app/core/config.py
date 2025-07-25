from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, validator
from typing import List, Optional
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Question Recommendation System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    # PostgreSQL Configuration
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+psycopg2://kullanici:sifre@localhost:5432/veritabani")

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

    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v, values):
        if v:
            return v
        return f"postgresql://{values['POSTGRES_USER']}:" \
               f"{values['POSTGRES_PASSWORD']}@" \
               f"{values['POSTGRES_SERVER']}/" \
               f"{values['POSTGRES_DB']}"

    @property
    def redis_url(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 