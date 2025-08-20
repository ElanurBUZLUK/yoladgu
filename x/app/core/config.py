from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Any
from enum import Enum
import os
import secrets


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


class StorageBackend(str, Enum):
    LOCAL = "local"
    S3 = "s3"
    MINIO = "minio"


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Settings(BaseSettings):
    # App
    app_name: str = "Adaptive Question System"
    version: str = "1.0.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    
    # Database
    database_url: str = "postgresql://user:password@localhost:5432/testdb"
    test_database_url: Optional[str] = None  # Set to None to avoid testdb connection issues
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # Security - Production Hardening
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    encryption_key: str = "your-encryption-key-change-in-production"
    
    # JWT Settings - Enhanced
    jwt_secret: str = "your-jwt-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    jwt_issuer: str = "adaptive-learning-system"
    jwt_audience: str = "adaptive-learning-users"
    
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
    llm_timeout_seconds: int = 30
    llm_health_check_interval: int = 300  # 5 minutes
    
    # Embedding Configuration
    embedding_model: str = "text-embedding-3-small"
    local_embedding_model: Optional[str] = None  # e.g., "bge-small-en"
    pgvector_dim: int = 1536
    embedding_cache_ttl: int = 3600  # 1 hour
    embedding_batch_size: int = 100
    embedding_rate_limit_delay: float = 0.1  # 100ms
    
    # CORS - Environment-based
    cors_origins: str = "http://localhost:3000"
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_allow_headers: List[str] = ["*"]
    
    # File Upload & Storage
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    upload_dir: str = "./uploads"
    storage_backend: StorageBackend = StorageBackend.LOCAL
    
    # S3/MinIO Configuration
    s3_endpoint_url: Optional[str] = None
    s3_access_key_id: Optional[str] = None
    s3_secret_access_key: Optional[str] = None
    s3_bucket_name: str = "adaptive-learning-uploads"
    s3_region: str = "us-east-1"
    s3_use_ssl: bool = True
    
    # Rate Limiting - Enhanced
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 60
    rate_limit_requests_per_hour: int = 1000
    rate_limit_requests_per_day: int = 10000
    rate_limit_burst_size: int = 10
    rate_limit_storage_backend: str = "redis"  # redis or memory
    
    # Monitoring & Observability
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    prometheus_path: str = "/metrics"
    
    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"
    log_file: Optional[str] = None
    log_max_size: int = 100 * 1024 * 1024  # 100MB
    log_backup_count: int = 5
    
    # API Documentation
    api_docs_enabled: bool = True
    api_docs_title: str = "Adaptive Learning API"
    api_docs_description: str = "AI-powered adaptive learning system API"
    api_docs_version: str = "1.0.0"
    api_docs_contact_name: str = "API Support"
    api_docs_contact_email: str = "support@adaptive-learning.com"
    
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
    
    # Content Moderation
    content_moderation_enabled: bool = True
    forbidden_patterns: List[str] = []
    injection_detection_enabled: bool = True
    max_content_length: int = 10000
    
    # Cost Monitoring
    cost_monitoring_enabled: bool = True
    user_monthly_limit: float = 50.0
    organization_monthly_limit: float = 500.0
    endpoint_daily_limit: float = 10.0
    global_daily_limit: float = 1000.0
    
    # Testing
    test_data_cleanup: bool = True
    test_parallel_workers: int = 4
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validate_environment_config()
        self._generate_secure_keys_if_needed()

    def _validate_environment_config(self):
        """Validate configuration based on environment"""
        # Fail-fast validation for critical settings
        self._validate_critical_settings()
        
        if self.environment == Environment.PRODUCTION:
            if self.debug:
                raise ValueError("Debug mode cannot be enabled in production")
            
            if self.secret_key == "your-secret-key-change-in-production":
                raise ValueError("Secret key must be changed in production")
            
            if self.jwt_secret == "your-jwt-secret-change-in-production":
                raise ValueError("JWT secret must be changed in production")
            
            if self.encryption_key == "your-encryption-key-change-in-production":
                raise ValueError("Encryption key must be changed in production")
            
            if not self.openai_api_key and not self.anthropic_api_key:
                raise ValueError("At least one LLM API key must be configured in production")
            
            if self.storage_backend == StorageBackend.S3:
                if not all([self.s3_access_key_id, self.s3_secret_access_key, self.s3_bucket_name]):
                    raise ValueError("S3 configuration incomplete for production")

    def _validate_critical_settings(self):
        """Fail-fast validation for critical environment variables"""
        critical_errors = []
        
        # Database URL validation
        if not self.database_url or self.database_url == "postgresql://user:password@localhost:5432/testdb":
            critical_errors.append("DATABASE_URL must be configured")
        
        # Redis URL validation
        if not self.redis_url or self.redis_url == "redis://localhost:6379/0":
            critical_errors.append("REDIS_URL must be configured")
        
        # LLM API Key validation (at least one required)
        if not self.openai_api_key and not self.anthropic_api_key:
            if self.environment == Environment.PRODUCTION:
                critical_errors.append("At least one LLM API key (OPENAI_API_KEY or ANTHROPIC_API_KEY) must be configured")
            else:
                logger.warning("No LLM API keys configured - some features may not work")
        
        # MCP Demo flag validation
        if not hasattr(self, 'use_mcp_demo'):
            self.use_mcp_demo = os.getenv('USE_MCP_DEMO', 'false').lower() == 'true'
        
        # LanguageTool flag validation
        if not hasattr(self, 'use_languagetool'):
            self.use_languagetool = os.getenv('USE_LANGUAGETOOL', 'true').lower() == 'true'
        
        # Feature flags validation
        if not hasattr(self, 'enable_llm_fallback'):
            self.enable_llm_fallback = os.getenv('ENABLE_LLM_FALLBACK', 'true').lower() == 'true'
        
        if not hasattr(self, 'fallback_to_templates'):
            self.fallback_to_templates = os.getenv('FALLBACK_TO_TEMPLATES', 'true').lower() == 'true'
        
        # ELO/PFA parameters for LevelUpdateService
        if not hasattr(self, 'elo_k_factor'):
            self.elo_k_factor = float(os.getenv('ELO_K_FACTOR', '0.2'))
        
        if not hasattr(self, 'elo_tau_factor'):
            self.elo_tau_factor = float(os.getenv('ELO_TAU_FACTOR', '1.0'))
        
        # Math recommendation parameters
        if not hasattr(self, 'math_recommendation_weights'):
            self.math_recommendation_weights = {
                'difficulty_fit': float(os.getenv('MATH_WEIGHT_DIFFICULTY_FIT', '0.4')),
                'neighbor_wrongs': float(os.getenv('MATH_WEIGHT_NEIGHBOR_WRONGS', '0.4')),
                'diversity': float(os.getenv('MATH_WEIGHT_DIVERSITY', '0.2'))
            }
        
        # English cloze generation parameters
        if not hasattr(self, 'cloze_generation_params'):
            self.cloze_generation_params = {
                'max_retries': int(os.getenv('CLOZE_MAX_RETRIES', '3')),
                'self_repair_attempts': int(os.getenv('CLOZE_SELF_REPAIR_ATTEMPTS', '2')),
                'default_difficulty': int(os.getenv('CLOZE_DEFAULT_DIFFICULTY', '3'))
            }
        
        # CEFR assessment parameters
        if not hasattr(self, 'cefr_assessment_params'):
            self.cefr_assessment_params = {
                'max_retries': int(os.getenv('CEFR_MAX_RETRIES', '3')),
                'strict_validation': os.getenv('CEFR_STRICT_VALIDATION', 'true').lower() == 'true',
                'confidence_threshold': float(os.getenv('CEFR_CONFIDENCE_THRESHOLD', '0.7'))
            }
        
        if critical_errors:
            error_message = "Critical configuration errors:\n" + "\n".join(f"- {error}" for error in critical_errors)
            raise ValueError(error_message)

    def _generate_secure_keys_if_needed(self):
        """Generate secure keys for development/testing if not provided"""
        if self.environment in [Environment.DEVELOPMENT, Environment.TESTING]:
            if self.secret_key == "your-secret-key-change-in-production":
                self.secret_key = secrets.token_urlsafe(32)
            
            if self.jwt_secret == "your-jwt-secret-change-in-production":
                self.jwt_secret = secrets.token_urlsafe(32)
            
            if self.encryption_key == "your-encryption-key-change-in-production":
                self.encryption_key = secrets.token_urlsafe(32)

    @property
    def cors_origins_list(self) -> List[str]:
        if isinstance(self.cors_origins, str):
            return [origin.strip() for origin in self.cors_origins.split(",")]
        return self.cors_origins

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_testing(self) -> bool:
        return self.environment == Environment.TESTING

    @property
    def llm_providers_available(self) -> List[str]:
        """Return list of available LLM providers based on API keys"""
        providers = []
        if self.openai_api_key:
            providers.extend([LLMProvider.OPENAI_GPT4, LLMProvider.OPENAI_GPT35])
        if self.anthropic_api_key:
            providers.extend([
                LLMProvider.ANTHROPIC_CLAUDE_OPUS,
                LLMProvider.ANTHROPIC_CLAUDE_SONNET,
                LLMProvider.ANTHROPIC_CLAUDE_HAIKU
            ])
        if self.local_embedding_model:
            providers.append(LLMProvider.LOCAL_MODEL)
        return providers

    @property
    def storage_config(self) -> Dict[str, Any]:
        """Return storage configuration based on backend"""
        if self.storage_backend == StorageBackend.S3:
            return {
                "backend": "s3",
                "endpoint_url": self.s3_endpoint_url,
                "access_key_id": self.s3_access_key_id,
                "secret_access_key": self.s3_secret_access_key,
                "bucket_name": self.s3_bucket_name,
                "region": self.s3_region,
                "use_ssl": self.s3_use_ssl
            }
        else:
            return {
                "backend": "local",
                "upload_dir": self.upload_dir
            }

    @property
    def rate_limit_config(self) -> Dict[str, Any]:
        """Return rate limiting configuration"""
        return {
            "enabled": self.rate_limit_enabled,
            "requests_per_minute": self.rate_limit_requests_per_minute,
            "requests_per_hour": self.rate_limit_requests_per_hour,
            "requests_per_day": self.rate_limit_requests_per_day,
            "burst_size": self.rate_limit_burst_size,
            "storage_backend": self.rate_limit_storage_backend
        }

    @property
    def monitoring_config(self) -> Dict[str, Any]:
        """Return monitoring configuration"""
        return {
            "prometheus_enabled": self.prometheus_enabled,
            "prometheus_port": self.prometheus_port,
            "prometheus_path": self.prometheus_path,
            "log_level": self.log_level,
            "log_format": self.log_format,
            "log_file": self.log_file
        }


settings = Settings()