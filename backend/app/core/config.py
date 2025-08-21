from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Any
from enum import Enum
import os
import secrets
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/adaptive_learning"
    database_echo: bool = False
    database_pool_size: int = 20
    database_max_overflow: int = 30
    database_pool_timeout: int = 30
    test_database_url: Optional[str] = None  # Set to None to avoid testdb connection issues
    
    # Redis
    redis_url: Optional[str] = None  # Will be auto-configured for development
    
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
    
    # LLM Configuration - Secure API Key Management
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
    use_mcp_demo: bool = False
    
    # Vector Database Settings
    pgvector_enabled: bool = True
    vector_similarity_threshold: float = 0.7
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIM", "1536"))
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "1536"))  # Alias for backward compatibility
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
    
    # ELO/PFA parameters for LevelUpdateService
    elo_k_factor: float = 0.2
    elo_tau_factor: float = 1.0
    
    # Math recommendation parameters - Individual fields for environment loading
    math_weight_difficulty_fit: float = 0.4
    math_weight_neighbor_wrongs: float = 0.4
    math_weight_diversity: float = 0.2
    
    # Math recommendation parameters
    math_recommendation_weights: Dict[str, float] = {
        'difficulty_fit': 0.4,
        'neighbor_wrongs': 0.4,
        'diversity': 0.2
    }
    
    # English cloze generation parameters - Individual fields for environment loading
    cloze_max_retries: int = 3
    cloze_self_repair_attempts: int = 2
    cloze_default_difficulty: int = 3
    
    # English cloze generation parameters
    cloze_generation_params: Dict[str, Any] = {
        'max_retries': 3,
        'self_repair_attempts': 2,
        'default_difficulty': 3
    }
    
    # CEFR assessment parameters - Individual fields for environment loading
    cefr_max_retries: int = 3
    cefr_strict_validation: bool = True
    cefr_confidence_threshold: float = 0.7
    
    # CEFR assessment parameters
    cefr_assessment_params: Dict[str, Any] = {
        'max_retries': 3,
        'strict_validation': True,
        'confidence_threshold': 0.7
    }
    
    # Question Generation parameters - Individual fields for environment loading
    use_template_questions: bool = True
    use_gpt_questions: bool = True
    template_fallback: bool = True
    max_gpt_questions_per_day: int = 100
    gpt_creativity: float = 0.7
    question_diversity_threshold: float = 0.8
    
    # Question Generation parameters
    question_generation_params: Dict[str, Any] = {
        'use_templates': True,
        'use_gpt': True,
        'template_fallback': True,
        'max_gpt_questions_per_day': 100,
        'gpt_creativity': 0.7,
        'question_diversity_threshold': 0.8
    }
    
    # LanguageTool integration
    use_languagetool: bool = False
    languagetool_url: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        env_file_encoding = 'utf-8'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_api_keys_from_env()
        self._validate_environment_config()
        self._generate_secure_keys_if_needed()
        print(f"✅ Configuration loaded successfully - Environment: {self.environment.value}")
        print(f"   Database: {self.database_url}")
        print(f"   Redis: {self.redis_url}")
        print(f"   LLM Providers: {self.llm_providers_available}")

    def _load_api_keys_from_env(self):
        """Load API keys and configuration from environment variables securely"""
        # Load database configuration
        if os.getenv("DATABASE_URL"):
            self.database_url = os.getenv("DATABASE_URL")
        
        if os.getenv("TEST_DATABASE_URL"):
            self.test_database_url = os.getenv("TEST_DATABASE_URL")
        
        # Load Redis configuration
        if os.getenv("REDIS_URL"):
            self.redis_url = os.getenv("REDIS_URL")
        
        # Load OpenAI API key
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Load Anthropic API key
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Load other configuration from environment
        if os.getenv("ELO_K_FACTOR"):
            self.elo_k_factor = float(os.getenv("ELO_K_FACTOR"))
        
        if os.getenv("ELO_TAU_FACTOR"):
            self.elo_tau_factor = float(os.getenv("ELO_TAU_FACTOR"))
        
        # Load math recommendation weights
        if os.getenv("MATH_WEIGHT_DIFFICULTY_FIT"):
            self.math_weight_difficulty_fit = float(os.getenv("MATH_WEIGHT_DIFFICULTY_FIT"))
            self.math_recommendation_weights['difficulty_fit'] = self.math_weight_difficulty_fit
        if os.getenv("MATH_WEIGHT_NEIGHBOR_WRONGS"):
            self.math_weight_neighbor_wrongs = float(os.getenv("MATH_WEIGHT_NEIGHBOR_WRONGS"))
            self.math_recommendation_weights['neighbor_wrongs'] = self.math_weight_neighbor_wrongs
        if os.getenv("MATH_WEIGHT_DIVERSITY"):
            self.math_weight_diversity = float(os.getenv("MATH_WEIGHT_DIVERSITY"))
            self.math_recommendation_weights['diversity'] = self.math_weight_diversity
        
        # Load cloze generation parameters
        if os.getenv("CLOZE_MAX_RETRIES"):
            self.cloze_max_retries = int(os.getenv("CLOZE_MAX_RETRIES"))
            self.cloze_generation_params['max_retries'] = self.cloze_max_retries
        if os.getenv("CLOZE_SELF_REPAIR_ATTEMPTS"):
            self.cloze_self_repair_attempts = int(os.getenv("CLOZE_SELF_REPAIR_ATTEMPTS"))
            self.cloze_generation_params['self_repair_attempts'] = self.cloze_self_repair_attempts
        if os.getenv("CLOZE_DEFAULT_DIFFICULTY"):
            self.cloze_default_difficulty = int(os.getenv("CLOZE_DEFAULT_DIFFICULTY"))
            self.cloze_generation_params['default_difficulty'] = self.cloze_default_difficulty
        
        # Load CEFR assessment parameters
        if os.getenv("CEFR_MAX_RETRIES"):
            self.cefr_max_retries = int(os.getenv("CEFR_MAX_RETRIES"))
            self.cefr_assessment_params['max_retries'] = self.cefr_max_retries
        if os.getenv("CEFR_STRICT_VALIDATION"):
            self.cefr_strict_validation = os.getenv("CEFR_STRICT_VALIDATION").lower() == 'true'
            self.cefr_assessment_params['strict_validation'] = self.cefr_strict_validation
        if os.getenv("CEFR_CONFIDENCE_THRESHOLD"):
            self.cefr_confidence_threshold = float(os.getenv("CEFR_CONFIDENCE_THRESHOLD"))
            self.cefr_assessment_params['confidence_threshold'] = self.cefr_confidence_threshold
        
        # Load question generation parameters
        if os.getenv("USE_TEMPLATE_QUESTIONS"):
            self.use_template_questions = os.getenv("USE_TEMPLATE_QUESTIONS").lower() == 'true'
            self.question_generation_params['use_templates'] = self.use_template_questions
        if os.getenv("USE_GPT_QUESTIONS"):
            self.use_gpt_questions = os.getenv("USE_GPT_QUESTIONS").lower() == 'true'
            self.question_generation_params['use_gpt'] = self.use_gpt_questions
        if os.getenv("TEMPLATE_FALLBACK"):
            self.template_fallback = os.getenv("TEMPLATE_FALLBACK").lower() == 'true'
            self.question_generation_params['template_fallback'] = self.template_fallback
        if os.getenv("MAX_GPT_QUESTIONS_PER_DAY"):
            self.max_gpt_questions_per_day = int(os.getenv("MAX_GPT_QUESTIONS_PER_DAY"))
            self.question_generation_params['max_gpt_questions_per_day'] = self.max_gpt_questions_per_day
        if os.getenv("GPT_CREATIVITY"):
            self.gpt_creativity = float(os.getenv("GPT_CREATIVITY"))
            self.question_generation_params['gpt_creativity'] = self.gpt_creativity
        if os.getenv("QUESTION_DIVERSITY_THRESHOLD"):
            self.question_diversity_threshold = float(os.getenv("QUESTION_DIVERSITY_THRESHOLD"))
            self.question_generation_params['question_diversity_threshold'] = self.question_diversity_threshold
        
        # Load LanguageTool settings
        if os.getenv("USE_LANGUAGETOOL"):
            self.use_languagetool = os.getenv("USE_LANGUAGETOOL").lower() == 'true'
        if os.getenv("LANGUAGETOOL_URL"):
            self.languagetool_url = os.getenv("LANGUAGETOOL_URL")
        
        # Load other environment variables
        if os.getenv("ENVIRONMENT"):
            env_value = os.getenv("ENVIRONMENT").lower()
            if env_value in ["development", "testing", "production"]:
                self.environment = Environment(env_value)
        
        if os.getenv("DEBUG"):
            self.debug = os.getenv("DEBUG").lower() == 'true'

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
        
        # Redis URL validation - More flexible for development
        if not self.redis_url:
            if self.environment == Environment.PRODUCTION:
                critical_errors.append("REDIS_URL must be configured in production")
            else:
                # Auto-generate default Redis URL for development
                self.redis_url = "redis://localhost:6379/0"
                print("⚠️ Warning: Using default Redis URL for development")
                print("   You can set REDIS_URL in .env file to customize")
        
        # Check if Redis is actually accessible (only in production)
        if self.environment == Environment.PRODUCTION and self.redis_url:
            try:
                import redis
                r = redis.from_url(self.redis_url, socket_connect_timeout=5)
                r.ping()
                r.close()
            except Exception as e:
                critical_errors.append(f"Redis connection failed: {str(e)}")
        elif self.environment != Environment.PRODUCTION:
            # For development, just check if Redis package is available
            try:
                import redis
                print("✅ Redis package available for development")
            except ImportError:
                print("⚠️ Redis package not installed - some features may not work")
        
        # LLM API Key validation (at least one required)
        if not self.openai_api_key and not self.anthropic_api_key:
            if self.environment == Environment.PRODUCTION:
                critical_errors.append("At least one LLM API key (OPENAI_API_KEY or ANTHROPIC_API_KEY) must be configured")
            else:
                print("⚠️ Warning: No LLM API keys configured - some features may not work")
                print("   You can set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env file")
        
        if critical_errors:
            error_message = "Critical configuration errors:\n" + "\n".join(f"- {error}" for error in critical_errors)
            print("❌ Configuration validation failed:")
            print(error_message)
            if self.environment != Environment.PRODUCTION:
                print("💡 For development, you can:")
                print("   1. Set REDIS_URL in .env file")
                print("   2. Install Redis: sudo apt-get install redis-server")
                print("   3. Or use Docker: docker run -d -p 6379:6379 redis:alpine")
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
            "log_level": self.llm_log_level,
            "log_format": self.log_format,
            "log_file": self.log_file
        }

    @property
    def llm_log_level(self) -> str:
        """Return log level for LLM operations"""
        return self.log_level.value if hasattr(self.log_level, 'value') else str(self.log_level)


settings = Settings()