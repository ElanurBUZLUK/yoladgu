"""
Specific Exception Handling
Özel exception sınıfları ve exception handler - genel exception'ları özel hata türleriyle değiştirmek için
"""

import time
from typing import Any, Dict, Optional, List
from enum import Enum

import structlog
from fastapi import HTTPException

logger = structlog.get_logger()

# ============================================================================
# EXCEPTION TYPES
# ============================================================================

class ErrorType(Enum):
    """Hata türleri"""
    DATABASE_ERROR = "database_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    CACHE_ERROR = "cache_error"
    VECTOR_SEARCH_ERROR = "vector_search_error"
    EMBEDDING_ERROR = "embedding_error"
    RECOMMENDATION_ERROR = "recommendation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    VALIDATION_ERROR = "validation_error"
    CONFIGURATION_ERROR = "configuration_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    FALLBACK_ERROR = "fallback_error"

class ErrorSeverity(Enum):
    """Hata şiddeti"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class BaseAppException(Exception):
    """Temel uygulama exception'ı"""
    
    def __init__(self, message: str, error_type: ErrorType, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Dict[str, Any] = None, retryable: bool = False):
        self.message = message
        self.error_type = error_type
        self.severity = severity
        self.context = context or {}
        self.retryable = retryable
        self.timestamp = time.time()
        super().__init__(self.message)

class DatabaseError(BaseAppException):
    """Veritabanı hatası"""
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorType.DATABASE_ERROR, ErrorSeverity.HIGH, context, retryable=True)

class NetworkError(BaseAppException):
    """Ağ hatası"""
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorType.NETWORK_ERROR, ErrorSeverity.MEDIUM, context, retryable=True)

class TimeoutError(BaseAppException):
    """Zaman aşımı hatası"""
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorType.TIMEOUT_ERROR, ErrorSeverity.MEDIUM, context, retryable=True)

class CacheError(BaseAppException):
    """Cache hatası"""
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorType.CACHE_ERROR, ErrorSeverity.LOW, context, retryable=True)

class VectorSearchError(BaseAppException):
    """Vector search hatası"""
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorType.VECTOR_SEARCH_ERROR, ErrorSeverity.MEDIUM, context, retryable=True)

class EmbeddingError(BaseAppException):
    """Embedding hatası"""
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorType.EMBEDDING_ERROR, ErrorSeverity.MEDIUM, context, retryable=True)

class RecommendationError(BaseAppException):
    """Öneri hatası"""
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorType.RECOMMENDATION_ERROR, ErrorSeverity.MEDIUM, context, retryable=True)

class AuthenticationError(BaseAppException):
    """Kimlik doğrulama hatası"""
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorType.AUTHENTICATION_ERROR, ErrorSeverity.HIGH, context, retryable=False)

class ValidationError(BaseAppException):
    """Doğrulama hatası"""
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorType.VALIDATION_ERROR, ErrorSeverity.LOW, context, retryable=False)

class ConfigurationError(BaseAppException):
    """Konfigürasyon hatası"""
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorType.CONFIGURATION_ERROR, ErrorSeverity.HIGH, context, retryable=False)

class RateLimitError(BaseAppException):
    """Rate limit hatası"""
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorType.RATE_LIMIT_ERROR, ErrorSeverity.MEDIUM, context, retryable=True)

class FallbackError(BaseAppException):
    """Fallback hatası"""
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorType.FALLBACK_ERROR, ErrorSeverity.CRITICAL, context, retryable=False)

# ============================================================================
# EXCEPTION HANDLER
# ============================================================================

class SpecificExceptionHandler:
    """Özel exception handler"""
    
    def __init__(self):
        self.error_counts: Dict[ErrorType, int] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
    
    def handle_exception(self, exception: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Exception'ı işle ve uygun response döndür"""
        try:
            # Exception türünü belirle
            error_info = self._classify_exception(exception)
            
            # Context'i güncelle
            if context:
                error_info['context'].update(context)
            
            # Metrikleri güncelle
            self._update_metrics(error_info)
            
            # Log'la
            self._log_exception(error_info)
            
            # HTTP response oluştur
            return self._create_response(error_info)
            
        except Exception as e:
            logger.error("exception_handler_error", error=str(e))
            return self._create_fallback_response(str(exception))
    
    def _classify_exception(self, exception: Exception) -> Dict[str, Any]:
        """Exception'ı sınıflandır"""
        if isinstance(exception, BaseAppException):
            return {
                'error_type': exception.error_type,
                'severity': exception.severity,
                'message': exception.message,
                'context': exception.context,
                'retryable': exception.retryable,
                'timestamp': exception.timestamp
            }
        
        # Genel exception'ları özel türlere eşle
        exception_type = type(exception).__name__
        error_message = str(exception)
        
        if "database" in error_message.lower() or "sql" in error_message.lower():
            return {
                'error_type': ErrorType.DATABASE_ERROR,
                'severity': ErrorSeverity.HIGH,
                'message': error_message,
                'context': {'exception_type': exception_type},
                'retryable': True,
                'timestamp': time.time()
            }
        elif "timeout" in error_message.lower() or "timed out" in error_message.lower():
            return {
                'error_type': ErrorType.TIMEOUT_ERROR,
                'severity': ErrorSeverity.MEDIUM,
                'message': error_message,
                'context': {'exception_type': exception_type},
                'retryable': True,
                'timestamp': time.time()
            }
        elif "network" in error_message.lower() or "connection" in error_message.lower():
            return {
                'error_type': ErrorType.NETWORK_ERROR,
                'severity': ErrorSeverity.MEDIUM,
                'message': error_message,
                'context': {'exception_type': exception_type},
                'retryable': True,
                'timestamp': time.time()
            }
        elif "cache" in error_message.lower() or "redis" in error_message.lower():
            return {
                'error_type': ErrorType.CACHE_ERROR,
                'severity': ErrorSeverity.LOW,
                'message': error_message,
                'context': {'exception_type': exception_type},
                'retryable': True,
                'timestamp': time.time()
            }
        elif "vector" in error_message.lower() or "embedding" in error_message.lower():
            return {
                'error_type': ErrorType.VECTOR_SEARCH_ERROR,
                'severity': ErrorSeverity.MEDIUM,
                'message': error_message,
                'context': {'exception_type': exception_type},
                'retryable': True,
                'timestamp': time.time()
            }
        else:
            return {
                'error_type': ErrorType.RECOMMENDATION_ERROR,
                'severity': ErrorSeverity.MEDIUM,
                'message': error_message,
                'context': {'exception_type': exception_type},
                'retryable': False,
                'timestamp': time.time()
            }
    
    def _update_metrics(self, error_info: Dict[str, Any]):
        """Hata metriklerini güncelle"""
        error_type = error_info['error_type']
        
        # Hata sayısını artır
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Hata geçmişine ekle
        self.error_history.append(error_info)
        
        # Geçmiş boyutunu kontrol et
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)
    
    def _log_exception(self, error_info: Dict[str, Any]):
        """Exception'ı log'la"""
        log_level = "error" if error_info['severity'] in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else "warning"
        
        log_data = {
            'error_type': error_info['error_type'].value,
            'severity': error_info['severity'].value,
            'message': error_info['message'],
            'retryable': error_info['retryable'],
            'context': error_info['context']
        }
        
        if log_level == "error":
            logger.error("specific_exception", **log_data)
        else:
            logger.warning("specific_exception", **log_data)
    
    def _create_response(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """HTTP response oluştur"""
        status_code = self._get_status_code(error_info['error_type'], error_info['severity'])
        
        return {
            'error': {
                'type': error_info['error_type'].value,
                'message': error_info['message'],
                'severity': error_info['severity'].value,
                'retryable': error_info['retryable'],
                'timestamp': error_info['timestamp']
            },
            'status_code': status_code,
            'context': error_info['context']
        }
    
    def _create_fallback_response(self, message: str) -> Dict[str, Any]:
        """Fallback response oluştur"""
        return {
            'error': {
                'type': 'unknown_error',
                'message': message,
                'severity': 'high',
                'retryable': False,
                'timestamp': time.time()
            },
            'status_code': 500,
            'context': {}
        }
    
    def _get_status_code(self, error_type: ErrorType, severity: ErrorSeverity) -> int:
        """Hata türüne göre status code belirle"""
        if error_type == ErrorType.AUTHENTICATION_ERROR:
            return 401
        elif error_type == ErrorType.VALIDATION_ERROR:
            return 400
        elif error_type == ErrorType.RATE_LIMIT_ERROR:
            return 429
        elif error_type == ErrorType.TIMEOUT_ERROR:
            return 504
        elif error_type == ErrorType.DATABASE_ERROR:
            return 503
        elif error_type == ErrorType.NETWORK_ERROR:
            return 502
        elif severity == ErrorSeverity.CRITICAL:
            return 500
        elif severity == ErrorSeverity.HIGH:
            return 500
        else:
            return 500
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Hata istatistiklerini al"""
        return {
            'error_counts': {k.value: v for k, v in self.error_counts.items()},
            'total_errors': sum(self.error_counts.values()),
            'recent_errors': self.error_history[-10:] if self.error_history else [],
            'error_types': list(self.error_counts.keys())
        }
    
    def clear_history(self):
        """Hata geçmişini temizle"""
        self.error_history.clear()
        self.error_counts.clear()

# Global exception handler instance
exception_handler = SpecificExceptionHandler()

# ============================================================================
# EXCEPTION DECORATORS
# ============================================================================

def handle_specific_exceptions(func):
    """Fonksiyonu specific exception handling ile saran decorator"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_response = exception_handler.handle_exception(e, {
                'function': func.__name__,
                'args': str(args),
                'kwargs': str(kwargs)
            })
            
            # HTTPException oluştur
            raise HTTPException(
                status_code=error_response['status_code'],
                detail=error_response['error']
            )
    return wrapper

def with_fallback(fallback_func):
    """Fallback fonksiyonu ile saran decorator"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning("fallback_triggered", 
                             function=func.__name__, 
                             error=str(e))
                
                # Fallback fonksiyonunu çağır
                return await fallback_func(*args, **kwargs)
        return wrapper
    return decorator

# ============================================================================
# EXCEPTION HELPERS
# ============================================================================

def raise_database_error(message: str, context: Dict[str, Any] = None):
    """Database hatası fırlat"""
    raise DatabaseError(message, context)

def raise_network_error(message: str, context: Dict[str, Any] = None):
    """Network hatası fırlat"""
    raise NetworkError(message, context)

def raise_timeout_error(message: str, context: Dict[str, Any] = None):
    """Timeout hatası fırlat"""
    raise TimeoutError(message, context)

def raise_cache_error(message: str, context: Dict[str, Any] = None):
    """Cache hatası fırlat"""
    raise CacheError(message, context)

def raise_vector_search_error(message: str, context: Dict[str, Any] = None):
    """Vector search hatası fırlat"""
    raise VectorSearchError(message, context)

def raise_embedding_error(message: str, context: Dict[str, Any] = None):
    """Embedding hatası fırlat"""
    raise EmbeddingError(message, context)

def raise_recommendation_error(message: str, context: Dict[str, Any] = None):
    """Recommendation hatası fırlat"""
    raise RecommendationError(message, context)

def raise_authentication_error(message: str, context: Dict[str, Any] = None):
    """Authentication hatası fırlat"""
    raise AuthenticationError(message, context)

def raise_validation_error(message: str, context: Dict[str, Any] = None):
    """Validation hatası fırlat"""
    raise ValidationError(message, context)

def raise_configuration_error(message: str, context: Dict[str, Any] = None):
    """Configuration hatası fırlat"""
    raise ConfigurationError(message, context)

def raise_rate_limit_error(message: str, context: Dict[str, Any] = None):
    """Rate limit hatası fırlat"""
    raise RateLimitError(message, context)

def raise_fallback_error(message: str, context: Dict[str, Any] = None):
    """Fallback hatası fırlat"""
    raise FallbackError(message, context)
