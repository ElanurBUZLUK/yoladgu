#!/usr/bin/env python3
"""
Test script for Task 10: API Schema ve Validation Sistemi
Tests Pydantic schema models and API middleware/error handling
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.schemas import (
    # User schemas
    UserBase, UserCreate, UserUpdate, UserResponse, UserLogin, UserLoginResponse,
    UserRefreshToken, UserPasswordChange, UserPasswordReset, UserPasswordResetConfirm,
    UserLevelUpdate, UserSearchQuery, UserStats, UserSummary, UserBulkOperation,
    UserCountByRole, UserStatsSummary, UserValidationResult, UserProfile,
    UserActivity, UserAchievement, UserPreferences, UserNotification,
    
    # Error schemas
    ErrorResponse, ValidationErrorResponse, AuthenticationErrorResponse,
    AuthorizationErrorResponse, NotFoundErrorResponse, ConflictErrorResponse,
    RateLimitErrorResponse, InternalErrorResponse, ExternalServiceErrorResponse,
    DatabaseErrorResponse, FileUploadErrorResponse, LLMErrorResponse, MCPErrorResponse,
    ErrorType, ErrorSeverity, ErrorDetail, ErrorLog, ErrorSummary, ErrorMetrics,
    ErrorAlert, ErrorRecoveryAction, ErrorPreventionRule,
    
    # API documentation schemas
    ApiDocumentation, ApiEndpoint, ApiGroup, ApiVersion, ApiExample,
    ApiTutorial, ApiChangelog, ApiMetric, ApiHealthCheck, ApiRateLimit,
    ApiUsage, ApiSchema, ApiErrorCode, ApiSDK
)

from app.middleware.error_handler import error_handler
from app.middleware.rate_limiter import rate_limiter
from app.middleware.logging import logging_middleware

from app.models.user import UserRole, LearningStyle


def test_user_schemas():
    """Test user schema models"""
    print("Testing User Schemas...")
    
    # Test UserBase
    user_base = UserBase(
        username="testuser",
        email="test@example.com",
        role=UserRole.STUDENT,
        current_math_level=3,
        current_english_level=2,
        learning_style=LearningStyle.VISUAL
    )
    assert user_base.username == "testuser"
    assert user_base.email == "test@example.com"
    assert user_base.role == UserRole.STUDENT
    print("✅ UserBase schema test passed")
    
    # Test UserCreate
    user_create = UserCreate(
        username="newuser",
        email="new@example.com",
        password="password123",
        confirm_password="password123"
    )
    assert user_create.password == "password123"
    assert user_create.confirm_password == "password123"
    print("✅ UserCreate schema test passed")
    
    # Test UserUpdate
    user_update = UserUpdate(
        current_math_level=4,
        learning_style=LearningStyle.AUDITORY
    )
    assert user_update.current_math_level == 4
    assert user_update.learning_style == LearningStyle.AUDITORY
    print("✅ UserUpdate schema test passed")
    
    # Test UserLogin
    user_login = UserLogin(
        email="login@example.com",
        password="password123"
    )
    assert user_login.email == "login@example.com"
    print("✅ UserLogin schema test passed")
    
    # Test UserStats
    user_stats = UserStats(
        total_attempts=100,
        correct_attempts=75,
        accuracy_rate=0.75,
        avg_difficulty=3.2,
        avg_time_spent=45.5,
        math_level=3,
        english_level=2,
        learning_style="visual"
    )
    assert user_stats.accuracy_rate == 0.75
    assert user_stats.total_attempts == 100
    print("✅ UserStats schema test passed")
    
    # Test UserSearchQuery
    search_query = UserSearchQuery(
        query="test",
        role=UserRole.TEACHER,
        min_math_level=2,
        max_math_level=4
    )
    assert search_query.query == "test"
    assert search_query.role == UserRole.TEACHER
    print("✅ UserSearchQuery schema test passed")
    
    print("✅ All User Schema tests passed!")


def test_error_schemas():
    """Test error schema models"""
    print("Testing Error Schemas...")
    
    # Test ErrorResponse
    error_response = ErrorResponse(
        error_type=ErrorType.VALIDATION_ERROR,
        message="Validation failed",
        severity=ErrorSeverity.LOW
    )
    assert error_response.error_type == ErrorType.VALIDATION_ERROR
    assert error_response.severity == ErrorSeverity.LOW
    print("✅ ErrorResponse schema test passed")
    
    # Test ValidationErrorResponse
    validation_error = ValidationErrorResponse(
        message="Field validation failed",
        details=[
            ErrorDetail(
                field="email",
                message="Invalid email format",
                code="value_error.email"
            )
        ]
    )
    assert validation_error.error_type == ErrorType.VALIDATION_ERROR
    assert len(validation_error.details) == 1
    print("✅ ValidationErrorResponse schema test passed")
    
    # Test AuthenticationErrorResponse
    auth_error = AuthenticationErrorResponse(
        message="Authentication required",
        error_code="AUTH_REQUIRED"
    )
    assert auth_error.error_type == ErrorType.AUTHENTICATION_ERROR
    assert auth_error.severity == ErrorSeverity.HIGH
    print("✅ AuthenticationErrorResponse schema test passed")
    
    # Test RateLimitErrorResponse
    rate_limit_error = RateLimitErrorResponse(
        message="Rate limit exceeded",
        retry_after=60,
        rate_limit_info={
            "limit": 100,
            "remaining": 0,
            "reset_time": 1234567890
        }
    )
    assert rate_limit_error.retry_after == 60
    assert rate_limit_error.rate_limit_info["limit"] == 100
    print("✅ RateLimitErrorResponse schema test passed")
    
    # Test ErrorLog
    error_log = ErrorLog(
        error_id="err_123",
        error_type=ErrorType.INTERNAL_ERROR,
        message="Internal server error",
        severity=ErrorSeverity.CRITICAL,
        timestamp="2024-01-01T00:00:00Z"
    )
    assert error_log.error_id == "err_123"
    assert error_log.severity == ErrorSeverity.CRITICAL
    print("✅ ErrorLog schema test passed")
    
    print("✅ All Error Schema tests passed!")


def test_api_docs_schemas():
    """Test API documentation schema models"""
    print("Testing API Documentation Schemas...")
    
    # Test ApiEndpoint
    api_endpoint = ApiEndpoint(
        path="/api/v1/users",
        method="GET",
        summary="Get users",
        description="Retrieve list of users",
        category="users",
        security="admin"
    )
    assert api_endpoint.path == "/api/v1/users"
    assert api_endpoint.method == "GET"
    print("✅ ApiEndpoint schema test passed")
    
    # Test ApiGroup
    api_group = ApiGroup(
        name="User Management",
        description="User management endpoints",
        endpoints=[api_endpoint]
    )
    assert api_group.name == "User Management"
    assert len(api_group.endpoints) == 1
    print("✅ ApiGroup schema test passed")
    
    # Test ApiDocumentation
    api_docs = ApiDocumentation(
        title="Test API",
        version="1.0.0",
        description="Test API documentation",
        versions=[ApiVersion(
            version="1.0.0",
            status="stable",
            release_date="2024-01-01",
            groups=[api_group]
        )]
    )
    assert api_docs.title == "Test API"
    assert api_docs.version == "1.0.0"
    print("✅ ApiDocumentation schema test passed")
    
    # Test ApiMetric
    api_metric = ApiMetric(
        endpoint="/api/v1/users",
        method="GET",
        total_requests=1000,
        successful_requests=950,
        failed_requests=50,
        average_response_time=0.5,
        p95_response_time=1.2,
        p99_response_time=2.0,
        error_rate=0.05,
        time_period="1h"
    )
    assert api_metric.total_requests == 1000
    assert api_metric.error_rate == 0.05
    print("✅ ApiMetric schema test passed")
    
    print("✅ All API Documentation Schema tests passed!")


def test_error_handler_middleware():
    """Test error handler middleware"""
    print("Testing Error Handler Middleware...")
    
    # Test error handler initialization
    assert error_handler is not None
    assert hasattr(error_handler, 'error_logs')
    assert isinstance(error_handler.error_logs, list)
    print("✅ Error handler initialization test passed")
    
    # Test HTTP status code mapping
    status_code = error_handler.get_http_status_code(ErrorType.VALIDATION_ERROR)
    assert status_code == 400
    
    status_code = error_handler.get_http_status_code(ErrorType.AUTHENTICATION_ERROR)
    assert status_code == 401
    
    status_code = error_handler.get_http_status_code(ErrorType.RATE_LIMIT_ERROR)
    assert status_code == 429
    print("✅ HTTP status code mapping test passed")
    
    # Test log level mapping
    log_level = error_handler.get_log_level(ErrorSeverity.LOW)
    assert log_level == "INFO"
    
    log_level = error_handler.get_log_level(ErrorSeverity.HIGH)
    assert log_level == "ERROR"
    print("✅ Log level mapping test passed")
    
    # Test error summary
    summary = error_handler.get_error_summary()
    assert "total_errors" in summary
    assert summary["total_errors"] == 0
    print("✅ Error summary test passed")
    
    print("✅ All Error Handler Middleware tests passed!")


async def test_rate_limiter_middleware():
    """Test rate limiter middleware"""
    print("Testing Rate Limiter Middleware...")
    
    # Test rate limiter initialization
    assert rate_limiter is not None
    assert hasattr(rate_limiter, 'limits')
    assert "default" in rate_limiter.limits
    print("✅ Rate limiter initialization test passed")
    
    # Test limit configurations
    default_limit = rate_limiter.limits["default"]
    assert default_limit["requests"] == 100
    assert default_limit["window"] == 60
    
    auth_limit = rate_limiter.limits["auth"]
    assert auth_limit["requests"] == 10
    print("✅ Rate limit configurations test passed")
    
    # Test rate limit status
    status = await rate_limiter.get_rate_limit_status("test_user")
    assert isinstance(status, dict)
    assert "default" in status
    print("✅ Rate limit status test passed")
    
    # Test rate limit config update
    rate_limiter.update_rate_limit_config("test_limit", 50, 30)
    assert "test_limit" in rate_limiter.limits
    assert rate_limiter.limits["test_limit"]["requests"] == 50
    print("✅ Rate limit config update test passed")
    
    print("✅ All Rate Limiter Middleware tests passed!")


async def test_logging_middleware():
    """Test logging middleware"""
    print("Testing Logging Middleware...")
    
    # Test logging middleware initialization
    assert logging_middleware is not None
    assert hasattr(logging_middleware, 'logger')
    assert hasattr(logging_middleware, 'sensitive_paths')
    print("✅ Logging middleware initialization test passed")
    
    # Test sensitive path filtering
    assert "/api/v1/auth/login" in logging_middleware.sensitive_paths
    assert "/api/v1/auth/register" in logging_middleware.sensitive_paths
    print("✅ Sensitive paths test passed")
    
    # Test sensitive header filtering
    assert "authorization" in logging_middleware.sensitive_headers
    assert "cookie" in logging_middleware.sensitive_headers
    print("✅ Sensitive headers test passed")
    
    # Test log level determination
    # Mock request object for testing
    class MockRequest:
        def __init__(self, path):
            self.url = type('obj', (object,), {'path': path})()
    
    log_level = logging_middleware.get_request_log_level(MockRequest("/health"))
    assert log_level == "DEBUG"
    
    log_level = logging_middleware.get_request_log_level(MockRequest("/api/v1/auth/login"))
    assert log_level == "WARNING"
    
    log_level = logging_middleware.get_request_log_level(MockRequest("/api/v1/users"))
    assert log_level == "INFO"
    
    log_level = logging_middleware.get_response_log_level(200)
    assert log_level == "INFO"
    
    log_level = logging_middleware.get_response_log_level(404)
    assert log_level == "WARNING"
    
    log_level = logging_middleware.get_response_log_level(500)
    assert log_level == "ERROR"
    print("✅ Log level determination test passed")
    
    # Test log summary
    summary = await logging_middleware.get_log_summary()
    assert isinstance(summary, dict)
    assert "total_requests" in summary
    print("✅ Log summary test passed")
    
    print("✅ All Logging Middleware tests passed!")


def test_schema_validation():
    """Test schema validation rules"""
    print("Testing Schema Validation Rules...")
    
    # Test user validation
    try:
        # Invalid email format
        UserCreate(
            username="test",
            email="invalid-email",
            password="password123",
            confirm_password="password123"
        )
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "email" in str(e).lower()
        print("✅ Email validation test passed")
    
    try:
        # Username too short
        UserCreate(
            username="ab",  # Too short
            email="test@example.com",
            password="password123",
            confirm_password="password123"
        )
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "username" in str(e).lower()
        print("✅ Username length validation test passed")
    
    try:
        # Password too short
        UserCreate(
            username="testuser",
            email="test@example.com",
            password="123",  # Too short
            confirm_password="123"
        )
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "password" in str(e).lower()
        print("✅ Password length validation test passed")
    
    try:
        # Level out of range
        UserBase(
            username="test",
            email="test@example.com",
            current_math_level=10,  # Out of range (1-5)
            current_english_level=1
        )
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "level" in str(e).lower() or "ge" in str(e).lower()
        print("✅ Level range validation test passed")
    
    print("✅ All Schema Validation tests passed!")


def test_schema_serialization():
    """Test schema serialization"""
    print("Testing Schema Serialization...")
    
    # Test user response serialization
    user_response = UserResponse(
        id="user_123",
        username="testuser",
        email="test@example.com",
        role=UserRole.STUDENT,
        current_math_level=3,
        current_english_level=2,
        learning_style=LearningStyle.VISUAL,
        is_active="true",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z"
    )
    
    # Convert to dict
    user_dict = user_response.dict()
    assert user_dict["id"] == "user_123"
    assert user_dict["username"] == "testuser"
    assert user_dict["role"] == "student"
    print("✅ User response serialization test passed")
    
    # Test error response serialization
    error_response = ErrorResponse(
        error_type=ErrorType.VALIDATION_ERROR,
        message="Test error",
        severity=ErrorSeverity.MEDIUM
    )
    
    error_dict = error_response.dict()
    assert error_dict["error_type"] == "validation_error"
    assert error_dict["message"] == "Test error"
    assert error_dict["severity"] == "medium"
    print("✅ Error response serialization test passed")
    
    print("✅ All Schema Serialization tests passed!")


async def main():
    """Main test function"""
    print("🚀 Starting Task 10 Tests: API Schema ve Validation Sistemi")
    print("=" * 60)
    
    try:
        # Test user schemas
        test_user_schemas()
        print()
        
        # Test error schemas
        test_error_schemas()
        print()
        
        # Test API documentation schemas
        test_api_docs_schemas()
        print()
        
        # Test error handler middleware
        test_error_handler_middleware()
        print()
        
        # Test rate limiter middleware
        await test_rate_limiter_middleware()
        print()
        
        # Test logging middleware
        await test_logging_middleware()
        print()
        
        # Test schema validation
        test_schema_validation()
        print()
        
        # Test schema serialization
        test_schema_serialization()
        print()
        
        print("🎉 All Task 10 tests passed successfully!")
        print("✅ Pydantic schema models implemented")
        print("✅ API middleware and error handling implemented")
        print("✅ Validation rules working correctly")
        print("✅ Error handling middleware functional")
        print("✅ Rate limiting middleware functional")
        print("✅ Logging middleware functional")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
