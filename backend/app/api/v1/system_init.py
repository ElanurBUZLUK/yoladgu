from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any
import logging

from app.core.config import settings
from app.core.security import security_service
from app.services.system_initialization_service import system_initialization_service
from app.services.sample_data_service import sample_data_service
from app.schemas.user import UserResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/system",
    tags=["System Initialization"]
)


@router.post("/initialize", response_model=Dict[str, Any])
async def initialize_system(
    background_tasks: BackgroundTasks
):
    """
    Initialize the complete system
    
    This endpoint performs:
    - Environment configuration validation
    - Database initialization and schema verification
    - Cache system initialization
    - LLM provider configuration testing
    - MCP server startup verification
    - Sample data initialization (development only)
    """
    try:
        logger.info("System initialization requested")
        
        # Run initialization
        result = await system_initialization_service.initialize_system()
        
        if result["success"]:
            logger.info("System initialization completed successfully")
            return {
                "message": "System initialization completed successfully",
                "result": result
            }
        else:
            logger.error(f"System initialization failed: {result['errors']}")
            return {
                "message": "System initialization completed with errors",
                "result": result
            }
            
    except Exception as e:
        logger.error(f"System initialization failed with exception: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"System initialization failed: {str(e)}"
        )


@router.get("/status", response_model=Dict[str, Any])
async def get_system_status():
    """
    Get current system initialization status
    """
    try:
        status = await system_initialization_service.get_system_status()
        return {
            "message": "System status retrieved successfully",
            "status": status
        }
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system status: {str(e)}"
        )


@router.get("/init", response_model=Dict[str, Any])
async def initialize_pgvector(apply: bool = False):
    """
    Initialize PgVector extension and indexes
    
    Args:
        apply: If True, create missing extensions/columns/indexes
    """
    try:
        from app.core.database import get_async_session
        from sqlalchemy.ext.asyncio import AsyncSession
        
        async for db in get_async_session():
            result = await system_initialization_service.run_all(db, apply=apply)
            break
        
        return {
            "message": "PgVector initialization completed",
            "apply": apply,
            "result": result
        }
    except Exception as e:
        logger.error(f"PgVector initialization failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"PgVector initialization failed: {str(e)}"
        )


@router.post("/sample", response_model=Dict[str, Any])
async def create_sample_data(apply: bool = False):
    """
    Create sample data for development/testing
    
    Args:
        apply: If True, actually create the data in database
    """
    try:
        from app.core.database import get_async_session
        from sqlalchemy.ext.asyncio import AsyncSession
        
        async for db in get_async_session():
            result = await sample_data_service.create_sample_data(db, apply=apply)
            break
        
        return {
            "message": "Sample data creation completed",
            "apply": apply,
            "result": result
        }
    except Exception as e:
        logger.error(f"Sample data creation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Sample data creation failed: {str(e)}"
        )


@router.get("/readyz", response_model=Dict[str, Any])
async def get_readyz_status():
    """
    Get current system readiness status.
    Performs checks for:
    - Database connectivity (SELECT 1)
    - Redis connectivity (PING)
    - Pgvector extension, embedding column, and index existence
    """
    try:
        health_status = await system_initialization_service.validate_system_health()

        if health_status["overall"]:
            return {"ok": True}
        else:
            # Log details of what failed for debugging
            failed_checks = {k: v for k, v in health_status.items() if not v}
            logger.error(f"Readiness check failed. Details: {failed_checks}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"System not ready. Failed checks: {failed_checks}"
            )

    except Exception as e:
        logger.error(f"Readiness check failed with exception: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Readiness check failed: {str(e)}"
        )


@router.post("/validate-environment", response_model=Dict[str, Any])
async def validate_environment_config():
    """
    Validate environment configuration only
    
    This endpoint validates:
    - Required environment variables
    - Database URL format
    - Redis URL format
    - File upload directory
    - Environment-specific settings
    """
    try:
        # Create a temporary service instance for validation only
        from app.services.system_initialization_service import SystemInitializationService
        temp_service = SystemInitializationService()
        
        await temp_service._validate_environment_config()
        
        return {
            "message": "Environment configuration validation completed",
            "status": temp_service.initialization_status["environment"],
            "errors": temp_service.errors,
            "warnings": temp_service.warnings
        }
        
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Environment validation failed: {str(e)}"
        )


@router.post("/test-database", response_model=Dict[str, Any])
async def test_database_connection():
    """
    Test database connection and schema
    
    This endpoint tests:
    - Database connectivity
    - Required table existence
    - Read/write operations
    - Database version
    """
    try:
        from app.services.system_initialization_service import SystemInitializationService
        temp_service = SystemInitializationService()
        
        await temp_service._initialize_database()
        
        return {
            "message": "Database connection test completed",
            "status": temp_service.initialization_status["database"],
            "errors": temp_service.errors,
            "warnings": temp_service.warnings
        }
        
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Database test failed: {str(e)}"
        )


@router.post("/test-cache", response_model=Dict[str, Any])
async def test_cache_system():
    """
    Test cache system functionality
    
    This endpoint tests:
    - Redis connectivity
    - Set/get operations
    - Delete operations
    - Increment operations
    """
    try:
        from app.services.system_initialization_service import SystemInitializationService
        temp_service = SystemInitializationService()
        
        await temp_service._initialize_cache()
        
        return {
            "message": "Cache system test completed",
            "status": temp_service.initialization_status["cache"],
            "errors": temp_service.errors,
            "warnings": temp_service.warnings
        }
        
    except Exception as e:
        logger.error(f"Cache test failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Cache test failed: {str(e)}"
        )


@router.post("/test-llm", response_model=Dict[str, Any])
async def test_llm_providers():
    """
    Test LLM provider configurations
    
    This endpoint tests:
    - Primary LLM provider connectivity
    - Secondary LLM provider connectivity
    - API key configurations
    - Basic text generation
    """
    try:
        from app.services.system_initialization_service import SystemInitializationService
        temp_service = SystemInitializationService()
        
        await temp_service._test_llm_providers()
        
        return {
            "message": "LLM provider test completed",
            "status": temp_service.initialization_status["llm_providers"],
            "errors": temp_service.errors,
            "warnings": temp_service.warnings
        }
        
    except Exception as e:
        logger.error(f"LLM test failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"LLM test failed: {str(e)}"
        )


@router.post("/test-mcp", response_model=Dict[str, Any])
async def test_mcp_server():
    """
    Test MCP server connectivity and tools
    
    This endpoint tests:
    - MCP server connection
    - Available tools
    - Tool functionality
    """
    try:
        from app.services.system_initialization_service import SystemInitializationService
        temp_service = SystemInitializationService()
        
        await temp_service._verify_mcp_server()
        
        return {
            "message": "MCP server test completed",
            "status": temp_service.initialization_status["mcp_server"],
            "errors": temp_service.errors,
            "warnings": temp_service.warnings
        }
        
    except Exception as e:
        logger.error(f"MCP test failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"MCP test failed: {str(e)}"
        )


@router.get("/config", response_model=Dict[str, Any])
async def get_system_config():
    """
    Get current system configuration (non-sensitive)
    """
    try:
        config = {
            "app_name": settings.app_name,
            "version": settings.version,
            "environment": settings.environment,
            "debug": settings.debug,
            "database_url": settings.database_url.split("@")[-1] if "@" in settings.database_url else "configured",
            "redis_url": settings.redis_url.split("@")[-1] if "@" in settings.redis_url else "configured",
            "cors_origins": settings.cors_origins_list,
            "max_file_size": settings.max_file_size,
            "upload_dir": settings.upload_dir,
            "rate_limit_enabled": settings.rate_limit_enabled,
            "prometheus_enabled": settings.prometheus_enabled,
            "primary_llm_provider": settings.primary_llm_provider,
            "secondary_llm_provider": settings.secondary_llm_provider,
            "enable_llm_fallback": settings.enable_llm_fallback,
            "fallback_to_templates": settings.fallback_to_templates,
            "max_retry_attempts": settings.max_retry_attempts,
            "daily_llm_budget": settings.daily_llm_budget,
            "access_token_expire_minutes": settings.access_token_expire_minutes,
            "openai_api_key_configured": bool(settings.openai_api_key),
            "anthropic_api_key_configured": bool(settings.anthropic_api_key)
        }
        
        return {
            "message": "System configuration retrieved successfully",
            "config": config
        }
        
    except Exception as e:
        logger.error(f"Failed to get system config: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system config: {str(e)}"
        )
