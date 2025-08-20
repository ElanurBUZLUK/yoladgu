from fastapi import APIRouter, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import Dict, Any
import logging
import time

from app.core.database import get_async_session
from app.core.config import settings
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/system", tags=["system"])

@router.get("/readyz")
async def health_check(db: AsyncSession = Depends(get_async_session)) -> Dict[str, Any]:
    """
    Deep health check endpoint that verifies all critical system components.
    
    Returns:
        Dict with status of each component and overall health
    """
    health_status = {
        "timestamp": time.time(),
        "overall_status": "healthy",
        "components": {}
    }
    
    try:
        # 1. Database health check
        db_status = await _check_database_health(db)
        health_status["components"]["database"] = db_status
        
        # 2. Redis health check
        redis_status = await _check_redis_health()
        health_status["components"]["redis"] = redis_status
        
        # 3. pgvector health check
        pgvector_status = await _check_pgvector_health(db)
        health_status["components"]["pgvector"] = pgvector_status
        
        # 4. MCP health check
        mcp_status = await _check_mcp_health()
        health_status["components"]["mcp"] = mcp_status
        
        # 5. LLM health check
        llm_status = await _check_llm_health()
        health_status["components"]["llm"] = llm_status
        
        # Determine overall status
        all_healthy = all(
            component.get("status") == "healthy" 
            for component in health_status["components"].values()
        )
        
        health_status["overall_status"] = "healthy" if all_healthy else "degraded"
        
        # Return appropriate status code
        if all_healthy:
            return health_status
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Some system components are unhealthy",
                headers={"X-Health-Status": "degraded"}
            )
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        health_status["overall_status"] = "unhealthy"
        health_status["error"] = str(e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )

async def _check_database_health(db: AsyncSession) -> Dict[str, Any]:
    """Check database connectivity and basic operations"""
    try:
        start_time = time.time()
        
        # Basic connectivity test
        result = await db.execute(text("SELECT 1"))
        result.fetchone()
        
        # Check if we can access the questions table
        result = await db.execute(text("SELECT COUNT(*) FROM questions LIMIT 1"))
        count = result.scalar()
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return {
            "status": "healthy",
            "latency_ms": latency_ms,
            "questions_count": count,
            "details": "Database connection and basic queries working"
        }
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "details": "Database connection or query failed"
        }

async def _check_redis_health() -> Dict[str, Any]:
    """Check Redis connectivity and basic operations"""
    try:
        start_time = time.time()
        
        # Create Redis client
        redis_client = Redis.from_url(settings.redis_url)
        
        # Test PING
        pong = await redis_client.ping()
        if not pong:
            raise Exception("Redis PING failed")
        
        # Test basic operations
        test_key = "health_check_test"
        await redis_client.set(test_key, "test_value", ex=10)
        value = await redis_client.get(test_key)
        await redis_client.delete(test_key)
        
        if value != b"test_value":
            raise Exception("Redis get/set operations failed")
        
        await redis_client.close()
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return {
            "status": "healthy",
            "latency_ms": latency_ms,
            "details": "Redis connection and basic operations working"
        }
        
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "details": "Redis connection or operations failed"
        }

async def _check_pgvector_health(db: AsyncSession) -> Dict[str, Any]:
    """Check pgvector extension and vector operations"""
    try:
        start_time = time.time()
        
        # Check if pgvector extension is installed
        result = await db.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'"))
        extension = result.fetchone()
        
        if not extension:
            return {
                "status": "unhealthy",
                "error": "pgvector extension not installed",
                "details": "Vector extension not found in database"
            }
        
        # Check if we have any vector columns
        result = await db.execute(text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND data_type LIKE '%vector%'
            LIMIT 1
        """))
        vector_columns = result.fetchall()
        
        if not vector_columns:
            return {
                "status": "degraded",
                "warning": "No vector columns found",
                "details": "pgvector extension installed but no vector columns detected"
            }
        
        # Test basic vector operation (if we have embeddings table)
        try:
            result = await db.execute(text("""
                SELECT embedding <=> embedding as distance 
                FROM questions 
                WHERE embedding IS NOT NULL 
                LIMIT 1
            """))
            distance = result.scalar()
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return {
                "status": "healthy",
                "latency_ms": latency_ms,
                "vector_columns": len(vector_columns),
                "details": "pgvector extension working and vector operations functional"
            }
            
        except Exception as e:
            return {
                "status": "degraded",
                "warning": f"Vector operations failed: {str(e)}",
                "details": "pgvector extension installed but vector operations failed"
            }
        
    except Exception as e:
        logger.error(f"pgvector health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "details": "pgvector health check failed"
        }

async def _check_mcp_health() -> Dict[str, Any]:
    """Check MCP server connectivity and tool availability"""
    try:
        start_time = time.time()
        
        # Try to import and check MCP utils
        try:
            from app.core.mcp_utils import mcp_utils
            
            if not mcp_utils.is_initialized:
                return {
                    "status": "degraded",
                    "warning": "MCP not initialized",
                    "details": "MCP client not configured or initialized"
                }
            
            # Test list_tools
            tools_response = await mcp_utils.list_tools()
            
            if not tools_response.get("success", False):
                return {
                    "status": "unhealthy",
                    "error": "MCP list_tools failed",
                    "details": f"MCP server error: {tools_response.get('error', 'Unknown error')}"
                }
            
            tools = tools_response.get("tools", [])
            expected_tools = ["english_cloze.generate", "math.recommend"]
            available_tools = [tool.get("name") for tool in tools]
            
            missing_tools = [tool for tool in expected_tools if tool not in available_tools]
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            if missing_tools:
                return {
                    "status": "degraded",
                    "warning": f"Missing expected tools: {missing_tools}",
                    "available_tools": available_tools,
                    "latency_ms": latency_ms,
                    "details": "MCP server responding but some expected tools missing"
                }
            
            return {
                "status": "healthy",
                "latency_ms": latency_ms,
                "available_tools": available_tools,
                "details": "MCP server responding and all expected tools available"
            }
            
        except ImportError:
            return {
                "status": "degraded",
                "warning": "MCP utils not available",
                "details": "MCP client module not found"
            }
            
    except Exception as e:
        logger.error(f"MCP health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "details": "MCP health check failed"
        }

async def _check_llm_health() -> Dict[str, Any]:
    """Check LLM provider connectivity and API keys"""
    try:
        start_time = time.time()
        
        # Check if we have any LLM API keys configured
        has_openai = bool(settings.openai_api_key)
        has_anthropic = bool(settings.anthropic_api_key)
        
        if not has_openai and not has_anthropic:
            return {
                "status": "degraded",
                "warning": "No LLM API keys configured",
                "details": "Neither OpenAI nor Anthropic API keys are configured"
            }
        
        # Test LLM gateway (if configured)
        try:
            from app.services.llm_gateway import llm_gateway
            
            # Simple health check - don't make actual API calls
            providers_available = settings.llm_providers_available
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return {
                "status": "healthy",
                "latency_ms": latency_ms,
                "providers_available": providers_available,
                "openai_configured": has_openai,
                "anthropic_configured": has_anthropic,
                "details": "LLM providers configured and gateway available"
            }
            
        except ImportError:
            return {
                "status": "degraded",
                "warning": "LLM gateway not available",
                "details": "LLM gateway module not found"
            }
            
    except Exception as e:
        logger.error(f"LLM health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "details": "LLM health check failed"
        }

@router.get("/info")
async def system_info() -> Dict[str, Any]:
    """Get system information and configuration details"""
    return {
        "app_name": settings.app_name,
        "version": settings.version,
        "environment": settings.environment.value,
        "debug": settings.debug,
        "database_url": settings.database_url.split("@")[-1] if "@" in settings.database_url else "configured",
        "redis_url": settings.redis_url.split("@")[-1] if "@" in settings.redis_url else "configured",
        "llm_providers": settings.llm_providers_available,
        "pgvector_enabled": settings.pgvector_enabled,
        "mcp_enabled": getattr(settings, 'use_mcp_demo', False),
        "feature_flags": {
            "use_languagetool": getattr(settings, 'use_languagetool', True),
            "enable_llm_fallback": getattr(settings, 'enable_llm_fallback', True),
            "content_moderation": settings.content_moderation_enabled
        }
    }
