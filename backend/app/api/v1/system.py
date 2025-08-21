from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import Dict, Any, List
import logging
import time

from app.database import database_manager
from app.core.config import settings
from app.services.vector_index_manager import vector_index_manager
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/system", tags=["system"])

@router.get("/readyz")
async def health_check(db: AsyncSession = Depends(database_manager.get_session)) -> Dict[str, Any]:
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
    """
    Get comprehensive system information including configuration and capabilities.
    
    Returns:
        Dict with system information, configuration, and feature flags
    """
    try:
        info = {
            "system": {
                "name": "Yoladgu Adaptive Learning Backend",
                "version": "2.0.0",
                "environment": settings.environment,
                "python_version": "3.12+",
                "framework": "FastAPI",
                "database": "PostgreSQL + pgvector",
                "cache": "Redis",
                "llm_providers": []
            },
            "features": {
                "embedding_system": {
                    "enabled": settings.pgvector_enabled,
                    "dimension": settings.embedding_dimension,
                    "model": settings.embedding_model,
                    "batch_size": settings.embedding_batch_size
                },
                "vector_search": {
                    "enabled": True,
                    "similarity_threshold": settings.vector_similarity_threshold,
                    "namespaces": ["english_errors", "english_questions", "math_errors", "math_questions", "cefr_rubrics"]
                },
                "question_generation": {
                    "enabled": True,
                    "template_based": True,
                    "gpt_based": True,
                    "hybrid_orchestration": True
                },
                "real_time_updates": True,
                "performance_monitoring": True
            },
            "configuration": {
                "database_url": f"postgresql://***:***@{settings.database_host}:{settings.database_port}/{settings.database_name}",
                "redis_url": f"redis://***:***@{settings.redis_host}:{settings.redis_port}",
                "cors_origins": settings.cors_origins[:3] + ["..."] if len(settings.cors_origins) > 3 else settings.cors_origins,
                "max_upload_size_mb": settings.max_upload_size_mb,
                "rate_limit_per_minute": settings.rate_limit_per_minute
            }
        }
        
        # Add LLM provider info
        if settings.openai_api_key:
            info["system"]["llm_providers"].append("OpenAI GPT-4")
        if settings.anthropic_api_key:
            info["system"]["llm_providers"].append("Anthropic Claude")
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system info: {str(e)}"
        )

@router.get("/vector/performance")
async def get_vector_performance_metrics() -> Dict[str, Any]:
    """
    Get comprehensive vector index performance metrics.
    
    Returns:
        Dict with performance metrics, index statistics, and optimization recommendations
    """
    try:
        # Get performance metrics
        performance_metrics = await vector_index_manager.get_performance_metrics()
        
        # Get real-time metrics
        real_time_metrics = await vector_index_manager.get_real_time_metrics()
        
        # Get domain performance summary
        domain_summary = await vector_index_manager.get_domain_performance_summary()
        
        # Get optimization recommendations
        optimization_recommendations = await vector_index_manager.get_optimization_recommendations()
        
        return {
            "timestamp": time.time(),
            "performance_metrics": performance_metrics,
            "real_time_metrics": real_time_metrics,
            "domain_performance": domain_summary,
            "optimization_recommendations": optimization_recommendations
        }
        
    except Exception as e:
        logger.error(f"Error getting vector performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance metrics: {str(e)}"
        )

@router.get("/vector/health")
async def get_vector_health_status() -> Dict[str, Any]:
    """
    Get comprehensive vector system health status.
    
    Returns:
        Dict with health status of all vector components
    """
    try:
        health_status = await vector_index_manager.get_system_health_status()
        return health_status
        
    except Exception as e:
        logger.error(f"Error getting vector health status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get health status: {str(e)}"
        )

@router.post("/vector/optimize/{domain}")
async def optimize_vector_indexes_for_domain(domain: str) -> Dict[str, Any]:
    """
    Optimize vector indexes for a specific domain.
    
    Args:
        domain: Domain to optimize (english, math, cefr)
    
    Returns:
        Dict with optimization results and status
    """
    try:
        if domain not in ["english", "math", "cefr"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Domain must be one of: english, math, cefr"
            )
        
        optimization_results = await vector_index_manager.optimize_indexes_for_domain(domain)
        return optimization_results
        
    except Exception as e:
        logger.error(f"Error optimizing indexes for domain {domain}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to optimize indexes: {str(e)}"
        )

@router.get("/vector/domains/{domain}/metrics")
async def get_domain_specific_metrics(domain: str) -> Dict[str, Any]:
    """
    Get performance metrics for a specific domain.
    
    Args:
        domain: Domain to get metrics for (english, math, cefr)
    
    Returns:
        Dict with domain-specific metrics and performance data
    """
    try:
        if domain not in ["english", "math", "cefr"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Domain must be one of: english, math, cefr"
            )
        
        # Get domain metrics
        domain_metrics = await vector_index_manager._get_domain_metrics(domain)
        
        # Get domain-specific performance data
        domain_performance = await vector_index_manager.get_domain_performance_summary()
        domain_data = domain_performance.get(domain, {})
        
        return {
            "domain": domain,
            "timestamp": time.time(),
            "metrics": domain_metrics,
            "performance": domain_data,
            "recommendations": []
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics for domain {domain}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get domain metrics: {str(e)}"
        )

@router.get("/vector/namespaces")
async def get_namespace_statistics() -> Dict[str, Any]:
    """
    Get statistics for all vector namespaces.
    
    Returns:
        Dict with namespace statistics and distribution
    """
    try:
        # Get namespace distribution from performance metrics
        performance_metrics = await vector_index_manager.get_performance_metrics()
        namespace_dist = performance_metrics.get("namespace_distribution", [])
        
        # Group by domain
        domain_namespaces = {
            "english": [],
            "math": [],
            "cefr": []
        }
        
        for namespace_info in namespace_dist:
            namespace = namespace_info["namespace"]
            if namespace.startswith("english"):
                domain_namespaces["english"].append(namespace_info)
            elif namespace.startswith("math"):
                domain_namespaces["math"].append(namespace_info)
            elif namespace.startswith("cefr") or namespace.startswith("user_assessments"):
                domain_namespaces["cefr"].append(namespace_info)
        
        return {
            "timestamp": time.time(),
            "total_namespaces": len(namespace_dist),
            "total_embeddings": sum(ns["count"] for ns in namespace_dist),
            "domain_distribution": domain_namespaces,
            "namespace_details": namespace_dist
        }
        
    except Exception as e:
        logger.error(f"Error getting namespace statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get namespace statistics: {str(e)}"
        )
