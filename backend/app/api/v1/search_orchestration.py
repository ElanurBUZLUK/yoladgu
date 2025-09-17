"""
Search Orchestration API Endpoints
Integrates E5 + RRF with existing retrieval systems
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import structlog

from app.db.session import get_db
from app.services.search_orchestration_service import search_orchestration_service

logger = structlog.get_logger()
router = APIRouter()

class SearchRequest(BaseModel):
    """Request for orchestrated search"""
    user_id: str = Field(..., description="User ID")
    query: str = Field(..., description="Search query")
    strategy: str = Field("legacy", description="Search strategy: legacy, hybrid_e5, advanced_rag")
    top_k: int = Field(10, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")

class SearchResponse(BaseModel):
    """Response for orchestrated search"""
    results: List[Dict[str, Any]]
    strategy: str
    metadata: Dict[str, Any]
    timestamp: str

@router.post("/search", response_model=SearchResponse)
async def orchestrated_search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Perform orchestrated search using specified strategy
    
    Supports:
    - legacy: Traditional vector search
    - hybrid_e5: E5-Large-v2 + BM25 + RRF
    - advanced_rag: Advanced RAG with custom prompts
    """
    try:
        logger.info("Starting orchestrated search", 
                   user_id=request.user_id, strategy=request.strategy, query=request.query)
        
        # Perform orchestrated search
        results = await search_orchestration_service.search(
            user_id=request.user_id,
            query=request.query,
            strategy=request.strategy,
            top_k=request.top_k,
            filters=request.filters
        )
        
        if results.get("error"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Search failed: {results['error']}"
            )
        
        return SearchResponse(
            results=results.get("results", []),
            strategy=results.get("strategy", request.strategy),
            metadata=results.get("metadata", {}),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Orchestrated search failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@router.get("/strategies")
async def get_available_strategies():
    """Get available search strategies"""
    try:
        strategies = await search_orchestration_service.get_strategy_info()
        return {
            "strategies": strategies,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error("Failed to get strategies", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get strategies: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check for search orchestration"""
    try:
        health_status = await search_orchestration_service.health_check()
        return {
            "status": "healthy" if health_status.get("search_orchestration_service") else "unhealthy",
            "components": health_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
