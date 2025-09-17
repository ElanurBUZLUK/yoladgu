"""
Hybrid Search API Endpoints
BM25 + E5-Large-v2 + RRF for advanced search
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import structlog

from app.db.session import get_db
from app.models.request import SearchRequest, IndexRebuildRequest
from app.models.response import SearchResponse, IndexStatsResponse
from ml.nlp.hybrid import hybrid_search_engine

logger = structlog.get_logger()
router = APIRouter()

class HybridSearchRequest(BaseModel):
    """Hybrid search request model"""
    query: str = Field(..., description="Search query")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    limit: int = Field(10, ge=1, le=100, description="Number of results to return")
    search_type: str = Field("rrf", description="Search type: bm25, dense, hybrid, rrf")
    user_context: Optional[Dict[str, Any]] = Field(None, description="User context for personalization")

class HybridSearchResponse(BaseModel):
    """Hybrid search response model"""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    search_type: str
    processing_time_ms: float
    user_context: Optional[Dict[str, Any]]
    timestamp: str

class IndexRebuildResponse(BaseModel):
    """Index rebuild response model"""
    status: str
    documents_indexed: int
    processing_time_seconds: float
    index_stats: Dict[str, Any]
    timestamp: str

@router.post("/search", response_model=HybridSearchResponse)
async def hybrid_search(
    request: HybridSearchRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Perform hybrid search using BM25 + E5-Large-v2 + RRF
    """
    start_time = datetime.now()
    
    try:
        logger.info("Hybrid search request", query=request.query, user_id=request.user_id)
        
        # Get user context if user_id provided
        user_context = request.user_context
        if request.user_id and not user_context:
            # TODO: Fetch user context from database
            user_context = {
                "user_id": request.user_id,
                "user_level": "B1",  # Default level
                "user_topics": ["grammar", "vocabulary"],  # Default topics
            }
        
        # Perform search
        results = await hybrid_search_engine.search(
            query=request.query,
            k=request.limit,
            user_context=user_context,
            search_type=request.search_type
        )
        
        # Convert results to response format
        search_results = []
        for result in results:
            search_results.append({
                "doc_id": result.doc_id,
                "score": result.score,
                "text": result.text,
                "metadata": result.metadata,
                "bm25_score": getattr(result, "bm25_score", None),
                "dense_score": getattr(result, "dense_score", None)
            })
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return HybridSearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            search_type=request.search_type,
            processing_time_ms=processing_time,
            user_context=user_context,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Hybrid search failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@router.post("/index/rebuild", response_model=IndexRebuildResponse)
async def rebuild_index(
    request: IndexRebuildRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Rebuild the hybrid search index with new documents
    """
    start_time = datetime.now()
    
    try:
        logger.info("Rebuilding hybrid search index")
        
        # TODO: Fetch documents from database
        # For now, use sample documents
        sample_documents = [
            {
                "id": "doc_1",
                "text": "English grammar is the structure of expressions in the English language.",
                "metadata": {
                    "type": "grammar",
                    "level": "A2",
                    "topics": ["grammar", "structure"],
                    "timestamp": datetime.now().timestamp()
                }
            },
            {
                "id": "doc_2", 
                "text": "Vocabulary learning is essential for language proficiency.",
                "metadata": {
                    "type": "vocabulary",
                    "level": "B1",
                    "topics": ["vocabulary", "learning"],
                    "timestamp": datetime.now().timestamp()
                }
            },
            {
                "id": "doc_3",
                "text": "Reading comprehension improves with practice and exposure to texts.",
                "metadata": {
                    "type": "reading",
                    "level": "B2",
                    "topics": ["reading", "comprehension"],
                    "timestamp": datetime.now().timestamp()
                }
            }
        ]
        
        # Add documents to hybrid search engine
        await hybrid_search_engine.add_documents(sample_documents)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        stats = hybrid_search_engine.get_stats()
        
        return IndexRebuildResponse(
            status="success",
            documents_indexed=len(sample_documents),
            processing_time_seconds=processing_time,
            index_stats=stats,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Index rebuild failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Index rebuild failed: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check for hybrid search system"""
    try:
        stats = hybrid_search_engine.get_stats()
        
        return {
            "status": "healthy",
            "system": "Hybrid Search Engine",
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/stats")
async def get_search_stats():
    """Get detailed search engine statistics"""
    try:
        stats = hybrid_search_engine.get_stats()
        
        return {
            "status": "success",
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )
