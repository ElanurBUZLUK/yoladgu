"""
RAG API endpoints for semantic search and retrieval
"""
from typing import List, Optional, Tuple
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import database_manager
from app.services.real_rag_service import real_rag_service
from app.schemas.rag import (
    RAGSearchRequest,
    RAGSearchResponse,
    RAGStatisticsResponse,
    EmbeddingGenerationRequest,
    EmbeddingGenerationResponse
)
from app.core.error_handling import ErrorHandler, ErrorCode, ErrorSeverity

router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])
error_handler = ErrorHandler()


@router.post("/search", response_model=RAGSearchResponse)
async def search_similar_questions(
    request: RAGSearchRequest,
    db: AsyncSession = Depends(database_manager.get_session)
):
    """
    Search for similar questions using RAG (semantic + keyword search)
    """
    try:
        # Parse difficulty range
        difficulty_range = None
        if request.min_difficulty is not None and request.max_difficulty is not None:
            difficulty_range = (request.min_difficulty, request.max_difficulty)
        
        # Perform hybrid search
        results = await real_rag_service.hybrid_search(
            query=request.query,
            subject=request.subject,
            difficulty_range=difficulty_range,
            limit=request.limit or 10,
            db=db
        )
        
        # Format response
        formatted_results = []
        for result in results:
            question = result["question"]
            formatted_results.append({
                "id": str(question.id),
                "content": question.content,
                "subject": question.subject,
                "difficulty_level": question.difficulty_level,
                "topic_category": question.topic_category,
                "question_type": question.question_type,
                "similarity_score": result["similarity_score"],
                "search_type": result.get("search_type", "semantic"),
                "metadata": result.get("metadata", {})
            })
        
        return RAGSearchResponse(
            query=request.query,
            subject=request.subject,
            results=formatted_results,
            total_count=len(formatted_results),
            search_type="hybrid"
        )
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.RAG_SEARCH_ERROR,
            message="Failed to perform RAG search",
            severity=ErrorSeverity.MEDIUM,
            context={
                "query": request.query,
                "subject": request.subject,
                "limit": request.limit
            }
        )


@router.post("/search/semantic", response_model=RAGSearchResponse)
async def semantic_search(
    request: RAGSearchRequest,
    db: AsyncSession = Depends(database_manager.get_session)
):
    """
    Search for similar questions using only semantic similarity
    """
    try:
        # Parse difficulty range
        difficulty_range = None
        if request.min_difficulty is not None and request.max_difficulty is not None:
            difficulty_range = (request.min_difficulty, request.max_difficulty)
        
        # Perform semantic search
        results = await real_rag_service.search_similar_questions(
            query=request.query,
            subject=request.subject,
            difficulty_range=difficulty_range,
            limit=request.limit or 10,
            db=db
        )
        
        # Format response
        formatted_results = []
        for result in results:
            question = result["question"]
            formatted_results.append({
                "id": str(question.id),
                "content": question.content,
                "subject": question.subject,
                "difficulty_level": question.difficulty_level,
                "topic_category": question.topic_category,
                "question_type": question.question_type,
                "similarity_score": result["similarity_score"],
                "search_type": "semantic",
                "metadata": result.get("metadata", {})
            })
        
        return RAGSearchResponse(
            query=request.query,
            subject=request.subject,
            results=formatted_results,
            total_count=len(formatted_results),
            search_type="semantic"
        )
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.RAG_SEARCH_ERROR,
            message="Failed to perform semantic search",
            severity=ErrorSeverity.MEDIUM,
            context={
                "query": request.query,
                "subject": request.subject,
                "limit": request.limit
            }
        )


@router.get("/statistics", response_model=RAGStatisticsResponse)
async def get_rag_statistics(
    db: AsyncSession = Depends(database_manager.get_session)
):
    """
    Get RAG system statistics and performance metrics
    """
    try:
        stats = await real_rag_service.get_rag_statistics(db=db)
        
        if "error" in stats:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get RAG statistics: {stats['error']}"
            )
        
        return RAGStatisticsResponse(**stats)
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.RAG_STATISTICS_ERROR,
            message="Failed to get RAG statistics",
            severity=ErrorSeverity.LOW,
            context={}
        )


@router.post("/embeddings/generate", response_model=EmbeddingGenerationResponse)
async def generate_question_embeddings(
    request: EmbeddingGenerationRequest,
    db: AsyncSession = Depends(database_manager.get_session)
):
    """
    Generate embeddings for questions that don't have them
    """
    try:
        result = await real_rag_service.generate_question_embeddings(
            question_ids=request.question_ids,
            db=db
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate embeddings: {result.get('error', 'Unknown error')}"
            )
        
        return EmbeddingGenerationResponse(**result)
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.EMBEDDING_GENERATION_ERROR,
            message="Failed to generate question embeddings",
            severity=ErrorSeverity.MEDIUM,
            context={
                "question_count": len(request.question_ids)
            }
        )


@router.post("/search-vectors/update")
async def update_search_vectors(
    db: AsyncSession = Depends(database_manager.get_session)
):
    """
    Update search vectors for all questions (full-text search)
    """
    try:
        result = await real_rag_service.update_search_vectors(db=db)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update search vectors: {result.get('error', 'Unknown error')}"
            )
        
        return {
            "success": True,
            "message": f"Updated search vectors for {result['updated']} questions",
            "updated_count": result["updated"]
        }
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.SEARCH_VECTOR_UPDATE_ERROR,
            message="Failed to update search vectors",
            severity=ErrorSeverity.MEDIUM,
            context={}
        )
