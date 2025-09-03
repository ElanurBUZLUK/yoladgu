"""
Recommendation endpoints.
"""

import logging
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import HTTPBearer

from app.models.request import RecommendRequest, SearchRequest, RerankRequest
from app.models.response import (
    RecommendResponse, SearchResponse, RerankResponse, 
    RecommendedItem, ErrorResponse
)
from app.services.orchestration_service import orchestration_service
from app.services.retrieval_service import retrieval_service

router = APIRouter()
security = HTTPBearer()
logger = logging.getLogger(__name__)


@router.post("/next", response_model=RecommendResponse)
async def recommend_next_questions(
    request: RecommendRequest,
    req: Request,
    token: str = Depends(security)
):
    """
    Recommend next questions for a student based on their profile and learning goals.
    
    This endpoint orchestrates the full recommendation pipeline:
    1. Retrieve candidate questions using hybrid search
    2. Re-rank candidates using cross-encoder
    3. Apply diversification and curriculum coverage
    4. Use bandit algorithm for final selection
    """
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    try:
        logger.info(f"Recommendation request {request_id} for user {request.user_id}")
        
        # TODO: Add JWT token validation and extract user info
        # For now, we'll trust the user_id from the request
        
        # Validate request parameters
        if not request.user_id:
            raise HTTPException(
                status_code=400,
                detail="user_id is required"
            )
        
        # Call orchestration service
        pipeline_result = await orchestration_service.recommend_next_questions(
            user_id=request.user_id,
            target_skills=request.target_skills,
            constraints=request.constraints,
            personalization=request.personalization,
            request_id=request_id
        )
        
        # Convert to API response format
        recommended_items = []
        for item in pipeline_result.get("items", []):
            recommended_item = RecommendedItem(
                item_id=item.get("item_id"),
                generated_item=item.get("generated_item"),
                reason_tags=item.get("reason_tags", []),
                propensity=item.get("propensity")
            )
            recommended_items.append(recommended_item)
        
        metadata = pipeline_result.get("metadata", {})
        
        response = RecommendResponse(
            items=recommended_items,
            policy_id=metadata.get("policy_id", "main_pipeline_v1"),
            bandit_version=metadata.get("bandit_version", "linucb_v1.3"),
            request_id=request_id
        )
        
        # Log successful completion
        end_time = datetime.utcnow()
        total_time = (end_time - start_time).total_seconds() * 1000
        
        logger.info(
            f"Recommendation request {request_id} completed in {total_time:.2f}ms, "
            f"returned {len(recommended_items)} items"
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Recommendation request {request_id} failed: {e}", exc_info=True)
        
        # Return structured error response
        raise HTTPException(
            status_code=500,
            detail={
                "code": "RECOMMENDATION_PIPELINE_ERROR",
                "message": "Failed to generate recommendations",
                "request_id": request_id,
                "details": {"error": str(e)}
            }
        )


@router.post("/search", response_model=SearchResponse)
async def search_questions(
    request: SearchRequest,
    req: Request,
    token: str = Depends(security)
):
    """
    Search for questions using hybrid retrieval (dense + sparse).
    
    Returns candidate questions that can be further processed by re-ranking.
    """
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    try:
        logger.info(f"Search request {request_id}")
        
        # TODO: Add JWT token validation
        
        # Validate request parameters
        if request.k <= 0:
            raise HTTPException(
                status_code=400,
                detail="k must be positive"
            )
        
        # Call retrieval service
        search_results = await retrieval_service.hybrid_search(
            query=request.query,
            goals=request.goals,
            lang=request.lang,
            k=request.k
        )
        
        # Convert to API response format
        from app.models.response import SearchCandidate
        
        candidates = []
        for result in search_results:
            candidate = SearchCandidate(
                item_id=result.get("item_id", ""),
                type=result.get("metadata", {}).get("type", "unknown"),
                retriever_scores=result.get("retriever_scores", {}),
                metadata=result.get("metadata")
            )
            candidates.append(candidate)
        
        response = SearchResponse(
            candidates=candidates,
            generated_at=datetime.utcnow()
        )
        
        # Log successful completion
        end_time = datetime.utcnow()
        total_time = (end_time - start_time).total_seconds() * 1000
        
        logger.info(
            f"Search request {request_id} completed in {total_time:.2f}ms, "
            f"returned {len(candidates)} candidates"
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Search request {request_id} failed: {e}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail={
                "code": "SEARCH_ERROR",
                "message": "Failed to search questions",
                "request_id": request_id,
                "details": {"error": str(e)}
            }
        )


@router.post("/rerank", response_model=RerankResponse)
async def rerank_candidates(
    request: RerankRequest,
    req: Request,
    token: str = Depends(security)
):
    """
    Re-rank candidate questions using cross-encoder model.
    
    Takes search candidates and re-ranks them based on relevance to user context.
    """
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    try:
        logger.info(f"Re-rank request {request_id}")
        
        # TODO: Add JWT token validation
        
        # Validate request parameters
        if not request.candidates:
            raise HTTPException(
                status_code=400,
                detail="candidates list cannot be empty"
            )
        
        if request.max_k <= 0:
            raise HTTPException(
                status_code=400,
                detail="max_k must be positive"
            )
        
        # Use cross-encoder re-ranking service
        from app.services.reranking_service import reranking_service
        from app.models.response import RankedItem
        
        # Perform cross-encoder re-ranking
        reranked_candidates = await reranking_service.rerank_candidates(
            query_context=request.query_repr,
            candidates=request.candidates,
            max_k=request.max_k,
            use_cache=True
        )
        
        # Convert to response format
        ranked_items = []
        for candidate in reranked_candidates:
            ranked_item = RankedItem(
                item_id=candidate.get("item_id", ""),
                score=candidate.get("rerank_score", 0.0)
            )
            ranked_items.append(ranked_item)
        
        # Get model version from service stats
        service_stats = reranking_service.get_stats()
        model_version = f"cross-encoder/{service_stats.get('model_name', 'unknown')}"
        
        response = RerankResponse(
            ranked=ranked_items,
            model_version=model_version
        )
        
        # Log successful completion
        end_time = datetime.utcnow()
        total_time = (end_time - start_time).total_seconds() * 1000
        
        logger.info(
            f"Re-rank request {request_id} completed in {total_time:.2f}ms, "
            f"ranked {len(ranked_items)} items"
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Re-rank request {request_id} failed: {e}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail={
                "code": "RERANK_ERROR",
                "message": "Failed to re-rank candidates",
                "request_id": request_id,
                "details": {"error": str(e)}
            }
        )