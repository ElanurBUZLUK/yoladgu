"""
Error-Aware Recommendation API endpoints.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import numpy as np
from numpy.typing import NDArray

from app.services.recommenders.error_aware import ErrorAwareRecommender, ErrorAwareConfig
from app.services.vector_service import vector_service

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# Global recommender instance
error_aware_recommender = ErrorAwareRecommender()


class RecommendationRequest(BaseModel):
    """Request model for error-aware recommendations."""
    student_id: str = Field(..., description="Target student ID")
    k: int = Field(10, ge=1, le=50, description="Number of recommendations to return")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Weight for similarity vs lift scoring")
    use_hnsw: bool = Field(True, description="Whether to use HNSW for neighbor search")
    window_size: Optional[int] = Field(None, ge=10, le=200, description="Window size for student attempts")
    decay_factor: Optional[float] = Field(None, ge=0.0, le=1.0, description="Time decay factor")


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    student_id: str
    recommendations: List[str]
    scores: List[float]
    metadata: Dict[str, Any]
    performance: Dict[str, Any]


class BatchRecommendationRequest(BaseModel):
    """Request model for batch recommendations."""
    requests: List[RecommendationRequest] = Field(..., max_items=100)


class BatchRecommendationResponse(BaseModel):
    """Response model for batch recommendations."""
    results: List[RecommendationResponse]
    summary: Dict[str, Any]


@router.post("/error-aware", response_model=RecommendationResponse)
async def get_error_aware_recommendations(
    request: RecommendationRequest
) -> RecommendationResponse:
    """
    Get error-aware recommendations for a student.
    
    This endpoint provides personalized question recommendations based on:
    - Student's error profile
    - Similar questions based on error patterns
    - Collaborative filtering using neighbor-lift scoring
    """
    try:
        # Update recommender config if custom parameters provided
        if request.window_size is not None or request.decay_factor is not None:
            config = ErrorAwareConfig(
                window_size=request.window_size or error_aware_recommender.config.window_size,
                decay_factor=request.decay_factor or error_aware_recommender.config.decay_factor,
                use_hnsw=request.use_hnsw
            )
            recommender = ErrorAwareRecommender(config)
        else:
            recommender = error_aware_recommender
        
        # TODO: In a real application, these would come from the database
        # For now, we'll generate mock data for demonstration
        
        # Mock data generation
        mock_data = _generate_mock_data()
        
        # Get recommendations
        recommendations = await recommender.recommend_error_aware(
            attempts=mock_data["attempts"],
            items_errors=mock_data["items_errors"],
            student_ids=mock_data["student_ids"],
            student_vectors=mock_data["student_vectors"],
            target_student_id=request.student_id,
            alpha=request.alpha,
            k=request.k
        )
        
        # Calculate scores for recommendations
        scores = _calculate_recommendation_scores(
            recommendations, 
            mock_data["items_errors"], 
            mock_data["vocab"]
        )
        
        # Get performance metrics
        stats = recommender.get_stats()
        
        return RecommendationResponse(
            student_id=request.student_id,
            recommendations=recommendations,
            scores=scores,
            metadata={
                "total_questions": len(mock_data["items_errors"]),
                "vocab_size": len(mock_data["vocab"]),
                "total_students": len(mock_data["student_ids"]),
                "config": {
                    "alpha": request.alpha,
                    "use_hnsw": request.use_hnsw,
                    "window_size": request.window_size or recommender.config.window_size
                }
            },
            performance={
                "response_time_ms": stats.get("avg_response_time_ms", 0.0),
                "hnsw_usage": stats.get("hnsw_usage", 0),
                "exact_fallback": stats.get("exact_fallback", 0)
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )


@router.post("/error-aware/batch", response_model=BatchRecommendationResponse)
async def get_batch_error_aware_recommendations(
    request: BatchRecommendationRequest
) -> BatchRecommendationResponse:
    """
    Get error-aware recommendations for multiple students in batch.
    """
    try:
        results = []
        
        for req in request.requests:
            # Generate recommendations for each student
            mock_data = _generate_mock_data()
            
            recommender = ErrorAwareRecommender()
            recommendations = await recommender.recommend_error_aware(
                attempts=mock_data["attempts"],
                items_errors=mock_data["items_errors"],
                student_ids=mock_data["student_ids"],
                student_vectors=mock_data["student_vectors"],
                target_student_id=req.student_id,
                alpha=req.alpha,
                k=req.k
            )
            
            scores = _calculate_recommendation_scores(
                recommendations, 
                mock_data["items_errors"], 
                mock_data["vocab"]
            )
            
            stats = recommender.get_stats()
            
            results.append(RecommendationResponse(
                student_id=req.student_id,
                recommendations=recommendations,
                scores=scores,
                metadata={
                    "total_questions": len(mock_data["items_errors"]),
                    "vocab_size": len(mock_data["vocab"]),
                    "total_students": len(mock_data["student_ids"])
                },
                performance={
                    "response_time_ms": stats.get("avg_response_time_ms", 0.0),
                    "hnsw_usage": stats.get("hnsw_usage", 0),
                    "exact_fallback": stats.get("exact_fallback", 0)
                }
            ))
        
        # Calculate summary statistics
        total_recommendations = sum(len(r.recommendations) for r in results)
        avg_response_time = np.mean([r.performance["response_time_ms"] for r in results])
        
        return BatchRecommendationResponse(
            results=results,
            summary={
                "total_students": len(results),
                "total_recommendations": total_recommendations,
                "avg_response_time_ms": avg_response_time,
                "success_rate": 1.0
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating batch recommendations: {str(e)}"
        )


@router.get("/error-aware/stats")
async def get_recommendation_stats() -> Dict[str, Any]:
    """Get statistics about the recommendation service."""
    try:
        stats = error_aware_recommender.get_stats()
        return {
            "service": "error_aware_recommender",
            "stats": stats,
            "config": {
                "window_size": error_aware_recommender.config.window_size,
                "neighbor_count": error_aware_recommender.config.neighbor_count,
                "similarity_weight": error_aware_recommender.config.similarity_weight,
                "use_hnsw": error_aware_recommender.config.use_hnsw
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stats: {str(e)}"
        )


@router.post("/error-aware/clear-cache")
async def clear_recommendation_cache() -> Dict[str, str]:
    """Clear the recommendation cache."""
    try:
        error_aware_recommender.clear_cache()
        return {"message": "Recommendation cache cleared successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing cache: {str(e)}"
        )


def _generate_mock_data() -> Dict[str, Any]:
    """Generate mock data for testing/demonstration purposes."""
    
    # Generate error tags
    error_tags = [
        "algebra_error", "fraction_error", "geometry_error", 
        "calculation_error", "concept_error", "syntax_error",
        "word_problem_error", "visualization_error"
    ]
    
    # Generate items with error tags
    items_errors = {}
    for i in range(50):
        item_id = f"Q{i+1:03d}"
        # Randomly assign 2-4 error tags to each item
        num_errors = np.random.randint(2, 5)
        item_errors = np.random.choice(error_tags, num_errors, replace=False).tolist()
        items_errors[item_id] = item_errors
    
    # Generate student attempts
    attempts = []
    for student_id in range(10):
        for item_id in range(50):
            # Randomly generate attempts with some correct/incorrect patterns
            is_correct = np.random.choice([True, False], p=[0.7, 0.3])
            age_days = np.random.randint(0, 30)
            
            attempts.append({
                "user_id": f"student_{student_id}",
                "item_id": f"Q{item_id+1:03d}",
                "correct": is_correct,
                "age_days": age_days
            })
    
    # Generate student error profile vectors
    vocab = list(set([tag for tags in items_errors.values() for tag in tags]))
    student_vectors = np.random.randn(10, len(vocab)).astype(np.float32)
    
    # Normalize vectors
    norms = np.linalg.norm(student_vectors, axis=1, keepdims=True)
    student_vectors = student_vectors / np.where(norms > 0, norms, 1.0)
    
    student_ids = [f"student_{i}" for i in range(10)]
    
    return {
        "items_errors": items_errors,
        "attempts": attempts,
        "student_vectors": student_vectors,
        "student_ids": student_ids,
        "vocab": vocab
    }


def _calculate_recommendation_scores(
    recommendations: List[str], 
    items_errors: Dict[str, List[str]], 
    vocab: List[str]
) -> List[float]:
    """Calculate scores for recommendations based on error complexity."""
    scores = []
    
    for rec in recommendations:
        if rec in items_errors:
            # Score based on number of error types (more complex = higher score)
            error_count = len(items_errors[rec])
            # Normalize to 0-1 range
            score = min(error_count / len(vocab), 1.0)
            scores.append(score)
        else:
            scores.append(0.0)
    
    return scores
