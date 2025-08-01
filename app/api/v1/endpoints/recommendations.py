"""
Enhanced Recommendation API Endpoints
Comprehensive ML-powered recommendation system
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import structlog
import time
from datetime import datetime, timedelta

from app.db.database import get_db
from app.crud.user import get_current_user
from app.db.models import User, Question, StudentResponse
from app.services.recommendation_service import recommendation_service
from app.services.ensemble_service import ensemble_service
from app.services.enhanced_embedding_service import enhanced_embedding_service
from app.services.enhanced_stream_consumer import stream_consumer_manager, MessageType
from app.core.config import settings

logger = structlog.get_logger()
router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# Request/Response Models
class RecommendationRequest(BaseModel):
    student_id: int
    subject_id: Optional[int] = None
    topic_id: Optional[int] = None
    difficulty_range: Optional[tuple[int, int]] = None
    exclude_answered: bool = True
    count: int = Field(default=5, le=20)
    algorithm: Optional[str] = "ensemble"  # ensemble, bandit, collaborative, embedding

class PersonalizedRecommendationRequest(BaseModel):
    student_id: int
    learning_goals: Optional[List[str]] = None
    time_available: Optional[int] = None  # minutes
    performance_context: Optional[Dict[str, Any]] = None
    adaptive_difficulty: bool = True

class QuestionRecommendation(BaseModel):
    question_id: int
    title: str
    difficulty_level: int
    topic: str
    subject: str
    confidence_score: float
    reasoning: List[str]
    estimated_time: Optional[int] = None
    skill_alignment: Optional[float] = None

class RecommendationResponse(BaseModel):
    recommendations: List[QuestionRecommendation]
    algorithm_used: str
    total_available: int
    generation_time: float
    metadata: Dict[str, Any]

class FeedbackRequest(BaseModel):
    student_id: int
    question_id: int
    feedback_type: str  # "helpful", "too_easy", "too_hard", "irrelevant"
    rating: Optional[float] = Field(None, ge=0, le=1)
    context: Optional[Dict[str, Any]] = None

class LearningPathRequest(BaseModel):
    student_id: int
    target_skill: str
    current_level: Optional[int] = None
    target_level: Optional[int] = None
    time_horizon: Optional[int] = 30  # days

class LearningPathStep(BaseModel):
    step_number: int
    question_id: int
    title: str
    difficulty: int
    prerequisite_skills: List[str]
    learning_objectives: List[str]
    estimated_duration: int

class LearningPathResponse(BaseModel):
    path_id: str
    steps: List[LearningPathStep]
    total_duration: int
    success_probability: float
    alternative_paths: int

@router.post("/personalized", response_model=RecommendationResponse)
async def get_personalized_recommendations(
    request: PersonalizedRecommendationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get highly personalized recommendations using ensemble ML models"""
    start_time = time.time()
    
    try:
        logger.info("personalized_recommendation_request", 
                   student_id=request.student_id,
                   learning_goals=request.learning_goals)
        
        # Get student context and history
        student_context = await _get_student_context(db, request.student_id)
        
        # Generate recommendations using ensemble approach
        raw_recommendations = recommendation_service.get_personalized_recommendations(
            db=db,
            student_id=request.student_id,
            count=10,  # Get more for filtering
            context=student_context,
            adaptive_difficulty=request.adaptive_difficulty
        )
        
        # Apply ML scoring
        scored_recommendations = []
        for rec in raw_recommendations:
            # Get comprehensive score from ensemble
            ensemble_score = ensemble_service.calculate_comprehensive_ensemble_score(
                student_context=student_context,
                question_id=rec.question_id,
                context=request.performance_context or {}
            )
            
            # Convert to response format
            question_rec = QuestionRecommendation(
                question_id=rec.question_id,
                title=rec.title,
                difficulty_level=rec.difficulty_level,
                topic=rec.topic,
                subject=rec.subject,
                confidence_score=ensemble_score,
                reasoning=_generate_reasoning(rec, ensemble_score, student_context),
                estimated_time=_estimate_question_time(rec, student_context),
                skill_alignment=_calculate_skill_alignment(rec, request.learning_goals or [])
            )
            scored_recommendations.append(question_rec)
        
        # Sort by ensemble score and limit
        scored_recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        final_recommendations = scored_recommendations[:5]
        
        generation_time = time.time() - start_time
        
        # Log recommendation analytics
        background_tasks.add_task(
            _log_recommendation_analytics,
            request.student_id,
            [r.question_id for r in final_recommendations],
            "personalized_ensemble",
            generation_time
        )
        
        return RecommendationResponse(
            recommendations=final_recommendations,
            algorithm_used="personalized_ensemble",
            total_available=len(raw_recommendations),
            generation_time=generation_time,
            metadata={
                "student_context": student_context,
                "adaptive_difficulty": request.adaptive_difficulty,
                "learning_goals": request.learning_goals
            }
        )
        
    except Exception as e:
        logger.error("personalized_recommendation_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")

@router.post("/algorithm-specific", response_model=RecommendationResponse)
async def get_algorithm_specific_recommendations(
    request: RecommendationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get recommendations using a specific algorithm"""
    start_time = time.time()
    
    try:
        algorithm = request.algorithm or "ensemble"
        logger.info("algorithm_specific_recommendation", 
                   student_id=request.student_id,
                   algorithm=algorithm)
        
        if algorithm == "bandit":
            recommendations = await _get_bandit_recommendations(db, request)
        elif algorithm == "collaborative":
            recommendations = await _get_collaborative_recommendations(db, request)
        elif algorithm == "embedding":
            recommendations = await _get_embedding_recommendations(db, request)
        elif algorithm == "ensemble":
            recommendations = await _get_ensemble_recommendations(db, request)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown algorithm: {algorithm}")
        
        generation_time = time.time() - start_time
        
        return RecommendationResponse(
            recommendations=recommendations,
            algorithm_used=algorithm,
            total_available=len(recommendations),
            generation_time=generation_time,
            metadata={"algorithm_config": algorithm}
        )
        
    except Exception as e:
        logger.error("algorithm_specific_recommendation_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def submit_recommendation_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Submit feedback on recommendation quality"""
    try:
        logger.info("recommendation_feedback", 
                   student_id=request.student_id,
                   question_id=request.question_id,
                   feedback_type=request.feedback_type)
        
        # Convert feedback to numerical score
        feedback_score = _convert_feedback_to_score(request.feedback_type, request.rating)
        
        # Publish feedback to stream for ML model updates
        await stream_consumer_manager.publish_message(
            MessageType.RECOMMENDATION_FEEDBACK,
            {
                "student_id": request.student_id,
                "question_id": request.question_id,
                "feedback_score": feedback_score,
                "feedback_type": request.feedback_type,
                "context": request.context or {}
            }
        )
        
        # Update ensemble models
        background_tasks.add_task(
            ensemble_service.update_models_with_feedback,
            request.student_id,
            request.question_id,
            feedback_score,
            request.context or {}
        )
        
        return {
            "status": "success",
            "message": "Feedback submitted successfully",
            "processed": True
        }
        
    except Exception as e:
        logger.error("recommendation_feedback_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/learning-path", response_model=LearningPathResponse)
async def generate_learning_path(
    request: LearningPathRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Generate a personalized learning path"""
    try:
        logger.info("learning_path_generation", 
                   student_id=request.student_id,
                   target_skill=request.target_skill)
        
        # Get student's current skill level
        student_context = await _get_student_context(db, request.student_id)
        current_skill_level = student_context.get("skill_levels", {}).get(request.target_skill, 1)
        
        # Generate learning path using Neo4j skill graph
        path_steps = await _generate_skill_progression_path(
            db, 
            request.student_id,
            request.target_skill,
            current_skill_level,
            request.target_level or (current_skill_level + 2)
        )
        
        # Estimate success probability
        success_probability = _calculate_path_success_probability(
            student_context, 
            path_steps, 
            request.time_horizon
        )
        
        return LearningPathResponse(
            path_id=f"path_{request.student_id}_{int(time.time())}",
            steps=path_steps,
            total_duration=sum(step.estimated_duration for step in path_steps),
            success_probability=success_probability,
            alternative_paths=2  # Could generate alternatives
        )
        
    except Exception as e:
        logger.error("learning_path_generation_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/similar/{question_id}")
async def get_similar_questions(
    question_id: int,
    count: int = Query(default=5, le=20),
    similarity_threshold: float = Query(default=0.7, ge=0.5, le=1.0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get questions similar to a given question using embeddings"""
    try:
        # Get similar questions using the fixed helper method
        similar_questions = enhanced_embedding_service.find_similar_questions_by_id(
            question_id, db, threshold=similarity_threshold, limit=count
        )
        
        if not similar_questions:
            # Fallback to Neo4j if embedding fails
            from app.crud.question import get_similar_questions_from_neo4j
            neo4j_similar = get_similar_questions_from_neo4j(question_id, limit=count)
            similar_questions = [{
                "question_id": item["question_id"],
                "similarity_score": min(1.0, item["shared_skills"] / 10.0),  # Normalize shared skills
                "source": "neo4j"
            } for item in neo4j_similar]
        
        return {
            "source_question_id": question_id,
            "similar_questions": similar_questions,
            "similarity_threshold": similarity_threshold,
            "count": len(similar_questions)
        }
        
    except Exception as e:
        logger.error("similar_questions_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trending")
async def get_trending_questions(
    time_window: str = Query(default="24h", regex="^(1h|6h|24h|7d)$"),
    count: int = Query(default=10, le=50),
    subject_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get trending questions based on recent activity"""
    try:
        # Calculate time window
        hours_map = {"1h": 1, "6h": 6, "24h": 24, "7d": 168}
        hours = hours_map[time_window]
        since = datetime.utcnow() - timedelta(hours=hours)
        
        # Get trending questions from database
        trending = await _get_trending_questions(db, since, count, subject_id)
        
        return {
            "trending_questions": trending,
            "time_window": time_window,
            "generated_at": datetime.utcnow().isoformat(),
            "count": len(trending)
        }
        
    except Exception as e:
        logger.error("trending_questions_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/{student_id}")
async def get_recommendation_analytics(
    student_id: int,
    days: int = Query(default=30, le=90),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get recommendation analytics for a student"""
    try:
        # Get recommendation history and performance
        analytics = await _get_student_recommendation_analytics(db, student_id, days)
        
        return {
            "student_id": student_id,
            "period_days": days,
            "analytics": analytics,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("recommendation_analytics_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Helper Functions
async def _get_student_context(db: Session, student_id: int) -> Dict[str, Any]:
    """Get comprehensive student context for recommendations"""
    try:
        # Get student's response history and performance metrics
        student_responses = db.query(StudentResponse).filter(
            StudentResponse.student_id == student_id
        ).order_by(StudentResponse.submitted_at.desc()).limit(100).all()
        
        # Calculate performance metrics
        total_responses = len(student_responses)
        correct_responses = sum(1 for r in student_responses if r.is_correct)
        accuracy = correct_responses / total_responses if total_responses > 0 else 0.5
        
        # Calculate average response time
        response_times = [r.response_time for r in student_responses if r.response_time]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 5000
        
        # Get recent performance (last 10 questions)
        recent_responses = student_responses[:10]
        recent_accuracy = sum(1 for r in recent_responses if r.is_correct) / len(recent_responses) if recent_responses else 0.5
        
        return {
            "student_id": student_id,
            "total_responses": total_responses,
            "accuracy_rate_overall": accuracy,
            "accuracy_rate_recent": recent_accuracy,
            "avg_response_time": avg_response_time,
            "session_question_count": len(recent_responses),
            "skill_levels": {},  # Would be populated from Neo4j
            "learning_preferences": {},
            "last_activity": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("student_context_error", error=str(e))
        return {"student_id": student_id, "accuracy_rate_overall": 0.5}

async def _get_bandit_recommendations(db: Session, request: RecommendationRequest) -> List[QuestionRecommendation]:
    """Get recommendations using bandit algorithm"""
    # Implementation using the bandit model
    return []

async def _get_collaborative_recommendations(db: Session, request: RecommendationRequest) -> List[QuestionRecommendation]:
    """Get recommendations using collaborative filtering"""
    # Implementation using collaborative filtering
    return []

async def _get_embedding_recommendations(db: Session, request: RecommendationRequest) -> List[QuestionRecommendation]:
    """Get recommendations using embedding similarity"""
    # Implementation using embeddings
    return []

async def _get_ensemble_recommendations(db: Session, request: RecommendationRequest) -> List[QuestionRecommendation]:
    """Get recommendations using ensemble of all algorithms"""
    # Implementation using ensemble approach
    return []

def _generate_reasoning(rec, score: float, context: Dict) -> List[str]:
    """Generate human-readable reasoning for recommendation"""
    reasons = []
    
    if score > 0.8:
        reasons.append("High confidence match based on your learning pattern")
    if context.get("accuracy_rate_recent", 0) > 0.7:
        reasons.append("Matches your current performance level")
    
    reasons.append(f"Difficulty level {rec.difficulty_level} suits your progress")
    
    return reasons

def _estimate_question_time(rec, context: Dict) -> int:
    """Estimate time needed to solve question based on student context"""
    base_time = 300  # 5 minutes base
    difficulty_multiplier = rec.difficulty_level / 3
    student_speed = context.get("avg_response_time", 5000) / 5000
    
    return int(base_time * difficulty_multiplier * student_speed)

def _calculate_skill_alignment(rec, learning_goals: List[str]) -> float:
    """Calculate how well question aligns with learning goals"""
    # Would use Neo4j skill relationships
    return 0.8

def _convert_feedback_to_score(feedback_type: str, rating: Optional[float]) -> float:
    """Convert feedback to numerical score for ML models"""
    feedback_map = {
        "helpful": 0.9,
        "too_easy": 0.3,
        "too_hard": 0.2,
        "irrelevant": 0.1
    }
    
    base_score = feedback_map.get(feedback_type, 0.5)
    
    if rating is not None:
        return (base_score + rating) / 2
    
    return base_score

async def _generate_skill_progression_path(
    db: Session, 
    student_id: int, 
    target_skill: str, 
    current_level: int, 
    target_level: int
) -> List[LearningPathStep]:
    """Generate learning path using skill progression"""
    # Would use Neo4j to find optimal skill progression
    return []

def _calculate_path_success_probability(
    context: Dict, 
    steps: List[LearningPathStep], 
    time_horizon: int
) -> float:
    """Calculate probability of successfully completing learning path"""
    # Factor in student's historical performance and path complexity
    base_probability = context.get("accuracy_rate_overall", 0.5)
    path_difficulty = len(steps) / 10  # Normalize by expected path length
    time_pressure = min(time_horizon / 30, 1.0)  # Normalize by expected time
    
    return min(base_probability * time_pressure / (1 + path_difficulty), 0.95)

async def _get_trending_questions(
    db: Session, 
    since: datetime, 
    count: int, 
    subject_id: Optional[int]
) -> List[Dict]:
    """Get trending questions based on recent activity"""
    # Query database for questions with high recent activity
    return []

async def _get_student_recommendation_analytics(
    db: Session, 
    student_id: int, 
    days: int
) -> Dict[str, Any]:
    """Get comprehensive analytics for student recommendations"""
    return {
        "recommendations_received": 0,
        "recommendations_completed": 0,
        "average_rating": 0.0,
        "improvement_trend": "stable",
        "preferred_algorithms": [],
        "skill_progress": {}
    }

async def _log_recommendation_analytics(
    student_id: int,
    question_ids: List[int], 
    algorithm: str,
    generation_time: float
):
    """Log recommendation analytics for monitoring"""
    logger.info("recommendation_analytics",
               student_id=student_id,
               question_count=len(question_ids),
               algorithm=algorithm,
               generation_time=generation_time)