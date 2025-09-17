"""
Math Recommendation API Endpoints
IRT + Multi-Skill Elo Integration for Adaptive Math Learning
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import structlog

from app.db.session import get_db
from ml.math.irt import irt_model, ItemParams, StudentAbility
from ml.math.multiskill_elo import multiskill_elo
from ml.math.selector import math_selector

logger = structlog.get_logger()
router = APIRouter()

class NextQuestionRequest(BaseModel):
    """Request for next question recommendation"""
    user_id: str = Field(..., description="User ID")
    avoid_recent: bool = Field(True, description="Avoid recently attempted questions")
    max_recent_count: int = Field(10, description="Maximum recent questions to avoid")

class AnswerEventRequest(BaseModel):
    """Request for answer event processing"""
    user_id: str = Field(..., description="User ID")
    item_id: str = Field(..., description="Question ID")
    correct: bool = Field(..., description="Whether answer was correct")
    response_time: Optional[float] = Field(None, description="Response time in seconds")
    skills: Optional[List[str]] = Field(None, description="Skills involved in the question")
    skill_weights: Optional[Dict[str, float]] = Field(None, description="Skill importance weights")

class NextQuestionResponse(BaseModel):
    """Response for next question recommendation"""
    question_id: str
    question_data: Dict[str, Any]
    expected_probability: float
    learning_gain: float
    difficulty_level: str
    required_skills: List[str]
    recommendation_reason: str
    timestamp: str

class AnswerEventResponse(BaseModel):
    """Response for answer event processing"""
    user_id: str
    item_id: str
    expected_score: float
    actual_score: float
    rating_changes: Dict[str, float]
    updated_abilities: Dict[str, float]
    timestamp: str

class UserStatsResponse(BaseModel):
    """Response for user statistics"""
    user_id: str
    overall_rating: float
    skill_ratings: Dict[str, float]
    skill_masteries: Dict[str, float]
    total_questions_attempted: int
    recent_accuracy: float
    recommendation_summary: Dict[str, Any]
    timestamp: str

# Sample questions database (in production, this would come from a real database)
SAMPLE_QUESTIONS = [
    {
        "id": "math_001",
        "type": "algebra",
        "question": "Solve for x: 2x + 5 = 13",
        "options": ["x = 4", "x = 3", "x = 5", "x = 6"],
        "correct_answer": 0,
        "required_skills": ["linear_equations", "algebra"],
        "skill_weights": {"linear_equations": 0.7, "algebra": 0.3},
        "difficulty_level": "beginner",
        "irt_params": {
            "item_id": "math_001",
            "a": 1.2,
            "b": -0.5,
            "c": 0.0,
            "skill_weights": {"linear_equations": 0.7, "algebra": 0.3}
        }
    },
    {
        "id": "math_002",
        "type": "geometry",
        "question": "Find the area of a circle with radius 7 cm",
        "options": ["154 cm²", "44 cm²", "22 cm²", "77 cm²"],
        "correct_answer": 0,
        "required_skills": ["circle_area", "geometry"],
        "skill_weights": {"circle_area": 0.8, "geometry": 0.2},
        "difficulty_level": "intermediate",
        "irt_params": {
            "item_id": "math_002",
            "a": 1.0,
            "b": 0.2,
            "c": 0.0,
            "skill_weights": {"circle_area": 0.8, "geometry": 0.2}
        }
    },
    {
        "id": "math_003",
        "type": "calculus",
        "question": "Find the derivative of f(x) = x³ + 2x² - 5x + 1",
        "options": ["3x² + 4x - 5", "3x² + 4x + 5", "x² + 4x - 5", "3x² - 4x - 5"],
        "correct_answer": 0,
        "required_skills": ["derivatives", "polynomials", "calculus"],
        "skill_weights": {"derivatives": 0.6, "polynomials": 0.3, "calculus": 0.1},
        "difficulty_level": "advanced",
        "irt_params": {
            "item_id": "math_003",
            "a": 1.5,
            "b": 1.2,
            "c": 0.0,
            "skill_weights": {"derivatives": 0.6, "polynomials": 0.3, "calculus": 0.1}
        }
    },
    {
        "id": "math_004",
        "type": "statistics",
        "question": "What is the mean of the numbers: 2, 4, 6, 8, 10?",
        "options": ["6", "5", "7", "8"],
        "correct_answer": 0,
        "required_skills": ["mean", "statistics"],
        "skill_weights": {"mean": 0.9, "statistics": 0.1},
        "difficulty_level": "beginner",
        "irt_params": {
            "item_id": "math_004",
            "a": 1.1,
            "b": -1.0,
            "c": 0.0,
            "skill_weights": {"mean": 0.9, "statistics": 0.1}
        }
    },
    {
        "id": "math_005",
        "type": "trigonometry",
        "question": "What is sin(30°)?",
        "options": ["1/2", "√3/2", "1", "0"],
        "correct_answer": 0,
        "required_skills": ["trigonometry", "special_angles"],
        "skill_weights": {"trigonometry": 0.4, "special_angles": 0.6},
        "difficulty_level": "intermediate",
        "irt_params": {
            "item_id": "math_005",
            "a": 1.3,
            "b": 0.5,
            "c": 0.0,
            "skill_weights": {"trigonometry": 0.4, "special_angles": 0.6}
        }
    }
]

@router.post("/next", response_model=NextQuestionResponse)
async def get_next_question(
    request: NextQuestionRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Get the next optimal math question for a user
    Uses IRT + Multi-Skill Elo for intelligent question selection
    """
    try:
        logger.info("Getting next question", user_id=request.user_id)
        
        # Ensure user exists in both systems
        if request.user_id not in irt_model.students:
            # Initialize new student
            student = StudentAbility(user_id=request.user_id, theta=0.0)
            irt_model.add_student(student)
            
            # Initialize Elo skills
            for question in SAMPLE_QUESTIONS:
                for skill in question["required_skills"]:
                    multiskill_elo.add_user_skill(request.user_id, skill)
        
        # Ensure items exist in Elo system
        for question in SAMPLE_QUESTIONS:
            if question["id"] not in multiskill_elo.item_ratings:
                multiskill_elo.add_item(
                    question["id"], 
                    question["required_skills"]
                )
            
            # Ensure item exists in IRT model
            if question["id"] not in irt_model.items:
                item_params = ItemParams.from_dict(question["irt_params"])
                irt_model.add_item(item_params)
        
        # Get recent questions (simplified - in production, query database)
        recent_questions = []  # TODO: Implement recent questions tracking
        
        # Select next question
        selected_question = math_selector.select_next_question(
            user_id=request.user_id,
            available_questions=SAMPLE_QUESTIONS,
            recent_questions=recent_questions,
            avoid_recent=request.avoid_recent
        )
        
        if not selected_question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No suitable questions found for user"
            )
        
        # Calculate metrics for response
        item_params = ItemParams.from_dict(selected_question["irt_params"])
        expected_prob = math_selector.calculate_expected_probability(
            request.user_id, 
            selected_question["id"], 
            item_params, 
            selected_question["required_skills"]
        )
        
        learning_gain = math_selector.calculate_learning_gain(
            request.user_id,
            item_params,
            selected_question["required_skills"],
            selected_question["skill_weights"]
        )
        
        return NextQuestionResponse(
            question_id=selected_question["id"],
            question_data=selected_question,
            expected_probability=expected_prob,
            learning_gain=learning_gain,
            difficulty_level=selected_question["difficulty_level"],
            required_skills=selected_question["required_skills"],
            recommendation_reason=f"Selected based on expected probability {expected_prob:.2f} and learning gain {learning_gain:.3f}",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Failed to get next question", user_id=request.user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get next question: {str(e)}"
        )

@router.post("/answer", response_model=AnswerEventResponse)
async def process_answer_event(
    request: AnswerEventRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Process an answer event and update user/item ratings
    Updates both IRT and Elo systems
    """
    try:
        logger.info("Processing answer event", 
                   user_id=request.user_id, 
                   item_id=request.item_id, 
                   correct=request.correct)
        
        # Ensure user and item exist
        if request.user_id not in irt_model.students:
            student = StudentAbility(user_id=request.user_id, theta=0.0)
            irt_model.add_student(student)
        
        # Add response to IRT model
        irt_model.add_response(
            user_id=request.user_id,
            item_id=request.item_id,
            correct=request.correct,
            response_time=request.response_time,
            skills=request.skills
        )
        
        # Update Elo ratings
        expected_score, actual_score = multiskill_elo.update_ratings(
            user_id=request.user_id,
            item_id=request.item_id,
            correct=request.correct,
            response_time=request.response_time,
            skill_weights=request.skill_weights
        )
        
        # Get updated ratings
        user_stats = multiskill_elo.get_user_stats(request.user_id)
        
        # Calculate rating changes (simplified)
        rating_changes = {}
        updated_abilities = {}
        
        if request.skills:
            for skill in request.skills:
                current_rating = multiskill_elo.get_user_skill_rating(request.user_id, skill)
                rating_changes[skill] = current_rating - multiskill_elo.initial_rating
                updated_abilities[skill] = multiskill_elo.get_skill_mastery(request.user_id, skill)
        
        # Update IRT parameters (simplified - in production, do batch calibration)
        # For now, just update student ability
        if request.user_id in irt_model.students:
            user_responses = [r for r in irt_model.responses if r["user_id"] == request.user_id]
            if user_responses:
                new_theta = irt_model.estimate_ability(request.user_id, user_responses)
                irt_model.students[request.user_id].theta = new_theta
                updated_abilities["overall_theta"] = new_theta
        
        return AnswerEventResponse(
            user_id=request.user_id,
            item_id=request.item_id,
            expected_score=expected_score,
            actual_score=actual_score,
            rating_changes=rating_changes,
            updated_abilities=updated_abilities,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Failed to process answer event", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process answer event: {str(e)}"
        )

@router.get("/stats/{user_id}", response_model=UserStatsResponse)
async def get_user_stats(user_id: str, db: AsyncSession = Depends(get_db)):
    """
    Get comprehensive user statistics and recommendations
    """
    try:
        logger.info("Getting user stats", user_id=user_id)
        
        # Get Elo stats
        elo_stats = multiskill_elo.get_user_stats(user_id)
        
        # Get IRT stats
        irt_stats = {}
        if user_id in irt_model.students:
            student = irt_model.students[user_id]
            irt_stats = {
                "theta": student.theta,
                "skill_thetas": student.skill_thetas
            }
        
        # Get recommendation summary
        recommendation_summary = math_selector.get_user_recommendation_summary(user_id)
        
        # Calculate recent accuracy (simplified)
        recent_responses = [r for r in irt_model.responses if r["user_id"] == user_id]
        recent_accuracy = 0.0
        if recent_responses:
            recent_correct = sum(1 for r in recent_responses[-10:] if r["correct"])
            recent_accuracy = recent_correct / min(10, len(recent_responses))
        
        return UserStatsResponse(
            user_id=user_id,
            overall_rating=elo_stats.get("overall_rating", multiskill_elo.initial_rating),
            skill_ratings={skill: data["rating"] for skill, data in elo_stats.get("skills", {}).items()},
            skill_masteries={skill: data["mastery"] for skill, data in elo_stats.get("skills", {}).items()},
            total_questions_attempted=len(recent_responses),
            recent_accuracy=recent_accuracy,
            recommendation_summary=recommendation_summary,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Failed to get user stats", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user stats: {str(e)}"
        )

@router.post("/irt/calibrate")
async def calibrate_irt_model(
    days: int = 30,
    db: AsyncSession = Depends(get_db)
):
    """
    Trigger IRT model batch calibration (Protected endpoint)
    
    Args:
        days: Number of days to look back for training data
        
    Returns:
        Calibration job status
    """
    try:
        logger.info("Starting IRT calibration job", days=days)
        
        # Import calibrator
        from jobs.irt_batch_calibrate import irt_calibrator
        
        # Run calibration in background
        results = await irt_calibrator.run_calibration(days=days)
        
        if results.get("success"):
            return {
                "status": "completed",
                "message": "IRT model calibrated successfully",
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Calibration failed: {results.get('error', 'Unknown error')}"
            )
        
    except Exception as e:
        logger.error("IRT calibration endpoint failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Calibration failed: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check for math recommendation system"""
    try:
        return {
            "status": "healthy",
            "system": "Math Recommendation System",
            "components": {
                "irt_model": {
                    "model_type": irt_model.model_type,
                    "items_count": len(irt_model.items),
                    "students_count": len(irt_model.students),
                    "responses_count": len(irt_model.responses)
                },
                "elo_system": {
                    "users_count": len(multiskill_elo.user_skills),
                    "items_count": len(multiskill_elo.item_ratings),
                    "initial_rating": multiskill_elo.initial_rating
                },
                "selector": {
                    "target_probability_range": (math_selector.target_prob_low, math_selector.target_prob_high),
                    "learning_gain_weight": math_selector.learning_gain_weight
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
