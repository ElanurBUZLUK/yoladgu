from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import logging
import hashlib
import json
import time # Added for idempotency key
from datetime import datetime # Added for session data

from app.database_enhanced import enhanced_database_manager as database_manager
from app.middleware.auth import get_current_student, get_current_teacher, get_current_student_mock, get_current_teacher_mock
from app.models.user import User
from app.models.question import Subject, QuestionType, Question, SourceType # Added SourceType
from app.services.math_recommend_service import math_recommend_service # NEW IMPORT
from app.services.math_profile_manager import math_profile_manager # Keep for other endpoints if needed
from app.schemas.question import (
    QuestionResponse, QuestionRecommendationResponse as QuestionRecResponse, 
    GetQuestionsByLevelResponse, GetQuestionPoolResponse, GetMathTopicsResponse,
    GetDifficultyDistributionResponse, GetRandomQuestionResponse
)
from app.utils.distlock_idem import idempotency_decorator, IdempotencyConfig
from app.core.error_handling import ErrorHandler, ErrorCode, ErrorSeverity
from sqlalchemy import select, and_, func, distinct
from app.services.cache_service import cache_service # Added for session data

logger = logging.getLogger(__name__)
error_handler = ErrorHandler()

router = APIRouter(prefix="/api/v1/math", tags=["math"])

class MathQuestion(BaseModel):
    id: str
    content: str
    options: List[str] = Field(min_items=3)
    correct_answer: str
    difficulty_level: float
    topic_category: Optional[str] = None

class GenerateMathQuestionResponse(BaseModel):
    success: bool
    question: Optional[MathQuestion] = None
    generation_info: Optional[dict] = None

class MathQuestionRequest(BaseModel):
    user_id: str
    k: int = Field(default=1, ge=1, le=10)

class QuestionRecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User ID for recommendation")
    limit: int = Field(10, description="Number of questions to recommend")
    exclude_recent: bool = Field(True, description="Exclude recently answered questions")

class JSONUploadResponse(BaseModel):
    success: bool
    message: str
    questions_imported: int
    questions_failed: int
    errors: List[str] = []

def _math_recommend_key_builder(*args, **kwargs) -> str:
    """Build idempotency key for math recommendation"""
    # Create a unique key based on user_id and timestamp
    user_id = kwargs.get('user_id', 'unknown')
    timestamp = int(time.time() / 60)  # Round to minute for idempotency
    return f"math_recommend_{user_id}_{timestamp}"

# Removed duplicate /recommend endpoints

@idempotency_decorator(
    key_builder=_math_recommend_key_builder,
    config=IdempotencyConfig(scope="math_recommendation", ttl_seconds=300)
)
async def _recommend_math_questions_internal(
    request: QuestionRecommendationRequest,
    current_user: User,
    db: AsyncSession
) -> QuestionRecResponse:
    """Internal function for math question recommendation with idempotency"""
    # Use the new MathRecommendService
    recommended_questions = await math_recommend_service.recommend_questions(
        user_id=request.user_id,
        session=db,
        num_questions=request.limit
    )
    
    # Fetch the user's math profile to get global_skill for the response
    profile = await math_profile_manager.get_or_create_profile(db, current_user)

    return QuestionRecResponse(
        questions=recommended_questions,
        recommendation_reason="Based on your current math skill level and recommended difficulty.",
        difficulty_adjustment=None, # Placeholder for now
        total_available=len(recommended_questions), # Placeholder for now
        user_level=profile.global_skill, # Use actual global skill
        next_recommendations=[] # Placeholder for now
    )

@router.post("/questions/generate", response_model=GenerateMathQuestionResponse)
async def generate_math_question(
    request: MathQuestionRequest,
    current_user: User = Depends(get_current_student_mock),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Generate Math questions with placement test and adaptive mode"""
    
    try:
        # Check if user is in placement test mode or adaptive mode
        session_key = f"math_session:{current_user.id}"
        session_data = await cache_service.get(session_key)
        
        if not session_data:
            # Initialize new session
            session_data = {
                "is_placement_test": True,
                "placement_progress": 0,
                "math_level": 3,  # Default math level
                "attempt_count": 0,
                "session_start": datetime.utcnow().isoformat()
            }
            await cache_service.set(session_key, session_data, expire=3600)
        
        # Check if placement test is complete
        if session_data.get("is_placement_test", True) and session_data.get("placement_progress", 0) >= 20:
            # Switch to adaptive mode
            session_data["is_placement_test"] = False
            session_data["math_level"] = session_data.get("math_level", 3)  # Default math level
            await cache_service.set(session_key, session_data, expire=3600)
        
        # Generate question based on mode
        if session_data.get("is_placement_test", True):
            # Placement test mode - use baseline questions
            question = await _generate_math_placement_question(db, current_user)
        else:
            # Adaptive mode - use math recommendation service
            question = await _generate_math_adaptive_question(db, current_user, session_data)
        
        if not question:
            return GenerateMathQuestionResponse(
                success=False,
                generation_info={"error": "no_question_generated"}
            )
        
        # Update session data
        session_data["attempt_count"] = session_data.get("attempt_count", 0) + 1
        if session_data.get("is_placement_test", True):
            session_data["placement_progress"] = session_data.get("placement_progress", 0) + 1
        await cache_service.set(session_key, session_data, expire=3600)
        
        # Invariant checks
        assert question["correct_answer"] in question["options"]
        assert len(set(question["options"])) == len(question["options"])
        
        return GenerateMathQuestionResponse(
            success=True,
            question=MathQuestion(**question),
            generation_info={
                "mode": "placement_test" if session_data.get("is_placement_test", True) else "adaptive",
                "math_level": session_data.get("math_level"),
                "placement_progress": session_data.get("placement_progress", 0),
                "attempt_count": session_data.get("attempt_count", 0)
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating Math question: {e}")
        return GenerateMathQuestionResponse(
            success=False,
            generation_info={"error": str(e)}
        )

async def _generate_math_placement_question(db: AsyncSession, user: User) -> Optional[Dict[str, Any]]:
    """Generate baseline question for math placement test"""
    try:
        # Get a random baseline question
        result = await db.execute(
            select(Question)
            .where(
                and_(
                    Question.subject == Subject.MATH,
                    Question.difficulty_level.between(1, 3),  # Start with moderate difficulty
                    Question.question_type == QuestionType.MULTIPLE_CHOICE
                )
            )
            .order_by(func.random())
            .limit(1)
        )
        
        question = result.scalar_one_or_none()
        if not question:
            return None
        
        return {
            "id": str(question.id),
            "content": question.content,
            "options": question.options if question.options else ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": question.correct_answer,
            "difficulty_level": question.difficulty_level,
            "topic_category": question.topic_category
        }
        
    except Exception as e:
        logger.error(f"Error generating math placement question: {e}")
        return None

async def _generate_math_adaptive_question(db: AsyncSession, user: User, session_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate adaptive question using math recommendation service"""
    try:
        # Use math recommendation service
        recommendation_result = await math_recommend_service.recommend_questions(
            user_id=str(user.id),
            session=db,
            num_questions=1,
            include_recent_errors=True,
            use_embeddings=True
        )
        
        if not recommendation_result.get("success") or not recommendation_result.get("recommendations"):
            # Fallback to traditional method
            return await _generate_math_placement_question(db, user)
        
        # Get the first recommended question
        recommendation = recommendation_result["recommendations"][0]
        
        # Get full question details from database
        result = await db.execute(
            select(Question)
            .where(Question.id == recommendation["id"])
        )
        
        question = result.scalar_one_or_none()
        if not question:
            return await _generate_math_placement_question(db, user)
        
        # Update session math level based on recommendation
        session_data["math_level"] = recommendation.get("difficulty_level", session_data.get("math_level", 3))
        
        return {
            "id": str(question.id),
            "content": question.content,
            "options": question.options if question.options else ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": question.correct_answer,
            "difficulty_level": question.difficulty_level,
            "topic_category": question.topic_category
        }
        
    except Exception as e:
        logger.error(f"Error generating math adaptive question: {e}")
        return await _generate_math_placement_question(db, user)

@router.post("/recommend", response_model=QuestionRecResponse)
async def recommend_math_questions(
    request: QuestionRecommendationRequest,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Recommend math questions based on user level and preferences with idempotency"""
    try:
        return await _recommend_math_questions_internal(request, current_user, db)
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.LLM_SERVICE_ERROR,
            message="Math question recommendation failed",
            severity=ErrorSeverity.MEDIUM,
            context={
                "user_id": str(current_user.id),
                "limit": request.limit,
                "exclude_recent": request.exclude_recent
            }
        )

class QuestionSearchResponse(BaseModel):
    questions: List[QuestionResponse]
    search_criteria: Dict[str, Any]
    total_count: int


@router.get("/questions/search", response_model=QuestionSearchResponse)
async def search_math_questions(
    difficulty_level: Optional[int] = Query(None, description="Difficulty level filter"),
    topic_category: Optional[str] = Query(None, description="Topic category filter"),
    question_type: Optional[str] = Query(None, description="Question type filter"),
    limit: int = Query(10, description="Number of questions to return"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Search math questions with filters"""
    try:
        conditions = [Question.subject == Subject.MATH, Question.is_active == True]
        if difficulty_level:
            conditions.append(Question.difficulty_level == difficulty_level)
        if topic_category:
            conditions.append(Question.topic_category == topic_category)
        if question_type:
            conditions.append(Question.question_type == QuestionType(question_type))
        
        stmt = select(Question).where(and_(*conditions)).limit(limit)
        result = await db.execute(stmt)
        questions = result.scalars().all()
        
        return QuestionSearchResponse(
            questions=[QuestionResponse.from_orm(q) for q in questions],
            search_criteria={
                "difficulty_level": difficulty_level,
                "topic_category": topic_category,
                "question_type": question_type
            },
            total_count=len(questions)
        )
    except Exception as e:
        logger.exception(f"Error in question search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question search failed: {str(e)}"
        )


@router.get("/questions/by-level/{level}", response_model=GetQuestionsByLevelResponse)
async def get_questions_by_level(
    level: int,
    limit: int = Query(10, description="Number of questions to return"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Get math questions by difficulty level"""
    try:
        profile = await math_profile_manager.get_or_create_profile(db, current_user)
        stmt = select(Question).where(
            and_(
                Question.subject == Subject.MATH,
                Question.difficulty_level == level,
                Question.is_active == True
            )
        ).limit(limit)
        
        result = await db.execute(stmt)
        questions = result.scalars().all()
        
        return GetQuestionsByLevelResponse(
            level=level,
            count=len(questions),
            questions=[QuestionResponse.from_orm(q) for q in questions],
            user_current_level=int(profile.global_skill)
        )
    except Exception as e:
        logger.exception(f"Error getting questions by level: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get questions by level: {str(e)}"
        )


@router.get("/questions/pool", response_model=GetQuestionPoolResponse)
async def get_question_pool(
    difficulty_level: Optional[int] = Query(None, description="Difficulty level filter"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Get question pool statistics"""
    try:
        conditions = [Question.subject == Subject.MATH, Question.is_active == True]
        if difficulty_level:
            conditions.append(Question.difficulty_level == difficulty_level)
        
        count_stmt = select(func.count(Question.id)).where(and_(*conditions))
        result = await db.execute(count_stmt)
        total_count = result.scalar()
        
        return GetQuestionPoolResponse(
            pool_size=total_count,
            subject="math",
            difficulty_level=difficulty_level,
            total_questions=total_count
        )
    except Exception as e:
        logger.exception(f"Error getting question pool: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get question pool: {str(e)}"
        )


@router.get("/questions/topics", response_model=GetMathTopicsResponse)
async def get_math_topics(
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Get all math topics organized by level"""
    try:
        stmt = select(distinct(Question.topic_category)).where(
            Question.subject == Subject.MATH,
            Question.is_active == True
        )
        result = await db.execute(stmt)
        all_topics = [row[0] for row in result.fetchall()]
        
        topics_by_level = {}
        for level in range(1, 6):
            stmt = select(distinct(Question.topic_category)).where(
                Question.subject == Subject.MATH,
                Question.difficulty_level == level,
                Question.is_active == True
            )
            result = await db.execute(stmt)
            topics_by_level[str(level)] = [row[0] for row in result.fetchall()]
    
        return GetMathTopicsResponse(
            total_topics=len(all_topics),
            all_topics=all_topics,
            topics_by_level=topics_by_level
        )
    except Exception as e:
        logger.exception(f"Error getting math topics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get math topics: {str(e)}"
        )


@router.get("/questions/difficulty-distribution", response_model=GetDifficultyDistributionResponse)
async def get_difficulty_distribution(
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Get difficulty distribution of math questions"""
    try:
        profile = await math_profile_manager.get_or_create_profile(db, current_user)
        distribution = {}
        for level in range(1, 6):
            stmt = select(func.count(Question.id)).where(
                and_(
                    Question.subject == Subject.MATH,
                    Question.difficulty_level == level,
                    Question.is_active == True
                )
            )
            result = await db.execute(stmt)
            count = result.scalar()
            distribution[str(level)] = count
        
        total_stmt = select(func.count(Question.id)).where(
            and_(
                Question.subject == Subject.MATH,
                Question.is_active == True
            )
        )
        result = await db.execute(total_stmt)
        total_questions = result.scalar()
    
        return GetDifficultyDistributionResponse(
            total_questions=total_questions,
            user_level=int(profile.global_skill),
            distribution=distribution
        )
    except Exception as e:
        logger.exception(f"Error getting difficulty distribution: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get difficulty distribution: {str(e)}"
        )


@router.get("/questions/random/{level}", response_model=GetRandomQuestionResponse)
async def get_random_question(
    level: int,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Get a random math question by level"""
    try:
        stmt = select(Question).where(
            and_(
                Question.subject == Subject.MATH,
                Question.difficulty_level == level,
                Question.is_active == True
            )
        ).order_by(func.random()).limit(1)
        
        result = await db.execute(stmt)
        question = result.scalar_one_or_none()
    
        if not question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No questions found for level {level}"
            )
        
        return GetRandomQuestionResponse(
            question=QuestionResponse.from_orm(question),
            difficulty_level=level
        )
    except Exception as e:
        logger.exception(f"Error getting random question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get random question: {str(e)}"
        )


@router.get("/questions/{question_id}", response_model=QuestionResponse)
async def get_question_by_id(
    question_id: str,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Get a specific math question by ID"""
    try:
        stmt = select(Question).where(
            and_(
                Question.id == question_id,
                Question.subject == Subject.MATH
            )
        )
        
        result = await db.execute(stmt)
        question = result.scalar_one_or_none()
        
        if not question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Question not found"
            )
    
        return QuestionResponse.from_orm(question)
    except Exception as e:
        logger.exception(f"Error getting question by ID: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get question: {str(e)}"
        )


@router.post("/questions/upload-json", response_model=JSONUploadResponse)
async def upload_math_questions_json(
    file: UploadFile = File(..., description="JSON file containing math questions"),
    current_user: User = Depends(get_current_teacher),  # Only teachers can upload
    db: AsyncSession = Depends(database_manager.get_session)
):
    """
    Upload math questions from a JSON file using enhanced format
    
    Expected JSON format:
    [
        {
            "stem": "What is 2 + 2?",
            "options": {
                "A": "1",
                "B": "2",
                "C": "3",
                "D": "4"
            },
            "correct_answer": "D",
            "topic": "arithmetic",
            "subtopic": "addition",
            "difficulty": 0.5,
            "source": "seed",
            "metadata": {
                "estimated_time": 30,
                "learning_objectives": ["basic addition"],
                "tags": ["arithmetic", "basic"],
                "cefr_level": "A1"
            }
        }
    ]
    """
    try:
        # Validate file type
        if not file.filename.endswith('.json'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only JSON files are allowed"
            )
        
        # Read and parse JSON
        content = await file.read()
        try:
            questions_data = json.loads(content.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON format: {str(e)}"
            )
        
        if not isinstance(questions_data, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="JSON must contain an array of questions"
            )
        
        # Process questions
        questions_imported = 0
        questions_failed = 0
        errors = []
        
        for i, question_data in enumerate(questions_data):
            try:
                # Validate required fields for enhanced format
                required_fields = ['stem', 'options', 'correct_answer', 'topic']
                for field in required_fields:
                    if field not in question_data:
                        raise ValueError(f"Missing required field: {field}")
                
                # Convert difficulty (continuous 0.0-2.0 to discrete 1-5)
                difficulty = question_data.get('difficulty', 1.0)
                if difficulty <= 0.5:
                    difficulty_level = 1
                elif difficulty <= 1.0:
                    difficulty_level = 2
                elif difficulty <= 1.5:
                    difficulty_level = 3
                elif difficulty <= 1.8:
                    difficulty_level = 4
                else:
                    difficulty_level = 5
                
                # Determine question type based on options
                options = question_data['options']
                if len(options) == 2 and all(opt in ['True', 'False', 'true', 'false'] for opt in options.values()):
                    question_type = QuestionType.TRUE_FALSE
                elif len(options) == 0:
                    question_type = QuestionType.OPEN_ENDED
                else:
                    question_type = QuestionType.MULTIPLE_CHOICE
                
                # Create question object with enhanced format
                question = Question(
                    subject=Subject.MATH,
                    content=question_data['stem'],
                    question_type=question_type,
                    difficulty_level=difficulty_level,
                    original_difficulty=difficulty_level,
                    topic_category=question_data['topic'],
                    correct_answer=question_data['correct_answer'],
                    options=question_data['options'],
                    source_type=SourceType.MANUAL,
                    estimated_difficulty=question_data.get('difficulty', 1.0),
                    question_metadata={
                        "subtopic": question_data.get('subtopic'),
                        "source": question_data.get('source', 'seed'),
                        "estimated_time": question_data.get('metadata', {}).get('estimated_time', 60),
                        "learning_objectives": question_data.get('metadata', {}).get('learning_objectives', []),
                        "tags": question_data.get('metadata', {}).get('tags', []),
                        "cefr_level": question_data.get('metadata', {}).get('cefr_level', 'A1')
                    },
                    created_by=str(current_user.id)
                )
                
                db.add(question)
                questions_imported += 1
                
            except Exception as e:
                questions_failed += 1
                error_msg = f"Question {i+1}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        # Commit all successful questions
        await db.commit()
        
        return JSONUploadResponse(
            success=True,
            message=f"Successfully imported {questions_imported} questions",
            questions_imported=questions_imported,
            questions_failed=questions_failed,
            errors=errors
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error uploading JSON questions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload questions: {str(e)}"
        )
