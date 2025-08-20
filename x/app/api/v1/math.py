from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import logging
import hashlib
import json

from app.core.database import get_async_session
from app.middleware.auth import get_current_student
from app.models.user import User
from app.models.question import Subject, QuestionType, Question
from app.services.math_recommend_service import math_recommend_service # NEW IMPORT
from app.services.math_profile_manager import math_profile_manager # Keep for other endpoints if needed
from app.schemas.question import (
    QuestionResponse, QuestionRecommendationResponse as QuestionRecResponse, 
    GetQuestionsByLevelResponse, GetQuestionPoolResponse, GetMathTopicsResponse,
    GetDifficultyDistributionResponse, GetRandomQuestionResponse
)
from app.utils.distlock_idem import idempotency_decorator, IdempotencyConfig
from sqlalchemy import select, and_, func, distinct

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/math/questions", tags=["math-questions"])

# New router for /api/v1/math endpoints
recommend_router = APIRouter(prefix="/api/v1/math", tags=["math-recommendation"])

class RecommendRequest(BaseModel):
    # Assuming a simple request for now, adjust as needed
    user_id: str
    # Add other fields relevant to math recommendation, e.g., current_topic, difficulty_level
    # For example:
    # current_topic: Optional[str] = None
    # desired_difficulty: Optional[int] = None

class QuestionRecommendationRequest(BaseModel):
    limit: int = Field(10, description="Number of questions to recommend")
    exclude_recent: bool = Field(True, description="Exclude recently answered questions")

def _math_recommend_key_builder(*args, **kwargs) -> str:
    """Build idempotency key for math recommendation"""
    # Extract user_id and limit from request
    request = None
    for arg in args:
        if isinstance(arg, QuestionRecommendationRequest):
            request = arg
            break
    
    if not request:
        # Try to get from kwargs
        request = kwargs.get('request')
    
    if not request:
        return "math_recommend:default"
    
    # Create deterministic key based on request parameters
    key_data = {
        "user_id": getattr(request, 'user_id', 'unknown'),
        "limit": getattr(request, 'limit', 10),
        "exclude_recent": getattr(request, 'exclude_recent', True)
    }
    
    key_string = json.dumps(key_data, sort_keys=True)
    return f"math_recommend:{hashlib.md5(key_string.encode()).hexdigest()}"

@recommend_router.post("/recommend")
async def recommend_math(body: RecommendRequest, svc = Depends(lambda: math_recommend_service)):
    """
    Recommends math content based on the request body.
    """
    try:
        recommended_content = await svc.recommend(body.dict())
        return recommended_content
    except Exception as e:
        logger.exception(f"Error in math recommendation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Math recommendation failed: {str(e)}"
        )

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
        user_id=current_user.id,
        session=db,
        limit=request.limit
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

@router.post("/recommend", response_model=QuestionRecResponse)
async def recommend_math_questions(
    request: QuestionRecommendationRequest,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Recommend math questions based on user level and preferences with idempotency"""
    try:
        return await _recommend_math_questions_internal(request, current_user, db)
    except Exception as e:
        logger.exception(f"Error in question recommendation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question recommendation failed: {str(e)}"
        )

class QuestionSearchResponse(BaseModel):
    questions: List[QuestionResponse]
    search_criteria: Dict[str, Any]
    total_count: int


@router.get("/search", response_model=QuestionSearchResponse)
async def search_math_questions(
    difficulty_level: Optional[int] = Query(None, description="Difficulty level filter"),
    topic_category: Optional[str] = Query(None, description="Topic category filter"),
    question_type: Optional[str] = Query(None, description="Question type filter"),
    limit: int = Query(10, description="Number of questions to return"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
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


@router.get("/by-level/{level}", response_model=GetQuestionsByLevelResponse)
async def get_questions_by_level(
    level: int,
    limit: int = Query(10, description="Number of questions to return"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
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


@router.get("/pool", response_model=GetQuestionPoolResponse)
async def get_question_pool(
    difficulty_level: Optional[int] = Query(None, description="Difficulty level filter"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
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


@router.get("/topics", response_model=GetMathTopicsResponse)
async def get_math_topics(
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
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


@router.get("/difficulty-distribution", response_model=GetDifficultyDistributionResponse)
async def get_difficulty_distribution(
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
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


@router.get("/random/{level}", response_model=GetRandomQuestionResponse)
async def get_random_question(
    level: int,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
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


@router.get("/{question_id}", response_model=QuestionResponse)
async def get_question_by_id(
    question_id: str,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
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
