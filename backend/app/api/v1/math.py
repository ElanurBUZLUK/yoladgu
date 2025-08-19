from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import logging

from app.core.database import get_async_session
from app.middleware.auth import get_current_student
from app.models.user import User
from app.models.question import Subject, QuestionType, Question
from app.services.math_selector import math_selector
from app.services.math_profile_manager import math_profile_manager
from app.schemas.question import (
    QuestionResponse, QuestionRecommendationResponse as QuestionRecResponse, 
    GetQuestionsByLevelResponse, GetQuestionPoolResponse, GetMathTopicsResponse,
    GetDifficultyDistributionResponse, GetRandomQuestionResponse
)
from sqlalchemy import select, and_, func, distinct

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/math/questions", tags=["math-questions"])


class QuestionRecommendationRequest(BaseModel):
    limit: int = Field(10, description="Number of questions to recommend")
    exclude_recent: bool = Field(True, description="Exclude recently answered questions")

class QuestionSearchResponse(BaseModel):
    questions: List[QuestionResponse]
    search_criteria: Dict[str, Any]
    total_count: int


@router.post("/recommend", response_model=QuestionRecResponse)
async def recommend_math_questions(
    request: QuestionRecommendationRequest,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Recommend math questions based on user level and preferences"""
    try:
        profile = await math_profile_manager.get_or_create_profile(db, current_user)
        
        recommended_questions, rationale = await math_selector.select_questions_for_recommendation(
            db, profile, request.limit, request.exclude_recent
        )
        
        return QuestionRecResponse(
            questions=[QuestionResponse.from_orm(q) for q in recommended_questions],
            recommendation_reason=rationale.get("reason", "Based on your current level and performance"),
            difficulty_adjustment=rationale.get("difficulty_adjustment"),
            total_available=rationale.get("total_available", 0),
            user_level=profile.global_skill,
            next_recommendations=[]
        )
    except Exception as e:
        logger.exception(f"Error in question recommendation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question recommendation failed: {str(e)}"
        )

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
