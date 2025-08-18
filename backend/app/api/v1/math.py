from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import logging
import time
from datetime import datetime

from app.core.database import get_async_session
from app.middleware.auth import get_current_student, get_current_teacher
from app.models.user import User
from app.models.question import Subject, QuestionType
from app.services.math_selector import math_selector
from app.services.math_profile_manager import math_profile_manager
from app.core.cache import cache_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/math/questions", tags=["math-questions"])


# Request/Response Models
class QuestionRecommendationRequest(BaseModel):
    subject: str = Field(..., description="Subject (math)")
    user_level: int = Field(..., description="User's current level")
    preferred_difficulty: int = Field(..., description="Preferred difficulty level")
    limit: int = Field(10, description="Number of questions to recommend")
    exclude_recent: bool = Field(True, description="Exclude recently answered questions")
    learning_style: str = Field("mixed", description="Learning style preference")


class QuestionRecommendationResponse(BaseModel):
    questions: List[Dict[str, Any]]
    recommendation_reason: str
    user_level: int
    difficulty_range: Dict[str, int]


class QuestionSearchRequest(BaseModel):
    difficulty_level: Optional[int] = Field(None, description="Difficulty level filter")
    topic_category: Optional[str] = Field(None, description="Topic category filter")
    question_type: Optional[str] = Field(None, description="Question type filter")
    limit: int = Field(10, description="Number of questions to return")


class QuestionSearchResponse(BaseModel):
    questions: List[Dict[str, Any]]
    search_criteria: Dict[str, Any]
    total_count: int


@router.post("/recommend", response_model=QuestionRecommendationResponse)
async def recommend_math_questions(
    request: QuestionRecommendationRequest,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Recommend math questions based on user level and preferences"""
    
    try:
        # Get user profile
        profile = await math_profile_manager.get_or_create_profile(db, current_user)
        
        # Get recommended questions using math selector
        recommended_questions, rationale = await math_selector.select_questions_for_recommendation(
            db, profile, request.limit, request.exclude_recent
        )
        
        # Format questions
        formatted_questions = []
        for question in recommended_questions:
            formatted_questions.append({
                "id": str(question.id),
                "content": question.content,
                "question_type": question.question_type.value,
                "difficulty_level": question.difficulty_level,
                "topic_category": question.topic_category,
                "options": question.options
            })
        
        return QuestionRecommendationResponse(
            questions=formatted_questions,
            recommendation_reason=rationale.get("reason", "Based on your current level and performance"),
            user_level=profile.global_skill,
            difficulty_range={
                "min": max(1, int(profile.global_skill - 1)),
                "max": min(5, int(profile.global_skill + 1))
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error in question recommendation: {e}")
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
        # Build search criteria
        search_criteria = {
            "subject": Subject.MATH,
            "is_active": True
        }
        
        if difficulty_level:
            search_criteria["difficulty_level"] = difficulty_level
        if topic_category:
            search_criteria["topic_category"] = topic_category
        if question_type:
            search_criteria["question_type"] = QuestionType(question_type)
        
        # Search questions
        from app.models.question import Question
        from sqlalchemy import select, and_
        
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
        
        # Format questions
        formatted_questions = []
        for question in questions:
            formatted_questions.append({
                "id": str(question.id),
                "content": question.content,
                "question_type": question.question_type.value,
                "difficulty_level": question.difficulty_level,
                "topic_category": question.topic_category,
                "options": question.options
            })
        
        return QuestionSearchResponse(
            questions=formatted_questions,
            search_criteria=search_criteria,
            total_count=len(formatted_questions)
        )
        
    except Exception as e:
        logger.error(f"❌ Error in question search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question search failed: {str(e)}"
        )


@router.get("/by-level/{level}")
async def get_questions_by_level(
    level: int,
    limit: int = Query(10, description="Number of questions to return"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Get math questions by difficulty level"""
    
    try:
        # Get user profile for current level
        profile = await math_profile_manager.get_or_create_profile(db, current_user)
        
        # Get questions by level
        from app.models.question import Question
        from sqlalchemy import select, and_
        
        stmt = select(Question).where(
            and_(
                Question.subject == Subject.MATH,
                Question.difficulty_level == level,
                Question.is_active == True
            )
        ).limit(limit)
        
        result = await db.execute(stmt)
        questions = result.scalars().all()
        
        # Format questions
        formatted_questions = []
        for question in questions:
            formatted_questions.append({
                "id": str(question.id),
                "content": question.content,
                "question_type": question.question_type.value,
                "difficulty_level": question.difficulty_level,
                "topic_category": question.topic_category,
                "options": question.options
            })
        
        return {
            "level": level,
            "count": len(formatted_questions),
            "questions": formatted_questions,
            "user_current_level": int(profile.global_skill)
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting questions by level: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get questions by level: {str(e)}"
        )


@router.get("/pool")
async def get_question_pool(
    difficulty_level: Optional[int] = Query(None, description="Difficulty level filter"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Get question pool statistics"""
    
    try:
        from app.models.question import Question
        from sqlalchemy import select, func, and_
        
        # Build query
        conditions = [Question.subject == Subject.MATH, Question.is_active == True]
        if difficulty_level:
            conditions.append(Question.difficulty_level == difficulty_level)
        
        # Get total count
        count_stmt = select(func.count(Question.id)).where(and_(*conditions))
        result = await db.execute(count_stmt)
        total_count = result.scalar()
        
        return {
            "pool_size": total_count,
            "subject": "math",
            "difficulty_level": difficulty_level,
            "total_questions": total_count
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting question pool: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get question pool: {str(e)}"
        )


@router.get("/topics")
async def get_math_topics(
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Get all math topics organized by level"""
    
    try:
    from app.models.question import Question
        from sqlalchemy import select, func, distinct
        
        # Get all unique topics
        stmt = select(distinct(Question.topic_category)).where(
            Question.subject == Subject.MATH,
            Question.is_active == True
        )
        result = await db.execute(stmt)
        all_topics = [row[0] for row in result.fetchall()]
        
        # Get topics by level
        topics_by_level = {}
    for level in range(1, 6):
            stmt = select(distinct(Question.topic_category)).where(
                Question.subject == Subject.MATH,
                Question.difficulty_level == level,
                Question.is_active == True
            )
            result = await db.execute(stmt)
            topics_by_level[str(level)] = [row[0] for row in result.fetchall()]
    
    return {
            "total_topics": len(all_topics),
            "all_topics": all_topics,
            "topics_by_level": topics_by_level
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting math topics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get math topics: {str(e)}"
        )


@router.get("/difficulty-distribution")
async def get_difficulty_distribution(
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Get difficulty distribution of math questions"""
    
    try:
        # Get user profile
        profile = await math_profile_manager.get_or_create_profile(db, current_user)
        
    from app.models.question import Question
        from sqlalchemy import select, func, and_
    
        # Get distribution by difficulty level
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
        
        # Get total count
        total_stmt = select(func.count(Question.id)).where(
            and_(
                Question.subject == Subject.MATH,
                Question.is_active == True
            )
        )
        result = await db.execute(total_stmt)
        total_questions = result.scalar()
    
    return {
        "total_questions": total_questions,
            "user_level": int(profile.global_skill),
            "distribution": distribution
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting difficulty distribution: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get difficulty distribution: {str(e)}"
        )


@router.get("/random/{level}")
async def get_random_question(
    level: int,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Get a random math question by level"""
    
    try:
        from app.models.question import Question
        from sqlalchemy import select, and_, func
        
        # Get random question
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
        
        return {
            "question": {
                "id": str(question.id),
                "content": question.content,
                "question_type": question.question_type.value,
                "difficulty_level": question.difficulty_level,
                "topic_category": question.topic_category,
                "options": question.options
            },
            "difficulty_level": level
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting random question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get random question: {str(e)}"
        )


@router.get("/stats")
async def get_question_statistics(
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Get math question statistics"""
    
    try:
        from app.models.question import Question
        from sqlalchemy import select, func, and_
        
        # Get total questions
        total_stmt = select(func.count(Question.id)).where(
            and_(
                Question.subject == Subject.MATH,
                Question.is_active == True
            )
        )
        result = await db.execute(total_stmt)
        total_questions = result.scalar()
        
        # Get average difficulty
        avg_stmt = select(func.avg(Question.difficulty_level)).where(
            and_(
                Question.subject == Subject.MATH,
                Question.is_active == True
            )
        )
        result = await db.execute(avg_stmt)
        average_difficulty = result.scalar() or 0
        
        # Get questions by type
        by_type = {}
        for qtype in QuestionType:
            stmt = select(func.count(Question.id)).where(
                and_(
                    Question.subject == Subject.MATH,
                    Question.question_type == qtype,
                    Question.is_active == True
                )
            )
            result = await db.execute(stmt)
            count = result.scalar()
            by_type[qtype.value] = count
        
        # Get questions by subject (only math for now)
        by_subject = {"math": total_questions}
        
        return {
            "total_questions": total_questions,
            "average_difficulty": float(average_difficulty),
            "by_subject": by_subject,
            "by_type": by_type
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting question statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get question statistics: {str(e)}"
        )


@router.get("/{question_id}")
async def get_question_by_id(
    question_id: str,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Get a specific math question by ID"""
    
    try:
        from app.models.question import Question
        from sqlalchemy import select, and_
        
        stmt = select(Question).where(
            and_(
                Question.id == question_id,
                Question.subject == Subject.MATH,
                Question.is_active == True
            )
        )
        
        result = await db.execute(stmt)
        question = result.scalar_one_or_none()
        
        if not question:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found"
        )
    
        return {
            "id": str(question.id),
            "content": question.content,
            "question_type": question.question_type.value,
            "difficulty_level": question.difficulty_level,
            "topic_category": question.topic_category,
            "options": question.options,
            "correct_answer": question.correct_answer
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting question by ID: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get question: {str(e)}"
        )