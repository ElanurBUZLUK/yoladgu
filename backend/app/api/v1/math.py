from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List

from app.core.database import get_async_session
from app.services.question_service import question_service
from app.schemas.question import (
    QuestionCreate, QuestionUpdate, QuestionResponse, 
    QuestionRecommendationRequest, QuestionRecommendationResponse,
    QuestionSearchQuery, QuestionStats, QuestionPool,
    QuestionDifficultyAdjustment, BulkQuestionOperation
)
from app.middleware.auth import (
    get_current_active_user, get_current_student, 
    get_current_teacher, get_current_admin
)
from app.models.user import User
from app.models.question import Subject, QuestionType, SourceType

router = APIRouter(prefix="/api/v1/math", tags=["mathematics"])


@router.post("/questions/recommend", response_model=QuestionRecommendationResponse)
async def recommend_math_questions(
    recommendation_request: QuestionRecommendationRequest,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Get recommended math questions based on user profile and preferences"""
    
    # Ensure subject is math
    recommendation_request.subject = Subject.MATH
    
    # Use user's current math level if not specified
    if not recommendation_request.user_level:
        recommendation_request.user_level = current_user.current_math_level
    
    # Use user's learning style if not specified
    if not recommendation_request.learning_style:
        recommendation_request.learning_style = current_user.learning_style.value
    
    return await question_service.recommend_questions(db, current_user, recommendation_request)


@router.get("/questions/search")
async def search_math_questions(
    query: Optional[str] = Query(None, description="Search term"),
    difficulty_level: Optional[int] = Query(None, ge=1, le=5, description="Difficulty level (1-5)"),
    question_type: Optional[QuestionType] = Query(None, description="Question type"),
    topic_category: Optional[str] = Query(None, description="Topic category"),
    source_type: Optional[SourceType] = Query(None, description="Source type"),
    min_difficulty: Optional[int] = Query(None, ge=1, le=5, description="Minimum difficulty"),
    max_difficulty: Optional[int] = Query(None, ge=1, le=5, description="Maximum difficulty"),
    skip: int = Query(0, ge=0, description="Number of questions to skip"),
    limit: int = Query(10, ge=1, le=100, description="Number of questions to return"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Search math questions with filters"""
    
    search_query = QuestionSearchQuery(
        query=query,
        subject=Subject.MATH,
        difficulty_level=difficulty_level,
        question_type=question_type,
        topic_category=topic_category,
        source_type=source_type,
        min_difficulty=min_difficulty,
        max_difficulty=max_difficulty
    )
    
    questions = await question_service.search_questions(db, search_query, skip, limit)
    
    # Convert to response format
    question_responses = [
        QuestionResponse(
            id=str(q.id),
            subject=q.subject,
            content=q.content,
            question_type=q.question_type,
            difficulty_level=q.difficulty_level,
            original_difficulty=q.original_difficulty,
            topic_category=q.topic_category,
            correct_answer=q.correct_answer,
            options=q.options,
            source_type=q.source_type,
            pdf_source_path=q.pdf_source_path,
            question_metadata=q.question_metadata,
            created_at=q.created_at.isoformat()
        )
        for q in questions
    ]
    
    return {
        "questions": question_responses,
        "total_found": len(question_responses),
        "skip": skip,
        "limit": limit,
        "search_criteria": search_query.dict(exclude_none=True)
    }


@router.get("/questions/pool", response_model=QuestionPool)
async def get_math_question_pool(
    difficulty_level: int = Query(..., ge=1, le=5, description="Difficulty level (1-5)"),
    topic_category: Optional[str] = Query(None, description="Topic category filter"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Get math question pool for specific difficulty level"""
    
    return await question_service.get_question_pool(db, Subject.MATH, difficulty_level, topic_category)


@router.get("/questions/stats", response_model=QuestionStats)
async def get_math_question_stats(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Get mathematics question statistics"""
    
    return await question_service.get_question_stats(db)


@router.get("/questions/by-level/{level}")
async def get_questions_by_level(
    level: int,
    limit: int = Query(10, ge=1, le=50, description="Number of questions to return"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Get math questions filtered by difficulty level"""
    
    if not (1 <= level <= 5):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Level must be between 1 and 5"
        )
    
    search_query = QuestionSearchQuery(
        subject=Subject.MATH,
        difficulty_level=level
    )
    
    questions = await question_service.search_questions(db, search_query, 0, limit)
    
    question_responses = [
        QuestionResponse(
            id=str(q.id),
            subject=q.subject,
            content=q.content,
            question_type=q.question_type,
            difficulty_level=q.difficulty_level,
            original_difficulty=q.original_difficulty,
            topic_category=q.topic_category,
            correct_answer=q.correct_answer,
            options=q.options,
            source_type=q.source_type,
            pdf_source_path=q.pdf_source_path,
            question_metadata=q.question_metadata,
            created_at=q.created_at.isoformat()
        )
        for q in questions
    ]
    
    return {
        "level": level,
        "questions": question_responses,
        "count": len(question_responses),
        "user_current_level": current_user.current_math_level
    }


@router.get("/questions/topics")
async def get_math_topics(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Get available math topics/categories"""
    
    # Get unique topic categories for math questions
    from sqlalchemy import select, distinct
    from app.models.question import Question
    
    result = await db.execute(
        select(distinct(Question.topic_category))
        .where(Question.subject == Subject.MATH)
        .order_by(Question.topic_category)
    )
    
    topics = [row[0] for row in result.fetchall()]
    
    # Group topics by difficulty level
    topic_by_level = {}
    for level in range(1, 6):
        level_result = await db.execute(
            select(distinct(Question.topic_category))
            .where(
                Question.subject == Subject.MATH,
                Question.difficulty_level == level
            )
            .order_by(Question.topic_category)
        )
        topic_by_level[str(level)] = [row[0] for row in level_result.fetchall()]
    
    return {
        "all_topics": topics,
        "topics_by_level": topic_by_level,
        "total_topics": len(topics)
    }


@router.get("/questions/difficulty-distribution")
async def get_difficulty_distribution(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Get math questions difficulty distribution"""
    
    from sqlalchemy import select, func
    from app.models.question import Question
    
    distribution = {}
    total_questions = 0
    
    for level in range(1, 6):
        result = await db.execute(
            select(func.count(Question.id))
            .where(
                Question.subject == Subject.MATH,
                Question.difficulty_level == level
            )
        )
        count = result.scalar() or 0
        distribution[str(level)] = count
        total_questions += count
    
    # Calculate percentages
    percentages = {}
    for level, count in distribution.items():
        percentages[level] = (count / total_questions * 100) if total_questions > 0 else 0
    
    return {
        "distribution": distribution,
        "percentages": percentages,
        "total_questions": total_questions,
        "user_level": current_user.current_math_level
    }


# Teacher/Admin endpoints for question management
@router.post("/questions/", response_model=QuestionResponse, status_code=status.HTTP_201_CREATED)
async def create_math_question(
    question_data: QuestionCreate,
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_async_session)
):
    """Create a new math question (Teacher/Admin only)"""
    
    # Ensure subject is math
    question_data.subject = Subject.MATH
    
    question = await question_service.create_question(db, question_data)
    
    return QuestionResponse(
        id=str(question.id),
        subject=question.subject,
        content=question.content,
        question_type=question.question_type,
        difficulty_level=question.difficulty_level,
        original_difficulty=question.original_difficulty,
        topic_category=question.topic_category,
        correct_answer=question.correct_answer,
        options=question.options,
        source_type=question.source_type,
        pdf_source_path=question.pdf_source_path,
        question_metadata=question.question_metadata,
        created_at=question.created_at.isoformat()
    )


@router.get("/questions/{question_id}", response_model=QuestionResponse)
async def get_math_question(
    question_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Get a specific math question by ID"""
    
    question = await question_service.get_question_by_id(db, question_id)
    
    if not question:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found"
        )
    
    if question.subject != Subject.MATH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question is not a math question"
        )
    
    return QuestionResponse(
        id=str(question.id),
        subject=question.subject,
        content=question.content,
        question_type=question.question_type,
        difficulty_level=question.difficulty_level,
        original_difficulty=question.original_difficulty,
        topic_category=question.topic_category,
        correct_answer=question.correct_answer,
        options=question.options,
        source_type=question.source_type,
        pdf_source_path=question.pdf_source_path,
        question_metadata=question.question_metadata,
        created_at=question.created_at.isoformat()
    )


@router.put("/questions/{question_id}", response_model=QuestionResponse)
async def update_math_question(
    question_id: str,
    question_update: QuestionUpdate,
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_async_session)
):
    """Update a math question (Teacher/Admin only)"""
    
    # Check if question exists and is math
    existing_question = await question_service.get_question_by_id(db, question_id)
    if not existing_question:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found"
        )
    
    if existing_question.subject != Subject.MATH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question is not a math question"
        )
    
    updated_question = await question_service.update_question(db, question_id, question_update)
    
    if not updated_question:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found"
        )
    
    return QuestionResponse(
        id=str(updated_question.id),
        subject=updated_question.subject,
        content=updated_question.content,
        question_type=updated_question.question_type,
        difficulty_level=updated_question.difficulty_level,
        original_difficulty=updated_question.original_difficulty,
        topic_category=updated_question.topic_category,
        correct_answer=updated_question.correct_answer,
        options=updated_question.options,
        source_type=updated_question.source_type,
        pdf_source_path=updated_question.pdf_source_path,
        question_metadata=updated_question.question_metadata,
        created_at=updated_question.created_at.isoformat()
    )


@router.delete("/questions/{question_id}")
async def delete_math_question(
    question_id: str,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_async_session)
):
    """Delete a math question (Admin only)"""
    
    # Check if question exists and is math
    existing_question = await question_service.get_question_by_id(db, question_id)
    if not existing_question:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found"
        )
    
    if existing_question.subject != Subject.MATH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question is not a math question"
        )
    
    success = await question_service.delete_question(db, question_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found"
        )
    
    return {"message": "Question deleted successfully", "question_id": question_id}


@router.put("/questions/{question_id}/difficulty", response_model=QuestionDifficultyAdjustment)
async def adjust_question_difficulty(
    question_id: str,
    new_difficulty: int = Query(..., ge=1, le=5, description="New difficulty level (1-5)"),
    reason: str = Query(..., description="Reason for adjustment"),
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_async_session)
):
    """Adjust math question difficulty level (Teacher/Admin only)"""
    
    # Check if question exists and is math
    existing_question = await question_service.get_question_by_id(db, question_id)
    if not existing_question:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found"
        )
    
    if existing_question.subject != Subject.MATH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question is not a math question"
        )
    
    return await question_service.adjust_question_difficulty(
        db, question_id, new_difficulty, reason, current_user.username
    )


@router.post("/questions/bulk-operation")
async def bulk_math_question_operation(
    operation_data: BulkQuestionOperation,
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_async_session)
):
    """Perform bulk operations on math questions (Admin only)"""
    
    # Verify all questions are math questions
    for question_id in operation_data.question_ids:
        question = await question_service.get_question_by_id(db, question_id)
        if question and question.subject != Subject.MATH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Question {question_id} is not a math question"
            )
    
    return await question_service.bulk_question_operation(db, operation_data)


@router.get("/questions/random/{difficulty_level}")
async def get_random_math_question(
    difficulty_level: int,
    topic_category: Optional[str] = Query(None, description="Filter by topic category"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Get a random math question for practice"""
    
    if not (1 <= difficulty_level <= 5):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Difficulty level must be between 1 and 5"
        )
    
    # Use recommendation system to get a single random question
    recommendation_request = QuestionRecommendationRequest(
        subject=Subject.MATH,
        user_level=difficulty_level,
        preferred_difficulty=difficulty_level,
        topic_categories=[topic_category] if topic_category else None,
        limit=1,
        exclude_recent=True,
        learning_style=current_user.learning_style.value
    )
    
    recommendations = await question_service.recommend_questions(db, current_user, recommendation_request)
    
    if not recommendations.questions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No questions found for the specified criteria"
        )
    
    return {
        "question": recommendations.questions[0],
        "difficulty_level": difficulty_level,
        "topic_category": topic_category,
        "recommendation_reason": recommendations.recommendation_reason
    }


@router.get("/health")
async def math_health():
    """Matematik modülü health check"""
    return {"status": "ok", "module": "mathematics"}