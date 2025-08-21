from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field

from app.database_enhanced import enhanced_database_manager
from app.middleware.auth import get_current_student
from app.models.user import User
from app.models.question import Subject

router = APIRouter(prefix="/api/v1/progress", tags=["progress"])

@router.get("/test")
async def test_progress():
    """Test endpoint for progress router"""
    return {"message": "Progress router is working!", "status": "ok"}

# Progress Save Request Model
class QuizResult(BaseModel):
    userId: int = Field(..., description="User ID")
    subject: str = Field(..., description="Subject (math/english)")
    score: float = Field(..., ge=0, le=100, description="Score percentage")
    totalQuestions: int = Field(..., ge=1, description="Total questions in quiz")
    correctAnswers: int = Field(..., ge=0, description="Number of correct answers")
    timeSpent: int = Field(..., ge=0, description="Time spent in seconds")
    difficulty: str = Field(..., description="Difficulty level")
    timestamp: str = Field(..., description="ISO timestamp")

# Progress Save Response Model
class ProgressSaveResponse(BaseModel):
    success: bool
    message: str
    progress_id: Optional[str] = None
    saved_at: str

@router.post("/save", response_model=ProgressSaveResponse)
async def save_progress(
    progress: QuizResult,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Save quiz progress to database"""
    
    try:
        # Validate subject
        if progress.subject.lower() not in ['math', 'english']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid subject. Must be 'math' or 'english'"
            )
        
        # Convert subject to enum
        subject_enum = Subject.MATH if progress.subject.lower() == 'math' else Subject.ENGLISH
        
        # Create progress record
        from app.models.student_progress import StudentProgress
        
        progress_record = StudentProgress(
            student_id=str(current_user.id),
            subject=subject_enum,
            score=progress.score,
            total_questions=progress.totalQuestions,
            correct_answers=progress.correctAnswers,
            time_spent_seconds=progress.timeSpent,
            difficulty_level=progress.difficulty,
            completed_at=datetime.fromisoformat(progress.timestamp.replace('Z', '+00:00')),
            created_at=datetime.utcnow()
        )
        
        # Save to database
        db.add(progress_record)
        await db.commit()
        await db.refresh(progress_record)
        
        # Log the progress save
        import structlog
        logger = structlog.get_logger()
        logger.info(
            "Progress saved successfully",
            user_id=str(current_user.id),
            subject=progress.subject,
            score=progress.score,
            progress_id=str(progress_record.id)
        )
        
        return ProgressSaveResponse(
            success=True,
            message="Progress saved successfully",
            progress_id=str(progress_record.id),
            saved_at=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import structlog
        logger = structlog.get_logger()
        logger.error(
            "Failed to save progress",
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save progress: {str(e)}"
        )

@router.get("/user", response_model=list)
async def get_user_progress(
    userId: int,
    subject: Optional[str] = Query(None, description="Filter by subject"),
    limit: int = Query(20, ge=1, le=100, description="Number of records to return"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(enhanced_database_manager.get_session)
):
    """Get user progress history"""
    
    try:
        from app.models.student_progress import StudentProgress
        from sqlalchemy import select, desc
        
        # Build query
        query = select(StudentProgress).where(
            StudentProgress.student_id == str(userId)
        ).order_by(desc(StudentProgress.created_at))
        
        # Add subject filter if provided
        if subject:
            if subject.lower() not in ['math', 'english']:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid subject. Must be 'math' or 'english'"
                )
            subject_enum = Subject.MATH if subject.lower() == 'math' else Subject.ENGLISH
            query = query.where(StudentProgress.subject == subject_enum)
        
        # Add limit
        query = query.limit(limit)
        
        # Execute query
        result = await db.execute(query)
        progress_records = result.scalars().all()
        
        # Convert to response format
        progress_list = []
        for record in progress_records:
            progress_list.append({
                "id": str(record.id),
                "subject": record.subject.value,
                "score": record.score,
                "total_questions": record.total_questions,
                "correct_answers": record.correct_answers,
                "time_spent_seconds": record.time_spent_seconds,
                "difficulty_level": record.difficulty_level,
                "completed_at": record.completed_at.isoformat(),
                "created_at": record.created_at.isoformat()
            })
        
        return progress_list
        
    except HTTPException:
        raise
    except Exception as e:
        import structlog
        logger = structlog.get_logger()
        logger.error(
            "Failed to get user progress",
            user_id=str(current_user.id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user progress: {str(e)}"
        )
