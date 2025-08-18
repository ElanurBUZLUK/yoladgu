from fastapi import APIRouter, Depends, HTTPException, status
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
from app.domains.english.hybrid_retriever import hybrid_retriever
from app.domains.english.moderation import content_moderator
from app.services.llm_gateway import llm_gateway
from app.core.cache import cache_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/english", tags=["english"])


# Request/Response Models
class EnglishQuestionRequest(BaseModel):
    topic_category: Optional[str] = Field(None, description="Belirli konu kategorisi")
    difficulty_level: Optional[int] = Field(None, ge=1, le=5, description="Zorluk seviyesi")
    question_type: Optional[QuestionType] = Field(None, description="Soru tipi")


class EnglishQuestionResponse(BaseModel):
    question: Dict[str, Any]
    context: Dict[str, Any]
    difficulty: int
    topic: str
    latency_ms: int


@router.get("/next-question", response_model=EnglishQuestionResponse)
async def next_english_question(
    request: EnglishQuestionRequest,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Get next English question using hybrid retrieval"""
    
    start_time = time.time()
    
    try:
        # Session retrieval logic - get user's current session state
        session_key = f"english_session:{current_user.id}"
        session_data = await cache_service.get(session_key)
        
        if not session_data:
            session_data = {
                "current_topic": request.topic_category or "general",
                "current_difficulty": request.difficulty_level or current_user.current_english_level,
                "attempted_questions": [],
                "session_start": datetime.utcnow().isoformat()
            }
            await cache_service.set(session_key, session_data, expire=3600)
        
        # Retrieve context using hybrid retriever
        retrieved_questions = await hybrid_retriever.retrieve_questions(
            db=db,
            user=current_user,
            topic=session_data["current_topic"],
            difficulty=session_data["current_difficulty"],
            limit=5,
            exclude_attempted=True
        )
        
        if not retrieved_questions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No suitable questions found"
            )
        
        # Select best question based on user profile
        selected_question = retrieved_questions[0]["question"]
        
        # Apply content moderation
        moderation_result = await content_moderator.moderate_content(
            selected_question.content
        )
        
        if not moderation_result["is_appropriate"]:
            # Try to find alternative question
            for question_data in retrieved_questions[1:]:
                alt_moderation = await content_moderator.moderate_content(
                    question_data["question"].content
                )
                if alt_moderation["is_appropriate"]:
                    selected_question = question_data["question"]
                    break
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No appropriate questions available"
                )
        
        # Update session data
        session_data["attempted_questions"].append(str(selected_question.id))
        await cache_service.set(session_key, session_data, expire=3600)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return EnglishQuestionResponse(
            question={
                "id": str(selected_question.id),
                "content": selected_question.content,
                "question_type": selected_question.question_type.value,
                "options": selected_question.options,
                "topic_category": selected_question.topic_category
            },
            context={
                "topic": session_data["current_topic"],
                "difficulty": session_data["current_difficulty"],
                "session_questions": len(session_data["attempted_questions"])
            },
            difficulty=session_data["current_difficulty"],
            topic=session_data["current_topic"],
            latency_ms=latency_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting next English question: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get next question"
        )