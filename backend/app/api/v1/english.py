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
from app.schemas.cefr_assessment import CEFRAssessmentRequest, CEFRAssessmentResponse
from app.services.cefr_assessment_service import cefr_assessment_service
from app.services.english_cloze_service import english_cloze_service # NEW IMPORT
from app.services.user_service import user_service # ADDED FOR TODO
from app.utils.distlock_idem import idempotency_decorator, IdempotencyConfig
import hashlib
import json

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

class ClozeGenerationRequest(BaseModel):
    student_id: str
    target_error_tag: Optional[str] = Field(None, description="Target error tag for cloze generation")
    k: int = Field(3, ge=1, le=10, description="Number of cloze questions to generate")

def _cloze_generation_key_builder(*args, **kwargs) -> str:
    """Build idempotency key for cloze generation"""
    # Extract parameters from request
    request = None
    for arg in args:
        if isinstance(arg, ClozeGenerationRequest):
            request = arg
            break
    
    if not request:
        request = kwargs.get('request')
    
    if not request:
        return "cloze_generation:default"
    
    # Create deterministic key based on request parameters
    key_data = {
        "student_id": request.student_id,
        "target_error_tag": request.target_error_tag,
        "k": request.k
    }
    
    key_string = json.dumps(key_data, sort_keys=True)
    return f"cloze_generation:{hashlib.md5(key_string.encode()).hexdigest()}"


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
        
        # Sync student context to MCP
        from app.mcp.context_manager import mcp_context_manager
        await mcp_context_manager.sync_student_context(db, str(current_user.id))
        
        # Sync session context to MCP
        await mcp_context_manager.sync_session_context(
            session_id=session_key,
            user_id=str(current_user.id),
            subject="english",
            current_topic=session_data["current_topic"],
            difficulty_level=session_data["current_difficulty"],
            attempted_questions=session_data["attempted_questions"]
        )
        
        # Retrieve context using hybrid retriever (MCP üzerinden)
        try:
            from app.core.mcp_utils import mcp_utils
            if mcp_utils.is_initialized:
                # MCP üzerinden question retrieval
                retrieval_result = await mcp_utils.call_tool(
                    tool_name="retrieve_questions",
                    arguments={
                        "user_id": str(current_user.id),
                        "topic": session_data["current_topic"],
                        "difficulty": session_data["current_difficulty"],
                        "limit": 5,
                        "exclude_attempted": True,
                        "subject": "english"
                    }
                )
                
                if retrieval_result["success"]:
                    retrieved_questions = retrieval_result["data"]
                else:
                    # Fallback to direct retrieval
                    retrieved_questions = await hybrid_retriever.retrieve_questions(
                        db=db,
                        user=current_user,
                        topic=session_data["current_topic"],
                        difficulty=session_data["current_difficulty"],
                        limit=5,
                        exclude_attempted=True
                    )
            else:
                # Direct retrieval
                retrieved_questions = await hybrid_retriever.retrieve_questions(
                    db=db,
                    user=current_user,
                    topic=session_data["current_topic"],
                    difficulty=session_data["current_difficulty"],
                    limit=5,
                    exclude_attempted=True
                )
        except Exception as e:
            logger.warning(f"MCP question retrieval failed, using fallback: {e}")
            # Fallback to direct retrieval
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


@router.post("/assess/cefr", response_model=CEFRAssessmentResponse, status_code=status.HTTP_200_OK)
async def assess_cefr_level(
    request: CEFRAssessmentRequest,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session),
    user_service_instance: user_service = Depends(lambda: user_service)
):
    """Assesss the user's CEFR level based on provided text and updates their profile."""
    try:
        assessment_result = await cefr_assessment_service.assess_cefr_level(
            user_id=str(current_user.id),
            assessment_text=request.assessment_text,
            assessment_type=request.assessment_type
        )

        # Map CEFR string level to a numeric level (1-6)
        cefr_to_numeric = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
        numeric_level = cefr_to_numeric.get(assessment_result.overall_level)

        if numeric_level:
            await user_service_instance.update_user_levels(
                db=db,
                user_id=str(current_user.id),
                english_level=numeric_level
            )
            logger.info(f"Updated user {current_user.id} English level to {numeric_level} based on CEFR assessment.")

        return assessment_result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during CEFR assessment: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to assess CEFR level: {str(e)}"
        )


class ClozeGenRequest(BaseModel):
    num_recent_errors: int = Field(5, description="Number of recent errors to consider for cloze generation")


class GenerateQuestionResponse(BaseModel):
    success: bool
    question: Dict[str, Any]
    generation_info: Dict[str, Any]


@idempotency_decorator(
    key_builder=lambda req, user, db, svc: f"cloze_generation:{user.id}:{req.num_recent_errors}",
    config=IdempotencyConfig(scope="cloze_generation", ttl_seconds=600)
)
async def _generate_cloze_internal(
    req: ClozeGenRequest,
    user: User,
    db: AsyncSession,
    svc
) -> GenerateQuestionResponse:
    """Internal function for cloze generation with idempotency"""
    cloze_question = await svc.generate_cloze_questions(session=db, user_id=str(user.id), num_questions=1, last_n_errors=req.num_recent_errors)

    if not cloze_question:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate cloze question or no questions were returned."
        )

    # The service returns a list of questions, but this endpoint expects a single question.
    # Take the first question from the list.
    first_question = cloze_question[0]

    return GenerateQuestionResponse(
        success=True,
        question=first_question.model_dump(), # Convert Pydantic model to dict
        generation_info={
            "error_type": first_question.topic_category, # Assuming error_type is stored in topic_category
            "sub_type": None, # Sub-type is not directly available in Question model
            "rule_explanation": first_question.question_metadata.get("rule_context") # Assuming rule_context is the explanation
        }
    )

@router.post("/questions/generate", response_model=GenerateQuestionResponse, status_code=status.HTTP_200_OK)
async def generate_cloze(
    req: ClozeGenRequest,
    user: User = Depends(get_current_student), # Added user dependency
    db: AsyncSession = Depends(get_async_session), # Added db dependency
    svc = Depends(lambda: english_cloze_service)
):
    """Generate a personalized English cloze question based on user's recent error patterns with idempotency."""
    try:
        return await _generate_cloze_internal(req, user, db, svc)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating question: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate question: {str(e)}"
        )

# Add the missing /api/v1/english/questions/generate endpoint (alternative path)
@router.post("/generate-cloze", response_model=GenerateQuestionResponse, status_code=status.HTTP_200_OK)
async def generate_cloze_alternative(
    req: ClozeGenRequest,
    user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session),
    svc = Depends(lambda: english_cloze_service)
):
    """Alternative English cloze generation endpoint - /api/v1/english/generate-cloze"""
    try:
        return await _generate_cloze_internal(req, user, db, svc)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating cloze question: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate cloze question: {str(e)}"
        )