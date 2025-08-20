from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import logging

from app.core.database import get_async_session
from app.middleware.auth import get_current_student
from app.models.user import User
from app.services.question_generator import question_generator
from app.utils.distlock_idem import idempotency_decorator, IdempotencyConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/questions", tags=["question-generation"])

class QuestionGenerationRequest(BaseModel):
    subject: str = Field(..., description="Subject: 'english' or 'math'")
    error_type: str = Field(..., description="Specific error type to target")
    difficulty_level: int = Field(3, ge=1, le=5, description="Difficulty level 1-5")
    use_gpt: bool = Field(True, description="Whether to use GPT as fallback")
    student_context: Optional[str] = Field(None, description="Student's error context for GPT")

class BatchQuestionGenerationRequest(BaseModel):
    subject: str = Field(..., description="Subject: 'english' or 'math'")
    error_types: List[str] = Field(..., description="List of error types to target")
    count: int = Field(3, ge=1, le=10, description="Number of questions to generate")
    difficulty_level: int = Field(3, ge=1, le=5, description="Difficulty level 1-5")
    use_gpt: bool = Field(True, description="Whether to use GPT as fallback")

class QuestionGenerationResponse(BaseModel):
    success: bool
    question: Dict[str, Any]
    generation_info: Dict[str, Any]

class BatchQuestionGenerationResponse(BaseModel):
    success: bool
    questions: List[Dict[str, Any]]
    generation_stats: Dict[str, Any]

def _question_generation_key_builder(*args, **kwargs) -> str:
    """Build idempotency key for question generation"""
    request = None
    for arg in args:
        if isinstance(arg, (QuestionGenerationRequest, BatchQuestionGenerationRequest)):
            request = arg
            break
    
    if not request:
        request = kwargs.get('request')
    
    if not request:
        return "question_generation:default"
    
    # Create deterministic key based on request parameters
    if isinstance(request, QuestionGenerationRequest):
        key_data = {
            "subject": request.subject,
            "error_type": request.error_type,
            "difficulty_level": request.difficulty_level,
            "use_gpt": request.use_gpt
        }
    else:  # BatchQuestionGenerationRequest
        key_data = {
            "subject": request.subject,
            "error_types": sorted(request.error_types),
            "count": request.count,
            "difficulty_level": request.difficulty_level,
            "use_gpt": request.use_gpt
        }
    
    import json
    import hashlib
    key_string = json.dumps(key_data, sort_keys=True)
    return f"question_generation:{hashlib.md5(key_string.encode()).hexdigest()}"

@idempotency_decorator(
    key_builder=_question_generation_key_builder,
    config=IdempotencyConfig(scope="question_generation", ttl_seconds=600)
)
async def _generate_question_internal(
    request: QuestionGenerationRequest,
    current_user: User
) -> Dict[str, Any]:
    """Internal function for question generation with idempotency"""
    try:
        # Generate question using the hybrid system
        generated_question = await question_generator.generate_question(
            subject=request.subject,
            error_type=request.error_type,
            difficulty_level=request.difficulty_level,
            use_gpt=request.use_gpt,
            student_context=request.student_context
        )
        
        if not generated_question:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate question"
            )
        
        return generated_question
        
    except Exception as e:
        logger.error(f"Error in question generation: {e}")
        raise

@router.post("/generate", response_model=QuestionGenerationResponse)
async def generate_question(
    request: QuestionGenerationRequest,
    current_user: User = Depends(get_current_student)
):
    """Generate a single practice question using hybrid approach (templates + GPT)"""
    try:
        question_data = await _generate_question_internal(request, current_user)
        
        return QuestionGenerationResponse(
            success=True,
            question=question_data,
            generation_info={
                "method": question_data.get("type", "unknown"),
                "difficulty_level": question_data.get("difficulty_level", 3),
                "error_type": question_data.get("error_type", ""),
                "subject": question_data.get("subject", ""),
                "timestamp": question_data.get("metadata", {}).get("timestamp", "")
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in question generation endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question generation failed: {str(e)}"
        )

@router.post("/generate/batch", response_model=BatchQuestionGenerationResponse)
async def generate_batch_questions(
    request: BatchQuestionGenerationRequest,
    current_user: User = Depends(get_current_student)
):
    """Generate multiple practice questions for different error types"""
    try:
        generated_questions = []
        
        for error_type in request.error_types:
            try:
                question_data = await question_generator.generate_question(
                    subject=request.subject,
                    error_type=error_type,
                    difficulty_level=request.difficulty_level,
                    use_gpt=request.use_gpt,
                    student_context=f"Batch generation for {error_type}"
                )
                
                if question_data:
                    generated_questions.append(question_data)
                
                # Limit the number of questions
                if len(generated_questions) >= request.count:
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to generate question for {error_type}: {e}")
                continue
        
        if not generated_questions:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate any questions"
            )
        
        # Get generation statistics
        generation_stats = question_generator.get_generation_stats()
        
        return BatchQuestionGenerationResponse(
            success=True,
            questions=generated_questions,
            generation_stats=generation_stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in batch question generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch question generation failed: {str(e)}"
        )

@router.get("/templates/{subject}")
async def get_available_templates(subject: str):
    """Get available question templates for a subject"""
    try:
        subject_templates = question_generator.templates_data.get(subject, {})
        
        return {
            "success": True,
            "subject": subject,
            "available_error_types": list(subject_templates.keys()),
            "template_count": len(subject_templates),
            "templates": subject_templates
        }
        
    except Exception as e:
        logger.error(f"Error getting templates for {subject}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get templates: {str(e)}"
        )

@router.get("/stats")
async def get_generation_stats():
    """Get question generation statistics"""
    try:
        stats = question_generator.get_generation_stats()
        
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting generation stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )

@router.post("/validate")
async def validate_question_answer(
    question_id: str,
    student_answer: str,
    current_user: User = Depends(get_current_student)
):
    """Validate a student's answer to a generated question"""
    try:
        # This would typically involve checking against the stored question
        # For now, return a simple validation response
        return {
            "success": True,
            "question_id": question_id,
            "student_answer": student_answer,
            "validation_result": "pending",  # Would be computed based on stored question
            "feedback": "Answer validation feature coming soon"
        }
        
    except Exception as e:
        logger.error(f"Error validating answer: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer validation failed: {str(e)}"
        )
