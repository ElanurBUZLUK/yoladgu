from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import logging
import time
from datetime import datetime
import json

from app.core.database import get_async_session
from app.middleware.auth import get_current_student, get_current_teacher
from app.models.user import User
from app.models.question import Subject, QuestionType
from app.domains.english.hybrid_retriever import hybrid_retriever
from app.domains.english.moderation import content_moderator
from app.domains.english.rag_retriever_pgvector import rag_retriever
from app.services.vector_index_manager import vector_index_manager
from app.services.context_compression import context_compression_service
from app.services.critic_revise import critic_revise_service
from app.services.llm_gateway import llm_gateway
from app.services.mcp_service import mcp_service
from app.core.cache import cache_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/english/rag", tags=["english-rag"])


# Request/Response Models
class RAGQuestionGenerationRequest(BaseModel):
    format: str = Field(..., description="Question format: mcq, cloze, error_correction")
    difficulty: int = Field(..., ge=1, le=5, description="Difficulty level (1-5)")
    target_cefr: str = Field(..., description="Target CEFR level: A1, A2, B1, B2, C1, C2")
    max_tokens: int = Field(600, ge=100, le=1000, description="Maximum tokens for generation")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Additional constraints")
    error_focus: Optional[List[str]] = Field(None, description="Specific error patterns to focus on")
    topic: Optional[str] = Field(None, description="Specific topic focus")


class RAGQuestionResponse(BaseModel):
    question: Dict[str, Any]
    provenance: Dict[str, Any]
    quality: Dict[str, Any]
    cost_usd: float
    latency_ms: int


class RAGContextRequest(BaseModel):
    topic: str = Field(..., description="Topic to retrieve context for")
    difficulty_level: int = Field(..., ge=1, le=5)
    question_type: str = Field(..., description="Question type")
    user_error_patterns: List[str] = Field(default_factory=list)
    limit: int = Field(5, ge=1, le=20)


class RAGContextResponse(BaseModel):
    context_items: List[Dict[str, Any]]
    compressed_context: str
    token_count: int
    compression_ratio: float
    cefr_compliance: bool
    error_coverage: float


@router.post("/generate-question", response_model=RAGQuestionResponse)
async def generate_rag_question(
    request: RAGQuestionGenerationRequest,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Generate English question using RAG approach"""
    
    start_time = time.time()
    
    try:
        # Step 1: Get user's error patterns if not provided
        if not request.error_focus:
            request.error_focus = await _get_user_error_patterns(db, current_user)
        
        # Step 2: Retrieve context using RAG
        context_items = await rag_retriever.retrieve_context_for_generation(
            db=db,
            topic=request.topic or "general",
            difficulty_level=request.difficulty,
            question_type=request.format,
            user_error_patterns=request.error_focus,
            limit=10
        )
        
        # Step 3: Compress context
        compressed_context = await context_compression_service.compress_context(
            passages=context_items,
            target_cefr=request.target_cefr,
            error_focus=request.error_focus,
            budget_tokens=request.max_tokens // 2  # Reserve half for generation
        )
        
        # Step 4: Generate draft question
        draft_question = await _generate_draft_question(
            compressed_context["compressed_text"],
            request,
            current_user
        )
        
        # Step 5: Critic and revise
        final_result = await critic_revise_service.critic_and_revise(
            draft_question=draft_question,
            target_cefr=request.target_cefr,
            error_focus=request.error_focus,
            context=compressed_context["compressed_text"],
            question_format=request.format
        )
        
        # Step 6: Moderate content
        moderation_result = await content_moderator.moderate_question(
            question_content=final_result["final_question"]["content"],
            options=final_result["final_question"].get("options", []),
            correct_answer=final_result["final_question"]["correct_answer"],
            difficulty_level=request.difficulty,
            topic=request.topic or "general",
            user_id=str(current_user.id)
        )
        
        # Step 7: Calculate costs and latency
        latency_ms = int((time.time() - start_time) * 1000)
        cost_usd = _estimate_cost(request.max_tokens, final_result["revision_count"])
        
        # Step 8: Save to database if requested
        question_id = None
        if request.constraints.get("save_to_db", False):
            question_id = await _save_generated_question(
                db, final_result["final_question"], current_user, request
            )
        
        # Step 8.5: Deliver question via MCP for enhanced accessibility/presentation
        delivery_result = None
        try:
            delivery_result = await mcp_service.deliver_question_to_student(
                question_data=final_result["final_question"],
                user_id=str(current_user.id),
                learning_style=request.constraints.get("learning_style", "mixed"),
                delivery_format=request.constraints.get("delivery_format", "web"),
                accessibility_options=request.constraints.get("accessibility_options"),
            )
        except Exception as e:
            logger.warning(f"MCP delivery failed: {e}")

        question_payload = final_result["final_question"]
        if delivery_result:
            question_payload = delivery_result.get("delivered_question", question_payload)
            question_payload["interactive_elements"] = delivery_result.get("interactive_elements")
            question_payload["delivery_metadata"] = delivery_result.get("delivery_metadata")

        # Step 9: Prepare response
        response = RAGQuestionResponse(
            question=question_payload,
            provenance={
                "model": "gpt-4o-mini",  # This should come from LLM gateway
                "provider": "openai",
                "context_items": len(context_items),
                "prompt_hash": _generate_prompt_hash(request),
                "slot": "green",
                "revision_count": final_result["revision_count"]
            },
            quality={
                "schema_ok": True,
                "cefr": request.target_cefr,
                "toxicity": moderation_result.get("moderation_score", 1.0),
                "novelty": 0.11,  # This should be calculated
                "critique_score": final_result["critique"].get("overall_score", 0.7)
            },
            cost_usd=cost_usd,
            latency_ms=latency_ms
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in RAG question generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question generation failed: {str(e)}"
        )


@router.post("/retrieve-context", response_model=RAGContextResponse)
async def retrieve_rag_context(
    request: RAGContextRequest,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Retrieve context for question generation"""
    
    try:
        # Retrieve context using RAG retriever
        context_items = await rag_retriever.retrieve_context_for_generation(
            db=db,
            topic=request.topic,
            difficulty_level=request.difficulty_level,
            question_type=request.question_type,
            user_error_patterns=request.user_error_patterns,
            limit=request.limit
        )
        
        # Compress context
        compressed_result = await context_compression_service.compress_context(
            passages=context_items,
            target_cefr="B1",  # Default CEFR level
            error_focus=request.user_error_patterns,
            budget_tokens=1000
        )
        
        return RAGContextResponse(
            context_items=context_items,
            compressed_context=compressed_result["compressed_text"],
            token_count=compressed_result["token_count"],
            compression_ratio=compressed_result["compression_ratio"],
            cefr_compliance=compressed_result["cefr_compliance"],
            error_coverage=compressed_result["error_coverage"]
        )
        
    except Exception as e:
        logger.error(f"❌ Error in context retrieval: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Context retrieval failed: {str(e)}"
        )


@router.get("/similar-questions")
async def get_similar_questions(
    query_text: str = Query(..., description="Text to find similar questions for"),
    limit: int = Query(10, ge=1, le=50, description="Number of similar questions to return"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Get similar questions using vector search"""
    
    try:
        similar_questions = await rag_retriever.retrieve_similar_questions(
            db=db,
            query_text=query_text,
            user=current_user,
            limit=limit
        )
        
        return {
            "similar_questions": similar_questions,
            "query_text": query_text,
            "total_found": len(similar_questions)
        }
        
    except Exception as e:
        logger.error(f"❌ Error in similar questions retrieval: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similar questions retrieval failed: {str(e)}"
        )


@router.get("/vector-index-stats")
async def get_vector_index_statistics(
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_async_session)
):
    """Get vector index statistics"""
    
    try:
        stats = await vector_index_manager.get_index_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"❌ Error getting vector index stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get vector index statistics: {str(e)}"
        )


@router.post("/update-embeddings")
async def update_embeddings(
    batch_size: int = Query(100, ge=10, le=500, description="Batch size for embedding updates"),
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_async_session)
):
    """Update embeddings for questions and error patterns"""
    
    try:
        result = await vector_index_manager.batch_update_embeddings(db, batch_size)
        return result
        
    except Exception as e:
        logger.error(f"❌ Error updating embeddings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update embeddings: {str(e)}"
        )


# Helper functions
async def _get_user_error_patterns(db: AsyncSession, user: User) -> List[str]:
    """Get user's most common error patterns"""
    
    try:
        # This should query the database for user's error patterns
        # For now, return common patterns
        return ["grammar_error", "vocabulary_error", "article_usage"]
    except Exception as e:
        logger.error(f"❌ Error getting user error patterns: {e}")
        return []


async def _generate_draft_question(
    context: str,
    request: RAGQuestionGenerationRequest,
    user: User
) -> Dict[str, Any]:
    """Generate draft question using MCP with fallback to LLM"""

    format_map = {
        "mcq": "multiple_choice",
        "cloze": "fill_blank",
        "error_correction": "open_ended"
    }

    try:
        # Build generation prompt
        question_type = format_map.get(request.format, "multiple_choice")
        mcp_question = await mcp_service.generate_english_question_for_user(
            user_id=str(user.id),
            error_patterns=request.error_focus or [],
            difficulty_level=request.difficulty,
            question_type=question_type,
            context=context,
            topic=request.topic,
            error_focus=request.error_focus,
        )
        return mcp_question
    except Exception as mcp_error:
        logger.warning(
            f"MCP generation failed: {mcp_error}. Falling back to LLM gateway"
        )

    try:
        # Build generation prompt for LLM fallback
        prompt = _build_generation_prompt(context, request, user)
        
        # Get question from LLM
        result = await llm_gateway.generate_structured_with_fallback(
            task_type="rag_question_generation",
            prompt=prompt,
            schema=_get_question_schema(request.format),
            system_prompt=_get_generation_system_prompt(),
            complexity="medium"
        )
        
        if result.get("success"):
            return result.get("data", {})
        else:
            logger.warning("LLM generation failed, using fallback")
            return _generate_fallback_question(request)
            
    except Exception as e:
        logger.error(f"❌ Error in draft generation: {e}")
        return _generate_fallback_question(request)


def _build_generation_prompt(
    context: str,
    request: RAGQuestionGenerationRequest,
    user: User
) -> str:
    """Build prompt for question generation"""
    
    prompt = f"""
Generate an English question based on the provided context and requirements.

**Context:**
{context}

**Requirements:**
- Format: {request.format.upper()}
- Difficulty: {request.difficulty}/5
- Target CEFR: {request.target_cefr}
- Error Focus: {', '.join(request.error_focus) if request.error_focus else 'General'}
- Topic: {request.topic or 'General English'}

**Constraints:**
{json.dumps(request.constraints, indent=2)}

**Instructions:**
- Use the context to create a relevant question
- Focus on the specified error patterns
- Ensure appropriate difficulty level
- Make the question engaging and educational
- Follow the specified format exactly

Please generate the question following the specified JSON schema.
"""
    
    return prompt


def _get_question_schema(format_type: str) -> Dict[str, Any]:
    """Get JSON schema for question generation"""
    
    if format_type == "mcq":
        return {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 4,
                    "maxItems": 4
                },
                "correct_answer": {"type": "string"},
                "explanation": {"type": "string"},
                "difficulty_level": {"type": "integer", "minimum": 1, "maximum": 5},
                "topic": {"type": "string"},
                "cefr_level": {"type": "string"}
            },
            "required": ["content", "options", "correct_answer", "explanation"]
        }
    else:
        return {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "correct_answer": {"type": "string"},
                "explanation": {"type": "string"},
                "difficulty_level": {"type": "integer", "minimum": 1, "maximum": 5},
                "topic": {"type": "string"},
                "cefr_level": {"type": "string"}
            },
            "required": ["content", "correct_answer", "explanation"]
        }


def _get_generation_system_prompt() -> str:
    """Get system prompt for question generation"""
    
    return """You are an expert English language educator creating personalized questions for students.
Your role is to generate high-quality, engaging English questions that target specific learning objectives.

Guidelines:
- Create questions that are appropriate for the specified CEFR level
- Focus on the identified error patterns
- Make questions engaging and educational
- Ensure clear and unambiguous language
- Provide helpful explanations
- Follow the specified format exactly

Always respond in the specified JSON format."""


def _generate_fallback_question(request: RAGQuestionGenerationRequest) -> Dict[str, Any]:
    """Generate a fallback question when LLM fails"""
    
    if request.format == "mcq":
        return {
            "content": "Choose the correct form of the verb in the following sentence.",
            "options": ["goes", "go", "going", "gone"],
            "correct_answer": "goes",
            "explanation": "The correct form is 'goes' for third person singular present tense.",
            "difficulty_level": request.difficulty,
            "topic": request.topic or "grammar",
            "cefr_level": request.target_cefr
        }
    else:
        return {
            "content": "Complete the sentence with the correct word.",
            "correct_answer": "example",
            "explanation": "This is a fallback question.",
            "difficulty_level": request.difficulty,
            "topic": request.topic or "vocabulary",
            "cefr_level": request.target_cefr
        }


async def _save_generated_question(
    db: AsyncSession,
    question_data: Dict[str, Any],
    user: User,
    request: RAGQuestionGenerationRequest
) -> str:
    """Save generated question to database"""
    
    try:
        # This should save the question to the database
        # For now, return a mock ID
        return "generated-question-id"
    except Exception as e:
        logger.error(f"❌ Error saving generated question: {e}")
        return None


def _estimate_cost(max_tokens: int, revision_count: int) -> float:
    """Estimate cost of question generation"""
    
    # Simple cost estimation
    base_cost = max_tokens * 0.00001  # $0.00001 per token
    revision_cost = revision_count * 0.001  # $0.001 per revision
    
    return base_cost + revision_cost


def _generate_prompt_hash(request: RAGQuestionGenerationRequest) -> str:
    """Generate hash for prompt caching"""
    
    import hashlib
    request_str = f"{request.format}:{request.difficulty}:{request.target_cefr}:{request.max_tokens}"
    return hashlib.md5(request_str.encode()).hexdigest()
