from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List

from app.core.database import get_async_session
from app.services.question_service import question_service
from app.services.llm_gateway import llm_gateway
from app.schemas.question import (
    QuestionCreate, QuestionUpdate, QuestionResponse, 
    QuestionRecommendationRequest, QuestionRecommendationResponse,
    QuestionSearchQuery, QuestionValidationResult
)
from app.middleware.auth import (
    get_current_active_user, get_current_student, 
    get_current_teacher, get_current_admin
)
from app.models.user import User
from app.models.question import Subject, QuestionType, SourceType

router = APIRouter(prefix="/api/v1/english", tags=["english"])


# English Question Generation Schema
from pydantic import BaseModel, Field

class EnglishQuestionGenerationRequest(BaseModel):
    difficulty_level: int = Field(..., ge=1, le=5, description="Difficulty level (1-5)")
    question_type: QuestionType = Field(default=QuestionType.MULTIPLE_CHOICE)
    topic_focus: Optional[str] = Field(None, description="Specific grammar/vocabulary focus")
    error_patterns: Optional[List[str]] = Field(default=[], description="User's error patterns to focus on")
    learning_style: Optional[str] = Field(None, description="User's learning style")
    count: int = Field(1, ge=1, le=10, description="Number of questions to generate")
    context: Optional[str] = Field(None, description="Additional context for question generation")


class GeneratedQuestionResponse(BaseModel):
    generated_question: dict
    quality_score: Optional[float] = None
    generation_method: str
    estimated_time: Optional[int] = None
    focus_areas: List[str] = []
    saved_to_database: bool = False
    question_id: Optional[str] = None


@router.post("/questions/generate", response_model=List[GeneratedQuestionResponse])
async def generate_english_questions(
    generation_request: EnglishQuestionGenerationRequest,
    save_to_db: bool = Query(False, description="Save generated questions to database"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Generate English questions using LLM with MCP integration"""
    
    try:
        generated_questions = []
        
        for i in range(generation_request.count):
            # Use LLM Gateway to generate question
            llm_result = await llm_gateway.generate_english_question(
                user_id=str(current_user.id),
                error_patterns=generation_request.error_patterns,
                difficulty_level=generation_request.difficulty_level,
                question_type=generation_request.question_type.value,
                topic=generation_request.topic_focus
            )
            
            if llm_result.get("success"):
                generated_question_data = llm_result.get("content", {})
                
                # Validate generated question
                validation_result = await _validate_generated_question(generated_question_data)
                
                response_data = GeneratedQuestionResponse(
                    generated_question=generated_question_data,
                    quality_score=validation_result.quality_score,
                    generation_method="llm_mcp",
                    estimated_time=generated_question_data.get("metadata", {}).get("estimated_time"),
                    focus_areas=generated_question_data.get("focus_areas", []),
                    saved_to_database=False
                )
                
                # Save to database if requested and valid
                if save_to_db and validation_result.is_valid:
                    try:
                        question_create = QuestionCreate(
                            subject=Subject.ENGLISH,
                            content=generated_question_data.get("content", ""),
                            question_type=QuestionType(generated_question_data.get("question_type", "multiple_choice")),
                            difficulty_level=generation_request.difficulty_level,
                            topic_category=generation_request.topic_focus or "general",
                            correct_answer=generated_question_data.get("correct_answer"),
                            options=generated_question_data.get("options"),
                            source_type=SourceType.GENERATED,
                            question_metadata={
                                "generated_by": "llm_mcp",
                                "user_id": str(current_user.id),
                                "error_patterns": generation_request.error_patterns,
                                "focus_areas": generated_question_data.get("focus_areas", []),
                                "quality_score": validation_result.quality_score,
                                "generation_timestamp": llm_result.get("metadata", {}).get("generated_at")
                            }
                        )
                        
                        saved_question = await question_service.create_question(db, question_create)
                        response_data.saved_to_database = True
                        response_data.question_id = str(saved_question.id)
                        
                    except Exception as e:
                        print(f"Error saving generated question: {e}")
                
                generated_questions.append(response_data)
            
            else:
                # Fallback to template-based generation
                fallback_question = await _generate_fallback_question(
                    generation_request.difficulty_level,
                    generation_request.question_type,
                    generation_request.topic_focus,
                    generation_request.error_patterns
                )
                
                response_data = GeneratedQuestionResponse(
                    generated_question=fallback_question,
                    quality_score=0.7,  # Template-based quality
                    generation_method="template_fallback",
                    estimated_time=60,
                    focus_areas=generation_request.error_patterns,
                    saved_to_database=False
                )
                
                generated_questions.append(response_data)
        
        return generated_questions
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question generation failed: {str(e)}"
        )


@router.post("/questions/recommend", response_model=QuestionRecommendationResponse)
async def recommend_english_questions(
    recommendation_request: QuestionRecommendationRequest,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Get recommended English questions (existing + generated)"""
    
    # Ensure subject is English
    recommendation_request.subject = Subject.ENGLISH
    
    # Use user's current English level if not specified
    if not recommendation_request.user_level:
        recommendation_request.user_level = current_user.current_english_level
    
    # Use user's learning style if not specified
    if not recommendation_request.learning_style:
        recommendation_request.learning_style = current_user.learning_style.value
    
    # Get existing questions first
    existing_recommendations = await question_service.recommend_questions(db, current_user, recommendation_request)
    
    # If we don't have enough questions, generate some
    if len(existing_recommendations.questions) < recommendation_request.limit:
        needed_questions = recommendation_request.limit - len(existing_recommendations.questions)
        
        # Generate additional questions
        generation_request = EnglishQuestionGenerationRequest(
            difficulty_level=recommendation_request.user_level or current_user.current_english_level,
            question_type=recommendation_request.question_types[0] if recommendation_request.question_types else QuestionType.MULTIPLE_CHOICE,
            error_patterns=recommendation_request.error_patterns or [],
            learning_style=recommendation_request.learning_style,
            count=min(needed_questions, 3)  # Generate up to 3 additional questions
        )
        
        try:
            generated_questions = await generate_english_questions(
                generation_request, save_to_db=True, current_user=current_user, db=db
            )
            
            # Convert generated questions to QuestionResponse format
            for gen_q in generated_questions:
                if gen_q.saved_to_database and gen_q.question_id:
                    # Fetch the saved question
                    saved_question = await question_service.get_question_by_id(db, gen_q.question_id)
                    if saved_question:
                        question_response = QuestionResponse(
                            id=str(saved_question.id),
                            subject=saved_question.subject,
                            content=saved_question.content,
                            question_type=saved_question.question_type,
                            difficulty_level=saved_question.difficulty_level,
                            original_difficulty=saved_question.original_difficulty,
                            topic_category=saved_question.topic_category,
                            correct_answer=saved_question.correct_answer,
                            options=saved_question.options,
                            source_type=saved_question.source_type,
                            pdf_source_path=saved_question.pdf_source_path,
                            question_metadata=saved_question.question_metadata,
                            created_at=saved_question.created_at.isoformat()
                        )
                        existing_recommendations.questions.append(question_response)
        
        except Exception as e:
            print(f"Error generating additional questions: {e}")
    
    return existing_recommendations


@router.get("/questions/search")
async def search_english_questions(
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
    """Search English questions with filters"""
    
    search_query = QuestionSearchQuery(
        query=query,
        subject=Subject.ENGLISH,
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


@router.get("/questions/topics")
async def get_english_topics(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Get available English topics/categories"""
    
    # Get unique topic categories for English questions
    from sqlalchemy import select, distinct
    from app.models.question import Question
    
    result = await db.execute(
        select(distinct(Question.topic_category))
        .where(Question.subject == Subject.ENGLISH)
        .order_by(Question.topic_category)
    )
    
    topics = [row[0] for row in result.fetchall()]
    
    # Add common English topics that can be generated
    common_topics = [
        "grammar", "vocabulary", "tenses", "prepositions", 
        "articles", "pronouns", "adjectives", "adverbs",
        "sentence_structure", "reading_comprehension"
    ]
    
    # Merge and deduplicate
    all_topics = list(set(topics + common_topics))
    all_topics.sort()
    
    # Group topics by difficulty level
    topic_by_level = {}
    for level in range(1, 6):
        level_result = await db.execute(
            select(distinct(Question.topic_category))
            .where(
                Question.subject == Subject.ENGLISH,
                Question.difficulty_level == level
            )
            .order_by(Question.topic_category)
        )
        topic_by_level[str(level)] = [row[0] for row in level_result.fetchall()]
    
    return {
        "all_topics": all_topics,
        "existing_topics": topics,
        "generatable_topics": common_topics,
        "topics_by_level": topic_by_level,
        "total_topics": len(all_topics)
    }


@router.post("/questions/validate")
async def validate_english_question(
    question_data: dict,
    current_user: User = Depends(get_current_teacher)
):
    """Validate an English question for quality and correctness"""
    
    validation_result = await _validate_generated_question(question_data)
    
    return {
        "validation_result": validation_result,
        "recommendations": _get_improvement_recommendations(validation_result),
        "validated_by": current_user.username
    }


@router.get("/questions/generation-stats")
async def get_generation_stats(
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_async_session)
):
    """Get English question generation statistics"""
    
    from sqlalchemy import select, func
    from app.models.question import Question
    
    # Total generated questions
    generated_result = await db.execute(
        select(func.count(Question.id))
        .where(
            Question.subject == Subject.ENGLISH,
            Question.source_type == SourceType.GENERATED
        )
    )
    generated_count = generated_result.scalar() or 0
    
    # Manual questions
    manual_result = await db.execute(
        select(func.count(Question.id))
        .where(
            Question.subject == Subject.ENGLISH,
            Question.source_type == SourceType.MANUAL
        )
    )
    manual_count = manual_result.scalar() or 0
    
    # By difficulty level
    difficulty_stats = {}
    for level in range(1, 6):
        level_result = await db.execute(
            select(func.count(Question.id))
            .where(
                Question.subject == Subject.ENGLISH,
                Question.difficulty_level == level,
                Question.source_type == SourceType.GENERATED
            )
        )
        difficulty_stats[str(level)] = level_result.scalar() or 0
    
    return {
        "total_english_questions": generated_count + manual_count,
        "generated_questions": generated_count,
        "manual_questions": manual_count,
        "generation_percentage": (generated_count / (generated_count + manual_count) * 100) if (generated_count + manual_count) > 0 else 0,
        "generated_by_difficulty": difficulty_stats
    }


# Teacher/Admin endpoints for question management
@router.post("/questions/", response_model=QuestionResponse, status_code=status.HTTP_201_CREATED)
async def create_english_question(
    question_data: QuestionCreate,
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_async_session)
):
    """Create a new English question manually (Teacher/Admin only)"""
    
    # Ensure subject is English
    question_data.subject = Subject.ENGLISH
    
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
async def get_english_question(
    question_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_session)
):
    """Get a specific English question by ID"""
    
    question = await question_service.get_question_by_id(db, question_id)
    
    if not question:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found"
        )
    
    if question.subject != Subject.ENGLISH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question is not an English question"
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


# Helper functions
async def _validate_generated_question(question_data: dict) -> QuestionValidationResult:
    """Validate a generated English question"""
    
    errors = []
    warnings = []
    suggestions = []
    
    # Basic validation
    if not question_data.get("content"):
        errors.append("Question content is missing")
    
    if not question_data.get("correct_answer"):
        errors.append("Correct answer is missing")
    
    # Question type specific validation
    question_type = question_data.get("question_type", "")
    
    if question_type == "multiple_choice":
        options = question_data.get("options", [])
        if len(options) < 2:
            errors.append("Multiple choice questions need at least 2 options")
        
        correct_answer = question_data.get("correct_answer")
        if correct_answer and correct_answer not in options:
            warnings.append("Correct answer should be one of the options")
    
    elif question_type == "fill_blank":
        content = question_data.get("content", "")
        if "_" not in content and "____" not in content:
            warnings.append("Fill in the blank questions should contain blanks")
    
    # English-specific validation
    content = question_data.get("content", "")
    if len(content) > 300:
        suggestions.append("Consider shortening the question for better readability")
    
    # Calculate quality score
    quality_score = 1.0
    if errors:
        quality_score -= len(errors) * 0.3
    if warnings:
        quality_score -= len(warnings) * 0.1
    
    quality_score = max(0.0, min(1.0, quality_score))
    
    return QuestionValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        suggestions=suggestions,
        quality_score=quality_score
    )


async def _generate_fallback_question(
    difficulty_level: int,
    question_type: QuestionType,
    topic_focus: Optional[str],
    error_patterns: List[str]
) -> dict:
    """Generate a fallback question using templates"""
    
    templates = {
        1: {
            "grammar": {
                "content": "Choose the correct form: I ____ to school every day.",
                "options": ["go", "goes", "going", "went"],
                "correct_answer": "go",
                "explanation": "Use 'go' with 'I' in present tense."
            }
        },
        2: {
            "tenses": {
                "content": "Fill in the blank: Yesterday, I _____ (watch) a movie.",
                "correct_answer": "watched",
                "explanation": "Use past tense 'watched' for yesterday."
            }
        },
        3: {
            "present_perfect": {
                "content": "Choose the correct answer: I ____ never been to Paris.",
                "options": ["have", "has", "had", "having"],
                "correct_answer": "have",
                "explanation": "Use 'have' with 'I' in present perfect tense."
            }
        }
    }
    
    # Select appropriate template based on difficulty and focus
    level_templates = templates.get(difficulty_level, templates[1])
    
    if topic_focus and topic_focus in level_templates:
        template = level_templates[topic_focus]
    else:
        # Pick first available template
        template = list(level_templates.values())[0]
    
    return {
        "content": template["content"],
        "question_type": question_type.value,
        "correct_answer": template["correct_answer"],
        "options": template.get("options"),
        "explanation": template.get("explanation"),
        "focus_areas": [topic_focus] if topic_focus else ["general"],
        "metadata": {
            "generated_by": "template",
            "difficulty_level": difficulty_level
        }
    }


def _get_improvement_recommendations(validation_result: QuestionValidationResult) -> List[str]:
    """Get recommendations for improving question quality"""
    
    recommendations = []
    
    if validation_result.errors:
        recommendations.append("Fix the validation errors before using this question")
    
    if validation_result.warnings:
        recommendations.append("Address the warnings to improve question quality")
    
    if validation_result.quality_score and validation_result.quality_score < 0.7:
        recommendations.append("Consider revising the question to improve quality score")
    
    recommendations.extend(validation_result.suggestions)
    
    if not recommendations:
        recommendations.append("Question looks good! Ready to use.")
    
    return recommendations


@router.get("/health")
async def english_health():
    """İngilizce modülü health check"""
    return {"status": "ok", "module": "english"}