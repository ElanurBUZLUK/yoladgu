from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import logging
import time
from datetime import datetime

from app.database_enhanced import enhanced_database_manager as database_manager
from app.middleware.auth import get_current_student, get_current_teacher
from app.models.user import User
from app.models.question import Subject, QuestionType, SourceType, Question
from app.domains.english.hybrid_retriever import hybrid_retriever
from app.domains.english.moderation import content_moderator
from app.services.llm_gateway import llm_gateway
from app.core.cache import cache_service
from app.schemas.cefr_assessment import CEFRAssessmentRequest, CEFRAssessmentResponse
from app.services.cefr_assessment_service import cefr_assessment_service
from app.services.english_cloze_service import english_cloze_service # NEW IMPORT
from app.services.user_service import user_service # ADDED FOR TODO
from app.utils.distlock_idem import idempotency_decorator, IdempotencyConfig
from app.core.error_handling import ErrorHandler, ErrorCode, ErrorSeverity
import hashlib
import json
from sqlalchemy import select, and_, func, desc
from app.models.student_attempt import StudentAttempt

logger = logging.getLogger(__name__)
error_handler = ErrorHandler()

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

class EnglishQuestion(BaseModel):
    id: str
    content: str
    options: List[str] = Field(min_items=3)
    correct_answer: str
    difficulty_level: float
    topic_category: Optional[str] = None

class GenerateQuestionResponse(BaseModel):
    success: bool
    question: Optional[EnglishQuestion] = None
    generation_info: Optional[dict] = None

class ClozeGenerationRequest(BaseModel):
    student_id: str
    k: int = Field(default=1, ge=1, le=10)


@router.post("/questions/generate", response_model=GenerateQuestionResponse)
async def generate_cloze_questions(
    request: ClozeGenerationRequest,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(database_manager.get_session),
):
    """Generate English questions with CEFR level tracking and adaptive mode"""
    
    try:
        # Check if user is in placement test mode or adaptive mode
        session_key = f"english_session:{current_user.id}"
        session_data = await cache_service.get(session_key)
        
        if not session_data:
            # Initialize new session
            session_data = {
                "is_placement_test": True,
                "placement_progress": 0,
                "cefr_level": None,
                "attempt_count": 0,
                "session_start": datetime.utcnow().isoformat()
            }
            await cache_service.set(session_key, session_data, expire=3600)
        
        # Check if placement test is complete
        if session_data.get("is_placement_test", True) and session_data.get("placement_progress", 0) >= 20:
            # Switch to adaptive mode
            session_data["is_placement_test"] = False
            session_data["cefr_level"] = session_data.get("cefr_level", "B1")  # Default CEFR level
            await cache_service.set(session_key, session_data, expire=3600)
        
        # Generate question based on mode
        if session_data.get("is_placement_test", True):
            # Placement test mode - use baseline questions
            question = await _generate_placement_question(db, current_user)
        else:
            # Adaptive mode - use CEFR-based question selection
            question = await _generate_adaptive_question(db, current_user, session_data)
        
        if not question:
            return GenerateQuestionResponse(
                success=False,
                generation_info={"error": "no_question_generated"}
            )
        
        # Update session data
        session_data["attempt_count"] = session_data.get("attempt_count", 0) + 1
        if session_data.get("is_placement_test", True):
            session_data["placement_progress"] = session_data.get("placement_progress", 0) + 1
        await cache_service.set(session_key, session_data, expire=3600)
        
        # Invariant checks
        assert question["correct_answer"] in question["options"]
        assert len(set(question["options"])) == len(question["options"])
        
        return GenerateQuestionResponse(
            success=True,
            question=EnglishQuestion(**question),
            generation_info={
                "mode": "placement_test" if session_data.get("is_placement_test", True) else "adaptive",
                "cefr_level": session_data.get("cefr_level"),
                "placement_progress": session_data.get("placement_progress", 0),
                "attempt_count": session_data.get("attempt_count", 0)
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating English question: {e}")
        return GenerateQuestionResponse(
            success=False,
            generation_info={"error": str(e)}
        )

async def _generate_placement_question(db: AsyncSession, user: User) -> Optional[Dict[str, Any]]:
    """Generate baseline question for placement test"""
    try:
        # Get a random baseline question
        result = await db.execute(
            select(Question)
            .where(
                and_(
                    Question.subject == Subject.ENGLISH,
                    Question.difficulty_level.between(1, 3),  # Start with moderate difficulty
                    Question.question_type == QuestionType.MULTIPLE_CHOICE
                )
            )
            .order_by(func.random())
            .limit(1)
        )
        
        question = result.scalar_one_or_none()
        if not question:
            return None
        
        return {
            "id": str(question.id),
            "content": question.content,
            "options": question.options if question.options else ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": question.correct_answer,
            "difficulty_level": question.difficulty_level,
            "topic_category": question.topic_category
        }
        
    except Exception as e:
        logger.error(f"Error generating placement question: {e}")
        return None

async def _generate_adaptive_question(db: AsyncSession, user: User, session_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Generate adaptive question based on CEFR level and user performance"""
    try:
        cefr_level = session_data.get("cefr_level", "B1")
        
        # Map CEFR level to difficulty
        cefr_to_difficulty = {
            "A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 5
        }
        target_difficulty = cefr_to_difficulty.get(cefr_level, 3)
        
        # Get user's recent performance to adjust difficulty
        recent_attempts = await db.execute(
            select(StudentAttempt)
            .where(
                and_(
                    StudentAttempt.user_id == user.id,
                    StudentAttempt.subject == Subject.ENGLISH
                )
            )
            .order_by(desc(StudentAttempt.attempt_date))
            .limit(10)
        )
        
        attempts = recent_attempts.scalars().all()
        if attempts:
            recent_accuracy = sum(1 for a in attempts if a.is_correct) / len(attempts)
            # Adjust difficulty based on recent performance
            if recent_accuracy > 0.8:
                target_difficulty = min(5, target_difficulty + 1)
            elif recent_accuracy < 0.4:
                target_difficulty = max(1, target_difficulty - 1)
        
        # Consider learning style for question type preference
        learning_style = user.learning_style if hasattr(user, 'learning_style') else None
        preferred_question_types = []
        
        if learning_style:
            if learning_style.value == "visual":
                preferred_question_types = [QuestionType.MULTIPLE_CHOICE, QuestionType.TRUE_FALSE]
            elif learning_style.value == "auditory":
                preferred_question_types = [QuestionType.FILL_BLANK, QuestionType.OPEN_ENDED]
            elif learning_style.value == "kinesthetic":
                preferred_question_types = [QuestionType.OPEN_ENDED, QuestionType.FILL_BLANK]
            else:  # mixed
                preferred_question_types = [QuestionType.MULTIPLE_CHOICE, QuestionType.FILL_BLANK, QuestionType.OPEN_ENDED]
        
        # Build query with learning style consideration
        query = select(Question).where(
            and_(
                Question.subject == Subject.ENGLISH,
                Question.difficulty_level == target_difficulty
            )
        )
        
        # Add question type filter if learning style is specified
        if preferred_question_types:
            query = query.where(Question.question_type.in_(preferred_question_types))
        else:
            query = query.where(Question.question_type == QuestionType.MULTIPLE_CHOICE)
        
        query = query.order_by(func.random()).limit(1)
        
        result = await db.execute(query)
        question = result.scalar_one_or_none()
        
        if not question:
            # Fallback to any available question
            result = await db.execute(
                select(Question)
                .where(
                    and_(
                        Question.subject == Subject.ENGLISH,
                        Question.question_type == QuestionType.MULTIPLE_CHOICE
                    )
                )
                .order_by(func.random())
                .limit(1)
            )
            question = result.scalar_one_or_none()
        
        if not question:
            return None
        
        return {
            "id": str(question.id),
            "content": question.content,
            "options": question.options if question.options else ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": question.correct_answer,
            "difficulty_level": question.difficulty_level,
            "topic_category": question.topic_category
        }
        
    except Exception as e:
        logger.error(f"Error generating adaptive question: {e}")
        return None


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
    db: AsyncSession = Depends(database_manager.get_session)
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
    db: AsyncSession = Depends(database_manager.get_session),
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

class JSONUploadResponse(BaseModel):
    success: bool
    message: str
    questions_imported: int
    questions_failed: int
    errors: List[str] = []


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
    db: AsyncSession = Depends(database_manager.get_session), # Added db dependency
    svc = Depends(lambda: english_cloze_service)
):
    """Generate a personalized English cloze question based on user's recent error patterns with idempotency."""
    try:
        return await _generate_cloze_internal(req, user, db, svc)
    except HTTPException:
        raise
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.LLM_SERVICE_ERROR,
            message="Failed to generate English cloze question",
            severity=ErrorSeverity.HIGH,
            context={
                "user_id": str(user.id),
                "error_type": req.error_type,
                "request_id": req.model_dump_json()[:100]  # Truncated for security
            }
        )

# Add the missing /api/v1/english/questions/generate endpoint (alternative path)
@router.post("/generate-cloze", response_model=GenerateQuestionResponse, status_code=status.HTTP_200_OK)
async def generate_cloze_alternative(
    req: ClozeGenRequest,
    user: User = Depends(get_current_student),
    db: AsyncSession = Depends(database_manager.get_session),
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


@router.post("/questions/upload-json", response_model=JSONUploadResponse)
async def upload_english_questions_json(
    file: UploadFile = File(..., description="JSON file containing English questions"),
    current_user: User = Depends(get_current_teacher),  # Only teachers can upload
    db: AsyncSession = Depends(database_manager.get_session)
):
    """
    Upload English questions from a JSON file using enhanced format
    
    Expected JSON format:
    [
        {
            "stem": "Choose the correct form: I ____ to school every day.",
            "options": {
                "A": "go",
                "B": "goes",
                "C": "going",
                "D": "went"
            },
            "correct_answer": "A",
            "topic": "grammar",
            "subtopic": "present_tense",
            "difficulty": 0.8,
            "source": "seed",
            "metadata": {
                "estimated_time": 30,
                "learning_objectives": ["present tense with 'I'"],
                "tags": ["grammar", "present_tense", "basic"],
                "cefr_level": "A1"
            }
        }
    ]
    """
    try:
        # Validate file type
        if not file.filename.endswith('.json'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only JSON files are allowed"
            )
        
        # Read and parse JSON
        content = await file.read()
        try:
            questions_data = json.loads(content.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON format: {str(e)}"
            )
        
        if not isinstance(questions_data, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="JSON must contain an array of questions"
            )
        
        # Process questions
        questions_imported = 0
        questions_failed = 0
        errors = []
        
        for i, question_data in enumerate(questions_data):
            try:
                # Validate required fields for enhanced format
                required_fields = ['stem', 'options', 'correct_answer', 'topic']
                for field in required_fields:
                    if field not in question_data:
                        raise ValueError(f"Missing required field: {field}")
                
                # Convert difficulty (continuous 0.0-2.0 to discrete 1-5)
                difficulty = question_data.get('difficulty', 1.0)
                if difficulty <= 0.5:
                    difficulty_level = 1
                elif difficulty <= 1.0:
                    difficulty_level = 2
                elif difficulty <= 1.5:
                    difficulty_level = 3
                elif difficulty <= 1.8:
                    difficulty_level = 4
                else:
                    difficulty_level = 5
                
                # Determine question type based on options
                options = question_data['options']
                if len(options) == 2 and all(opt in ['True', 'False', 'true', 'false'] for opt in options.values()):
                    question_type = QuestionType.TRUE_FALSE
                elif len(options) == 0:
                    question_type = QuestionType.OPEN_ENDED
                else:
                    question_type = QuestionType.MULTIPLE_CHOICE
                
                # Create question object with enhanced format
                question = Question(
                    subject=Subject.ENGLISH,
                    content=question_data['stem'],
                    question_type=question_type,
                    difficulty_level=difficulty_level,
                    original_difficulty=difficulty_level,
                    topic_category=question_data['topic'],
                    correct_answer=question_data['correct_answer'],
                    options=question_data['options'],
                    source_type=SourceType.MANUAL,
                    estimated_difficulty=question_data.get('difficulty', 1.0),
                    question_metadata={
                        "subtopic": question_data.get('subtopic'),
                        "source": question_data.get('source', 'seed'),
                        "estimated_time": question_data.get('metadata', {}).get('estimated_time', 60),
                        "learning_objectives": question_data.get('metadata', {}).get('learning_objectives', []),
                        "tags": question_data.get('metadata', {}).get('tags', []),
                        "cefr_level": question_data.get('metadata', {}).get('cefr_level', 'A1')
                    },
                    created_by=str(current_user.id)
                )
                
                db.add(question)
                questions_imported += 1
                
            except Exception as e:
                questions_failed += 1
                error_msg = f"Question {i+1}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        # Commit all successful questions
        await db.commit()
        
        return JSONUploadResponse(
            success=True,
            message=f"Successfully imported {questions_imported} English questions",
            questions_imported=questions_imported,
            questions_failed=questions_failed,
            errors=errors
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error uploading English JSON questions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload English questions: {str(e)}"
        )