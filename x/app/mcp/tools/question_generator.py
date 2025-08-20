import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field, ValidationError
import json
from redis.asyncio import Redis

from app.services.llm_gateway import llm_gateway
from app.repositories.question_repository import QuestionRepository
from app.models.question import Question, Subject, QuestionType, DifficultyLevel, SourceType
from app.schemas.question import QuestionCreate
from app.utils.distlock_idem import idempotent_singleflight, IdempotencyConfig
from app.core.config import settings

logger = logging.getLogger(__name__)

class GeneratedQuestionSchema(BaseModel):
    content: str
    correct_answer: Optional[str] = None
    options: Optional[List[str]] = None
    explanation: Optional[str] = None
    topic_category: str
    difficulty_level: int
    question_type: QuestionType

class QuestionGeneratorService:
    def __init__(
        self,
        llm_gateway_service: llm_gateway,
        question_repo: QuestionRepository
    ):
        self.llm_gateway = llm_gateway_service
        self.question_repo = question_repo
        self.redis_client = Redis.from_url(settings.redis_url) # Initialize Redis client

    async def _generate_questions_worker(
        self,
        session: AsyncSession,
        subject: Subject,
        topic: str,
        difficulty_level: DifficultyLevel,
        question_type: QuestionType,
        num_questions: int
    ) -> List[Dict[str, Any]]:
        """Worker function for question generation logic."""
        prompt = f"""
        Generate {num_questions} {question_type.value} questions about {topic} for {subject.value} subject.
        The questions should be suitable for a difficulty level of {difficulty_level.value} (1-5).
        Provide the output in a JSON array, where each object adheres to the following schema:
        {json.dumps(GeneratedQuestionSchema.model_json_schema(), indent=2)}

        Ensure the questions are clear, concise, and the correct_answer/options are accurate.
        For multiple_choice questions, provide 4 options.
        For fill_blank questions, provide the original sentence and the blanked version.
        """

        system_prompt = "You are an expert question generator for educational content. You must generate questions in the specified JSON format."

        # Try MCP first
        from app.core.mcp_utils import mcp_utils
        
        try:
            if mcp_utils.is_initialized:
                mcp_response = await mcp_utils.call_tool(
                    tool_name="llm_generate",
                    arguments={
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "output_type": "json",
                        "schema": List[GeneratedQuestionSchema].model_json_schema(),
                        "temperature": 0.3,
                        "max_tokens": 1500,
                        "context": f"Subject: {subject.value}, Topic: {topic}, Difficulty: {difficulty_level.value}",
                        "user_id": "question_generator",
                        "session_id": f"gen_{subject.value}_{topic}"
                    }
                )
                
                if mcp_response["success"]:
                    llm_response = mcp_response["data"]
                    if isinstance(llm_response, str):
                        import json
                        llm_response = json.loads(llm_response)
                else:
                    logger.warning(f"MCP question generation failed: {mcp_response.get('error')}")
                    # Fallback to direct LLM
                    llm_response = await self.llm_gateway.generate_json(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        schema=List[GeneratedQuestionSchema].model_json_schema(),
                        max_retries=3
                    )
            else:
                logger.warning("MCP not initialized, using direct LLM")
                # Fallback to direct LLM
                llm_response = await self.llm_gateway.generate_json(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    schema=List[GeneratedQuestionSchema].model_json_schema(),
                    max_retries=3
                )
        except Exception as e:
            logger.warning(f"MCP question generation failed, using fallback: {e}")
            # Fallback to direct LLM
            llm_response = await self.llm_gateway.generate_json(
                prompt=prompt,
                system_prompt=system_prompt,
                schema=List[GeneratedQuestionSchema].model_json_schema(),
                max_retries=3
            )

        if not llm_response["success"]:
            logger.error(f"LLM Gateway failed to generate questions: {llm_response.get('error')}")
            return []

        generated_questions_data = llm_response["parsed_json"]
        created_questions_data = []

        for q_data in generated_questions_data:
            try:
                # Validate each generated question against the schema
                validated_q = GeneratedQuestionSchema(**q_data)
                
                # Create QuestionCreate schema for database insertion
                question_create = QuestionCreate(
                    subject=subject,
                    content=validated_q.content,
                    question_type=validated_q.question_type,
                    difficulty_level=validated_q.difficulty_level,
                    topic_category=validated_q.topic_category,
                    correct_answer=validated_q.correct_answer,
                    options=validated_q.options,
                    source_type=SourceType.GENERATED,
                    question_metadata={
                        "explanation": validated_q.explanation
                    }
                )
                db_question = await self.question_repo.create(session, question_create)
                created_questions_data.append(db_question.to_dict()) # Convert to dict for caching
            except ValidationError as e:
                logger.error(f"Validation error for generated question: {e.errors()}")
            except Exception as e:
                logger.error(f"Error saving generated question to DB: {e}")

        logger.info(f"Successfully generated and saved {len(created_questions_data)} questions.")
        return created_questions_data

    async def generate_questions(
        self,
        session: AsyncSession,
        subject: Subject,
        topic: str,
        difficulty_level: DifficultyLevel,
        question_type: QuestionType,
        num_questions: int = 1
    ) -> List[Question]:
        """Generates new questions using LLM and saves them to the database, with idempotency."""
        idempotency_key = f"question_generation:{subject.value}:{topic}:{difficulty_level.value}:{question_type.value}:{num_questions}"
        config = IdempotencyConfig(scope="question_generation", ttl_seconds=300) # Cache for 5 minutes

        try:
            generated_questions_data = await idempotent_singleflight(
                client=self.redis_client,
                key=idempotency_key,
                config=config,
                worker=lambda: self._generate_questions_worker(session, subject, topic, difficulty_level, question_type, num_questions)
            )
            # Convert back from dict to Question objects
            return [Question(**q_data) for q_data in generated_questions_data]

        except Exception as e:
            logger.error(f"Error in question generation service: {e}", exc_info=True)
            return []

# Global instance
question_generator_service = QuestionGeneratorService(
    llm_gateway_service=llm_gateway,
    question_repo=QuestionRepository()
)