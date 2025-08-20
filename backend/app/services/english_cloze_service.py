import logging
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import ValidationError, TypeAdapter

from app.services.llm_gateway import llm_gateway
from app.repositories.question_repository import QuestionRepository
# --- YENİ EKLENENLER ---
# Bu bileşenlerin oluşturulduğunu varsayıyoruz (Görev 1 & 2)
from app.services.llm_context.builders.cloze_question_context_builder import ClozeQuestionContextBuilder
from app.services.llm_context.schemas.cloze_question_context import ClozeQuestionGenerationContext
# --- BİTİŞ ---
from app.models.question import Question, Subject, QuestionType, SourceType
from app.schemas.question import ClozeQuestionSchema, QuestionCreate
from app.models.error_pattern import ErrorPattern

logger = logging.getLogger(__name__)

class EnglishClozeService:
    def __init__(
        self,
        llm_gateway_service: llm_gateway,
        context_builder: ClozeQuestionContextBuilder,
        question_repo: QuestionRepository,
    ):
        self.llm_gateway = llm_gateway_service
        self.context_builder = context_builder
        self.question_repo = question_repo

    async def generate_cloze_questions(
        self,
        session: AsyncSession,
        user_id: str,
        num_questions: int = 1,
        last_n_errors: int = 5
    ) -> List[Question]:
        """Generates cloze questions based on user's recent errors and relevant rules.
        """
        try:
            # 1. Get last N errors for the user
            # 1. Yapısal bağlamı oluşturmak için ContextBuilder'ı kullan (Görev 2 & 4)
            context: ClozeQuestionGenerationContext = await self.context_builder.build(
                session=session,
                user_id=user_id,
                num_questions=num_questions,
                last_n_errors=last_n_errors
            )

            # 2. Bağlam nesnesinden prompt ve sistem mesajını al (Görev 3)
            # Bu metodlar, Jinja2 şablonunu context verileriyle doldurur.
            prompt = context.to_prompt()
            system_prompt = context.to_system_prompt()

            # 3. MCP üzerinden LLM çağır
            from app.core.mcp_utils import mcp_utils
            
            try:
                if mcp_utils.is_initialized:
                    mcp_response = await mcp_utils.call_tool(
                        tool_name="generate_english_cloze",
                        arguments={
                            "student_id": user_id,
                            "num_recent_errors": last_n_errors,
                            "difficulty_level": 3,  # Default level
                            "question_type": "cloze",
                            "prompt": prompt,
                            "system_prompt": system_prompt,
                            "schema": TypeAdapter(List[ClozeQuestionSchema]).json_schema(),
                            "num_questions": num_questions
                        }
                    )
                    
                    if mcp_response["success"]:
                        llm_response = mcp_response["data"]
                        if isinstance(llm_response, str):
                            import json
                            llm_response = json.loads(llm_response)
                    else:
                        logger.warning(f"MCP cloze generation failed: {mcp_response.get('error')}")
                        # Fallback to direct LLM
                        llm_response = await self.llm_gateway.generate_json(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            schema=TypeAdapter(List[ClozeQuestionSchema]).json_schema(),
                            max_retries=3
                        )
                else:
                    logger.warning("MCP not initialized, using direct LLM")
                    # Fallback to direct LLM
                    llm_response = await self.llm_gateway.generate_json(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        schema=TypeAdapter(List[ClozeQuestionSchema]).json_schema(),
                        max_retries=3
                    )
            except Exception as e:
                logger.warning(f"MCP cloze generation failed, using fallback: {e}")
                # Fallback to direct LLM
                llm_response = await self.llm_gateway.generate_json(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    schema=TypeAdapter(List[ClozeQuestionSchema]).json_schema(),
                    max_retries=3
                )

            if not llm_response["success"]:
                logger.error(f"LLM Gateway failed to generate cloze questions: {llm_response.get('error')}")
                return []

            generated_cloze_data = llm_response["parsed_json"]
            created_questions = []

            for q_data in generated_cloze_data:
                try: # Bu kısım büyük ölçüde aynı kalır
                    # Validate each generated question against the schema
                    validated_q = ClozeQuestionSchema(**q_data)
                    
                    # Basic validation: correct answer not in distractors
                    if validated_q.correct_answer in validated_q.distractors:
                        logger.warning(f"Generated cloze question has correct answer in distractors. Skipping: {validated_q.original_sentence}")
                        continue

                    # Create QuestionCreate schema for database insertion
                    question_create = QuestionCreate(
                        subject=Subject.ENGLISH,
                        content=validated_q.cloze_sentence,
                        question_type=QuestionType.FILL_BLANK,
                        difficulty_level=validated_q.difficulty_level,
                        topic_category=validated_q.error_type_addressed, # Using error type as topic
                        correct_answer=validated_q.correct_answer,
                        options=validated_q.distractors + [validated_q.correct_answer], # Combine for options
                        source_type=SourceType.GENERATED,
                        question_metadata={
                            "original_sentence": validated_q.original_sentence,
                            "rule_context": validated_q.rule_context
                        }
                    )
                    db_question = await self.question_repo.create(session, question_create)
                    created_questions.append(db_question)
                except ValidationError as e:
                    logger.error(f"Validation error for generated cloze question: {e.errors()}")
                except Exception as e:
                    logger.error(f"Error saving generated cloze question to DB: {e}")

            logger.info(f"Successfully generated and saved {len(created_questions)} cloze questions.")
            return created_questions

        except Exception as e:
            logger.error(f"Error in EnglishClozeService: {e}", exc_info=True)
            return []

# Global instance
from app.services.retriever import HybridRetriever
from app.repositories.error_pattern_repository import ErrorPatternRepository

# Context builder instance'ı oluştur
cloze_context_builder = ClozeQuestionContextBuilder(
    retriever=HybridRetriever(),
    error_pattern_repo=ErrorPatternRepository()
)

# Global service instance
english_cloze_service = EnglishClozeService(
    llm_gateway_service=llm_gateway,
    context_builder=cloze_context_builder,
    question_repo=QuestionRepository()
)