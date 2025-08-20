import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import ValidationError, TypeAdapter
import json
import re

from app.services.llm_gateway import llm_gateway
from app.services.question_generator import question_generator
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

    def _repair_json_response(self, llm_response: str, max_repair_attempts: int = 3) -> Optional[List[Dict[str, Any]]]:
        """JSON response'u onarmaya çalışır"""
        for attempt in range(max_repair_attempts):
            try:
                # 1. Temiz JSON parsing dene
                if isinstance(llm_response, str):
                    # Markdown code blocks'ları temizle
                    llm_response = re.sub(r'```json\s*', '', llm_response)
                    llm_response = re.sub(r'```\s*$', '', llm_response)
                    llm_response = llm_response.strip()
                    
                    parsed = json.loads(llm_response)
                    
                    # Array formatını kontrol et
                    if isinstance(parsed, list):
                        return parsed
                    elif isinstance(parsed, dict) and "questions" in parsed:
                        return parsed["questions"]
                    elif isinstance(parsed, dict) and "data" in parsed:
                        return parsed["data"]
                    else:
                        return [parsed]  # Tek obje ise array'e çevir
                
                elif isinstance(llm_response, list):
                    return llm_response
                elif isinstance(llm_response, dict):
                    if "questions" in llm_response:
                        return llm_response["questions"]
                    elif "data" in llm_response:
                        return llm_response["data"]
                    else:
                        return [llm_response]
                        
            except json.JSONDecodeError as e:
                logger.warning(f"JSON repair attempt {attempt + 1} failed: {e}")
                if attempt < max_repair_attempts - 1:
                    # Basit repair stratejileri
                    if "```" in llm_response:
                        # Markdown temizleme
                        llm_response = re.sub(r'```.*?```', '', llm_response, flags=re.DOTALL)
                    if llm_response.count('{') != llm_response.count('}'):
                        # Eksik parantezleri ekle
                        if llm_response.count('{') > llm_response.count('}'):
                            llm_response += '}' * (llm_response.count('{') - llm_response.count('}'))
                        else:
                            llm_response = '{' * (llm_response.count('}') - llm_response.count('{')) + llm_response
                    continue
                else:
                    logger.error(f"Failed to repair JSON after {max_repair_attempts} attempts")
                    return None
        
        return None

    def _validate_and_repair_question_data(self, q_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Tek bir soru verisini validate eder ve gerekirse onarır"""
        try:
            # Zorunlu alanları kontrol et
            required_fields = ["cloze_sentence", "correct_answer", "distractors", "original_sentence"]
            for field in required_fields:
                if field not in q_data or not q_data[field]:
                    logger.warning(f"Missing required field: {field}")
                    return None
            
            # Distractors array kontrolü
            if not isinstance(q_data.get("distractors"), list) or len(q_data["distractors"]) < 2:
                logger.warning("Invalid distractors array")
                return None
            
            # Correct answer distractors'da olmamalı
            if q_data["correct_answer"] in q_data["distractors"]:
                logger.warning("Correct answer found in distractors, removing duplicates")
                q_data["distractors"] = [d for d in q_data["distractors"] if d != q_data["correct_answer"]]
            
            # Default değerler
            q_data.setdefault("difficulty_level", 3)
            q_data.setdefault("error_type_addressed", "grammar")
            
            return q_data
            
        except Exception as e:
            logger.error(f"Error validating question data: {e}")
            return None

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
            
            llm_response = None
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

            if not llm_response or not llm_response.get("success", False):
                logger.error(f"LLM Gateway failed to generate cloze questions: {llm_response}")
                return []

            # 4. JSON response'u parse et ve onar
            raw_response = llm_response.get("parsed_json") or llm_response.get("data") or llm_response.get("response")
            if not raw_response:
                logger.error("No response data found in LLM response")
                return []

            # JSON repair
            generated_cloze_data = self._repair_json_response(raw_response)
            if not generated_cloze_data:
                logger.error("Failed to parse or repair JSON response")
                return []

            created_questions = []

            for q_data in generated_cloze_data:
                try:
                    # Validate and repair individual question data
                    repaired_q_data = self._validate_and_repair_question_data(q_data)
                    if not repaired_q_data:
                        logger.warning(f"Skipping invalid question data: {q_data}")
                        continue

                    # Validate each generated question against the schema
                    validated_q = ClozeQuestionSchema(**repaired_q_data)
                    
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