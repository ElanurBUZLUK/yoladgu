import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import ValidationError, TypeAdapter
import json
import re

from app.services.llm_gateway import llm_gateway
from app.services.question_generator import question_generator
from app.repositories.question_repository import QuestionRepository
from app.services.llm_context.builders.cloze_question_context_builder import ClozeQuestionContextBuilder
from app.services.llm_context.schemas.cloze_question_context import ClozeQuestionGenerationContext
from app.models.question import Question, Subject, QuestionType, SourceType
from app.schemas.question import ClozeQuestionSchema, QuestionCreate
from app.models.error_pattern import ErrorPattern
from app.repositories.error_log_repository import error_log_repository
from app.services.error_classifier_service import error_classifier_service

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
        num_questions: int = 3,
        last_n_errors: int = 5,
        target_error_type: Optional[str] = None
    ) -> List[Question]:
        """
        Generate personalized English cloze questions using hybrid QG system (templates + GPT).
        
        Args:
            session: Database session
            user_id: User ID
            num_questions: Number of questions to generate
            last_n_errors: Number of recent errors to consider
            target_error_type: Specific error type to target (optional)
            
        Returns:
            List of generated Question objects
        """
        try:
            # 1. Get user's recent error patterns
            recent_errors = await error_log_repository.get_recent_errors(
                session, user_id, last_n_errors
            )
            
            if not recent_errors:
                logger.info(f"No recent errors found for user {user_id}, using default generation")
                # Generate generic questions using QG system
                return await self._generate_generic_cloze_questions_qg(session, num_questions)
            
            # 2. Analyze error patterns to determine target error types
            error_types = []
            for error in recent_errors:
                if error.error_type:
                    error_types.append(error.error_type)
                elif error.error_description:
                    # Use error classifier to determine type
                    classification = await error_classifier_service.classify_error(
                        error.error_description
                    )
                    if classification and classification.error_type:
                        error_types.append(classification.error_type)
            
            # Remove duplicates and get most common
            if error_types:
                from collections import Counter
                error_type_counts = Counter(error_types)
                target_error_types = [error_type for error_type, count in error_type_counts.most_common(3)]
            else:
                target_error_types = ["general_grammar_error"]
            
            # 3. Use QG system to generate questions
            created_questions = []
            
            for error_type in target_error_types:
                try:
                    # Map error types to QG system error types
                    qg_error_type = self._map_error_type_to_qg(error_type)
                    
                    # Generate question using QG system
                    generated_question = await question_generator.generate_question(
                        subject="english",
                        error_type=qg_error_type,
                        difficulty_level=3,  # Default level, could be personalized
                        use_gpt=True,
                        student_context=f"Student struggles with {error_type}: {[e.error_description for e in recent_errors if e.error_type == error_type][:2]}"
                    )
                    
                    if generated_question:
                        # Convert QG response to Question object
                        question = await self._convert_qg_to_question(session, generated_question, user_id)
                        if question:
                            created_questions.append(question)
                        
                        # Stop if we have enough questions
                        if len(created_questions) >= num_questions:
                            break
                            
                except Exception as e:
                    logger.warning(f"Failed to generate question for error type {error_type}: {e}")
                    continue
            
            # 4. If QG system didn't generate enough questions, fallback to LLM
            if len(created_questions) < num_questions:
                logger.info("QG system didn't generate enough questions, falling back to LLM")
                fallback_questions = await self._generate_llm_fallback_questions(
                    session, target_error_types, num_questions - len(created_questions)
                )
                created_questions.extend(fallback_questions)
            
            logger.info(f"Successfully generated {len(created_questions)} cloze questions using QG system.")
            return created_questions
            
        except Exception as e:
            logger.error(f"Error in EnglishClozeService with QG: {e}", exc_info=True)
            # Fallback to original method
            return await self._generate_llm_fallback_questions(session, ["general_grammar_error"], num_questions)

    async def _generate_generic_cloze_questions_qg(self, session: AsyncSession, num_questions: int) -> List[Question]:
        """Generate generic questions using QG system"""
        try:
            created_questions = []
            
            # Use common error types for generic generation
            common_error_types = ["present_perfect_error", "preposition_error", "verb_tense_error"]
            
            for error_type in common_error_types:
                try:
                    generated_question = await question_generator.generate_question(
                        subject="english",
                        error_type=error_type,
                        difficulty_level=2,  # Basic level for generic questions
                        use_gpt=False,  # Use templates only for generic questions
                        student_context="General practice for English learners"
                    )
                    
                    if generated_question:
                        question = await self._convert_qg_to_question(session, generated_question, "generic_user")
                        if question:
                            created_questions.append(question)
                        
                        if len(created_questions) >= num_questions:
                            break
                            
                except Exception as e:
                    logger.warning(f"Failed to generate generic question for {error_type}: {e}")
                    continue
            
            return created_questions
            
        except Exception as e:
            logger.error(f"Error in generic QG generation: {e}")
            return []

    async def _convert_qg_to_question(self, session: AsyncSession, qg_data: Dict[str, Any], user_id: str) -> Optional[Question]:
        """Convert QG system response to Question object"""
        try:
            # Create QuestionCreate schema from QG data
            question_create = QuestionCreate(
                subject=Subject.ENGLISH,
                content=qg_data.get("question", ""),
                question_type=QuestionType.FILL_BLANK,
                difficulty_level=qg_data.get("difficulty_level", 3),
                topic_category=qg_data.get("error_type", "general_grammar"),
                correct_answer=qg_data.get("answer", ""),
                options=qg_data.get("options", []) + [qg_data.get("answer", "")],
                source_type=SourceType.GENERATED,
                question_metadata={
                    "generation_method": qg_data.get("type", "unknown"),
                    "explanation": qg_data.get("explanation", ""),
                    "error_type": qg_data.get("error_type", ""),
                    "qg_metadata": qg_data.get("metadata", {})
                }
            )
            
            # Save to database
            db_question = await self.question_repo.create(session, question_create)
            return db_question
            
        except Exception as e:
            logger.error(f"Error converting QG data to Question: {e}")
            return None

    def _map_error_type_to_qg(self, error_type: str) -> str:
        """Map internal error types to QG system error types"""
        error_type_mapping = {
            "present_perfect": "present_perfect_error",
            "preposition": "preposition_error",
            "verb_tense": "verb_tense_error",
            "article": "article_error",
            "word_order": "word_order_error",
            "general_grammar": "verb_tense_error"  # Default fallback
        }
        
        return error_type_mapping.get(error_type.lower(), "verb_tense_error")

    async def _generate_llm_fallback_questions(self, session: AsyncSession, target_error_types: List[str], num_questions: int) -> List[Question]:
        """Fallback to original LLM-based generation method"""
        try:
            # Original LLM generation logic here
            prompt = f"""
            Generate {num_questions} English cloze questions targeting these error types: {', '.join(target_error_types)}
            
            Each question should:
            - Be a fill-in-the-blank sentence
            - Target the specific grammar rule the student struggles with
            - Include 3-4 distractors (wrong answers)
            - Have clear explanations
            - Be appropriate for intermediate English learners
            
            Format as JSON array with fields: cloze_sentence, correct_answer, distractors, original_sentence, error_type_addressed, difficulty_level, rule_context
            """
            
            system_prompt = """You are an expert English language teacher. Generate clear, educational cloze questions that help students learn from their mistakes."""
            
            # Use LLM gateway
            llm_response = await self.llm_gateway.generate_json(
                prompt=prompt,
                system_prompt=system_prompt,
                schema=TypeAdapter(List[ClozeQuestionSchema]).json_schema(),
                max_retries=3
            )
            
            if not llm_response or not llm_response.get("success", False):
                logger.error(f"LLM Gateway failed to generate cloze questions: {llm_response}")
                return []

            # Parse and process response
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
                        topic_category=validated_q.error_type_addressed,
                        correct_answer=validated_q.correct_answer,
                        options=validated_q.distractors + [validated_q.correct_answer],
                        source_type=SourceType.GENERATED,
                        question_metadata={
                            "original_sentence": validated_q.original_sentence,
                            "rule_context": validated_q.rule_context,
                            "generation_method": "llm_fallback"
                        }
                    )
                    db_question = await self.question_repo.create(session, question_create)
                    created_questions.append(db_question)
                except ValidationError as e:
                    logger.error(f"Validation error for generated cloze question: {e.errors()}")
                except Exception as e:
                    logger.error(f"Error saving generated cloze question to DB: {e}")

            return created_questions
            
        except Exception as e:
            logger.error(f"Error in LLM fallback generation: {e}")
            return []

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
                logger.warning("Correct answer found in distractors, removing")
                q_data["distractors"] = [d for d in q_data["distractors"] if d != q_data["correct_answer"]]
            
            return q_data
            
        except Exception as e:
            logger.error(f"Error in question data validation: {e}")
            return None

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
