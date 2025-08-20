import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import ValidationError, TypeAdapter
import json
import re
from datetime import datetime

from app.services.llm_gateway import llm_gateway
from app.services.question_generator import question_generator
from app.repositories.question_repository import QuestionRepository
from app.services.embedding_service import embedding_service
from app.services.vector_index_manager import vector_index_manager
from app.services.llm_context.builders.cloze_question_context_builder import ClozeQuestionContextBuilder
from app.services.llm_context.schemas.cloze_question_context import ClozeQuestionGenerationContext
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
        
        # Error recording and analysis settings
        self.error_analysis_config = {
            "max_recent_errors": 5,
            "similarity_threshold": 0.7,
            "min_errors_for_personalization": 3,
            "error_embedding_namespace": "english_errors",
            "question_embedding_namespace": "english_questions"
        }

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

    async def record_student_error(
        self,
        session: AsyncSession,
        user_id: str,
        error_text: str,
        error_type: str,
        error_context: str = "",
        question_id: Optional[str] = None,
        student_answer: str = ""
    ) -> Dict[str, Any]:
        """Record student error with real-time embedding and vector DB storage"""
        try:
            logger.info(f"Recording error for user {user_id}: {error_type}")
            
            # 1. Generate error embedding
            error_embedding = await embedding_service.get_embedding(
                text=error_text,
                domain="english",
                content_type="error_pattern"
            )
            
            # 2. Create error pattern record
            error_pattern = ErrorPattern(
                user_id=user_id,
                error_type=error_type,
                error_context=error_context,
                error_text=error_text,
                student_answer=student_answer,
                question_id=question_id,
                embedding=error_embedding,
                created_at=datetime.utcnow()
            )
            
            # 3. Save to database
            session.add(error_pattern)
            await session.commit()
            await session.refresh(error_pattern)
            
            # 4. Store in vector DB for real-time similarity search
            await self._store_error_embedding_in_vector_db(
                error_pattern_id=str(error_pattern.id),
                user_id=user_id,
                error_text=error_text,
                error_type=error_type,
                error_context=error_context,
                embedding=error_embedding,
                metadata={
                    "question_id": question_id,
                    "student_answer": student_answer,
                    "timestamp": error_pattern.created_at.isoformat()
                }
            )
            
            logger.info(f"✅ Error recorded and embedded for user {user_id}")
            
            return {
                "success": True,
                "error_pattern_id": str(error_pattern.id),
                "embedding_generated": True,
                "vector_db_stored": True
            }
            
        except Exception as e:
            logger.error(f"❌ Error recording failed for user {user_id}: {e}")
            await session.rollback()
            return {
                "success": False,
                "error": str(e)
            }

    async def _store_error_embedding_in_vector_db(
        self,
        error_pattern_id: str,
        user_id: str,
        error_text: str,
        error_type: str,
        error_context: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ):
        """Store error embedding in vector DB for real-time similarity search"""
        try:
            # Store in english_errors namespace
            await vector_index_manager.upsert_embedding(
                obj_ref=error_pattern_id,
                namespace=self.error_analysis_config["error_embedding_namespace"],
                embedding=embedding,
                metadata={
                    "domain": "english",
                    "content_type": "error_pattern",
                    "user_id": user_id,
                    "error_type": error_type,
                    "error_context": error_context,
                    "error_text": error_text,
                    "created_at": datetime.utcnow().isoformat(),
                    **metadata
                }
            )
            
            logger.info(f"✅ Error embedding stored in vector DB: {error_pattern_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to store error embedding in vector DB: {e}")

    async def generate_personalized_cloze_questions(
        self,
        session: AsyncSession,
        user_id: str,
        num_questions: int = 3,
        error_context: Optional[str] = None
    ) -> List[Question]:
        """Generate personalized cloze questions based on student's error patterns"""
        try:
            logger.info(f"🎯 Generating personalized cloze questions for user {user_id}")
            
            # 1. Get recent student errors with embeddings
            recent_errors = await self._get_recent_student_errors(session, user_id)
            
            if len(recent_errors) < self.error_analysis_config["min_errors_for_personalization"]:
                logger.info(f"Not enough errors for personalization ({len(recent_errors)}), generating generic questions")
                return await self._generate_generic_cloze_questions(session, num_questions)
            
            # 2. Find similar error patterns using semantic search
            similar_patterns = await self._find_similar_error_patterns_enhanced(recent_errors)
            
            # 3. Generate cloze questions for each error type
            generated_questions = []
            for pattern in similar_patterns[:num_questions]:
                try:
                    cloze_question = await self._generate_cloze_for_error_type_enhanced(
                        session, pattern, user_id, recent_errors
                    )
                    if cloze_question:
                        generated_questions.append(cloze_question)
                except Exception as e:
                    logger.error(f"Error generating cloze for pattern {pattern}: {e}")
                    continue
            
            # 4. If not enough questions generated, fill with generic ones
            if len(generated_questions) < num_questions:
                remaining_count = num_questions - len(generated_questions)
                generic_questions = await self._generate_generic_cloze_questions(session, remaining_count)
                generated_questions.extend(generic_questions)
            
            logger.info(f"✅ Successfully generated {len(generated_questions)} personalized cloze questions")
            return generated_questions
            
        except Exception as e:
            logger.error(f"❌ Error in personalized cloze generation: {e}", exc_info=True)
            # Fallback to generic generation
            return await self._generate_generic_cloze_questions(session, num_questions)

    async def _get_recent_student_errors(
        self, 
        session: AsyncSession, 
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Get recent student errors with embeddings"""
        try:
            # Get recent errors from database
            recent_errors = await session.execute(
                """
                SELECT ep.*, q.content as question_content, q.difficulty_level
                FROM error_patterns ep
                LEFT JOIN questions q ON ep.question_id = q.id
                WHERE ep.user_id = :user_id
                ORDER BY ep.created_at DESC
                LIMIT :limit
                """,
                {
                    "user_id": user_id,
                    "limit": self.error_analysis_config["max_recent_errors"]
                }
            )
            
            errors_with_embeddings = []
            for row in recent_errors:
                error_data = dict(row)
                
                # Generate embedding if missing
                if not error_data.get("embedding"):
                    error_text = f"{error_data['error_type']}: {error_data['error_text']}"
                    embedding = await embedding_service.get_embedding(
                        text=error_text,
                        domain="english",
                        content_type="error_pattern"
                    )
                    
                    # Update error pattern with embedding
                    await session.execute(
                        "UPDATE error_patterns SET embedding = :embedding WHERE id = :id",
                        {"embedding": embedding, "id": error_data["id"]}
                    )
                    error_data["embedding"] = embedding
                
                errors_with_embeddings.append(error_data)
            
            await session.commit()
            return errors_with_embeddings
            
        except Exception as e:
            logger.error(f"Error getting recent student errors: {e}")
            return []

    async def _find_similar_error_patterns_enhanced(
        self, 
        user_errors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find similar error patterns using enhanced semantic search"""
        try:
            similar_patterns = []
            
            for user_error in user_errors:
                if not user_error.get("embedding"):
                    continue
                
                # Search for similar patterns in the system
                similar_results = await vector_index_manager.search_similar_content(
                    user_error["embedding"],
                    namespace=self.error_analysis_config["error_embedding_namespace"],
                    similarity_threshold=self.error_analysis_config["similarity_threshold"],
                    limit=10,
                    metadata_filters={
                        "domain": "english",
                        "content_type": "error_pattern"
                    }
                )
                
                for result in similar_results:
                    # Calculate enhanced similarity score
                    enhanced_score = self._calculate_enhanced_similarity_score(
                        result, user_error
                    )
                    
                    similar_patterns.append({
                        "error_type": result.get("metadata", {}).get("error_type", user_error["error_type"]),
                        "similarity_score": enhanced_score,
                        "pattern_context": result.get("metadata", {}).get("error_context", ""),
                        "user_pattern": user_error,
                        "vector_result": result
                    })
            
            # Sort by enhanced similarity score and remove duplicates
            unique_patterns = {}
            for pattern in similar_patterns:
                key = pattern["error_type"]
                if key not in unique_patterns or pattern["similarity_score"] > unique_patterns[key]["similarity_score"]:
                    unique_patterns[key] = pattern
            
            # Return top patterns sorted by score
            sorted_patterns = sorted(
                unique_patterns.values(),
                key=lambda x: x["similarity_score"],
                reverse=True
            )
            
            logger.info(f"Found {len(sorted_patterns)} similar error patterns")
            return sorted_patterns
            
        except Exception as e:
            logger.error(f"Error finding similar error patterns: {e}")
            return user_errors

    def _calculate_enhanced_similarity_score(
        self,
        vector_result: Dict[str, Any],
        user_error: Dict[str, Any]
    ) -> float:
        """Calculate enhanced similarity score considering multiple factors"""
        try:
            base_similarity = vector_result.get("similarity", 0.0)
            
            # Get metadata from vector result
            metadata = vector_result.get("metadata", {})
            
            # Error type match bonus
            type_match_bonus = 0.0
            if metadata.get("error_type") == user_error.get("error_type"):
                type_match_bonus = 0.1
            
            # Difficulty level similarity bonus
            difficulty_bonus = 0.0
            user_difficulty = user_error.get("difficulty_level", 3)
            pattern_difficulty = metadata.get("difficulty_level", 3)
            if abs(user_difficulty - pattern_difficulty) <= 1:
                difficulty_bonus = 0.05
            
            # Context relevance bonus
            context_bonus = 0.0
            if metadata.get("error_context") and user_error.get("error_context"):
                if any(word in metadata["error_context"].lower() 
                       for word in user_error["error_context"].lower().split()):
                    context_bonus = 0.05
            
            # Calculate final enhanced score
            enhanced_score = min(1.0, base_similarity + type_match_bonus + difficulty_bonus + context_bonus)
            
            return enhanced_score
            
        except Exception as e:
            logger.error(f"Error calculating enhanced similarity score: {e}")
            return vector_result.get("similarity", 0.0)

    async def _generate_cloze_for_error_type_enhanced(
        self,
        session: AsyncSession,
        error_pattern: Dict[str, Any],
        user_id: str,
        recent_errors: List[Dict[str, Any]]
    ) -> Optional[Question]:
        """Generate enhanced cloze question for specific error type"""
        try:
            # Build enhanced context
            context = await self.context_builder.build_context(
                user_id=user_id,
                error_type=error_pattern["error_type"],
                error_context=error_pattern.get("pattern_context", ""),
                num_recent_errors=len(recent_errors)
            )
            
            # Generate cloze question using enhanced prompt
            prompt = self._build_enhanced_error_specific_prompt(error_pattern, context, recent_errors)
            system_prompt = self._get_enhanced_cloze_system_prompt()
            
            # Use LLM to generate cloze question
            llm_response = await self.llm_gateway.generate_json(
                prompt=prompt,
                system_prompt=system_prompt,
                schema=TypeAdapter(ClozeQuestionSchema).json_schema(),
                max_retries=3
            )
            
            if not llm_response or not llm_response.get("success", False):
                return None
            
            # Parse and validate response
            raw_response = llm_response.get("parsed_json") or llm_response.get("data")
            if not raw_response:
                return None
            
            # Validate against schema
            validated_q = ClozeQuestionSchema(**raw_response)
            
            # Create question for database
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
                    "error_type": error_pattern["error_type"],
                    "similarity_score": error_pattern.get("similarity_score", 0.0),
                    "generation_method": "enhanced_embedding_based",
                    "user_id": user_id,
                    "error_pattern_id": error_pattern.get("user_pattern", {}).get("id"),
                    "generated_at": datetime.utcnow().isoformat()
                }
            )
            
            # Save to database
            db_question = await self.question_repo.create(session, question_create)
            
            # Generate and store question embedding
            question_text = f"{validated_q.cloze_sentence} {validated_q.rule_context}"
            question_embedding = await embedding_service.get_embedding(
                text=question_text,
                domain="english",
                content_type="cloze_question"
            )
            
            # Store in vector DB
            await vector_index_manager.upsert_embedding(
                obj_ref=str(db_question.id),
                namespace=self.error_analysis_config["question_embedding_namespace"],
                embedding=question_embedding,
                metadata={
                    "domain": "english",
                    "content_type": "cloze_question",
                    "error_type": error_pattern["error_type"],
                    "difficulty_level": validated_q.difficulty_level,
                    "user_id": user_id,
                    "error_pattern_id": error_pattern.get("user_pattern", {}).get("id"),
                    "generation_method": "enhanced_embedding_based",
                    "created_at": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"✅ Enhanced cloze question generated and stored: {db_question.id}")
            return db_question
            
        except Exception as e:
            logger.error(f"❌ Error generating enhanced cloze: {e}")
            return None

    def _build_enhanced_error_specific_prompt(
        self, 
        error_pattern: Dict[str, Any], 
        context: ClozeQuestionGenerationContext,
        recent_errors: List[Dict[str, Any]]
    ) -> str:
        """Build enhanced prompt with detailed error analysis"""
        
        # Build error analysis section
        error_analysis = "Recent Error Analysis:\n"
        for i, error in enumerate(recent_errors[:3], 1):
            error_analysis += f"{i}. {error['error_type']}: {error['error_text'][:100]}...\n"
        
        return f"""
        Generate a personalized cloze question specifically for the error type: {error_pattern['error_type']}
        
        Error Context: {error_pattern.get('pattern_context', '')}
        Similarity Score: {error_pattern.get('similarity_score', 0.0):.2f}
        
        {error_analysis}
        
        User's Learning Context: {context.recent_errors}
        
        Create a cloze question that:
        1. Directly addresses the specific error type: {error_pattern['error_type']}
        2. Uses appropriate difficulty level based on user's performance
        3. Includes clear distractors that test understanding of the rule
        4. Provides helpful rule context for learning
        5. Incorporates patterns from similar errors to prevent repetition
        
        Focus on making this question directly relevant to the user's learning needs and error patterns.
        """

    def _get_enhanced_cloze_system_prompt(self) -> str:
        """Enhanced system prompt for cloze generation"""
        return """
        You are an expert English language teacher specializing in cloze question generation.
        
        Your task is to create personalized cloze questions that:
        - Address specific grammar/vocabulary errors
        - Are appropriate for the student's current level
        - Include realistic distractors
        - Provide clear learning context
        
        Always respond in the exact JSON schema format provided.
        Focus on creating questions that help students learn from their mistakes.
        """

    async def _generate_generic_cloze_questions(
        self, 
        session: AsyncSession, 
        num_questions: int
    ) -> List[Question]:
        """Generate generic cloze questions as fallback"""
        try:
            # Use the existing question generator for generic questions
            generic_questions = await question_generator.generate_question(
                subject="english",
                error_type="general_grammar",
                difficulty_level=3,
                student_context="Generic cloze generation"
            )
            
            if not generic_questions:
                return []
            
            # Convert to Question objects and save
            saved_questions = []
            for q_data in generic_questions[:num_questions]:
                try:
                    question_create = QuestionCreate(
                        subject=Subject.ENGLISH,
                        content=q_data["question"],
                        question_type=QuestionType.FILL_BLANK,
                        difficulty_level=q_data.get("difficulty_level", 3),
                        topic_category="general_grammar",
                        correct_answer=q_data["answer"],
                        options=q_data.get("options", []),
                        source_type=SourceType.GENERATED,
                        question_metadata={
                            "generation_method": "generic_fallback",
                            "error_type": "general_grammar"
                        }
                    )
                    
                    db_question = await self.question_repo.create(session, question_create)
                    saved_questions.append(db_question)
                    
                except Exception as e:
                    logger.error(f"Error saving generic question: {e}")
                    continue
            
            return saved_questions
            
        except Exception as e:
            logger.error(f"Error in generic cloze generation: {e}")
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