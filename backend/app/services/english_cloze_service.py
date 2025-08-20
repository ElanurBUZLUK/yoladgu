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
from app.services.metadata_schema_service import metadata_schema_service, ContentType, Domain

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
            # 1. Get last N errors for the user with embeddings
            recent_errors = await self._get_recent_student_errors(session, user_id, last_n_errors)
            
            if not recent_errors:
                logger.info(f"No recent errors found for user {user_id}, generating generic cloze questions")
                return await self._generate_generic_cloze_questions(session, num_questions)
            
            # 2. Find similar error patterns using embeddings
            similar_error_patterns = await self._find_similar_error_patterns_enhanced(recent_errors)
            
            # 3. Generate personalized cloze questions based on error patterns
            personalized_questions = []
            
            for error in recent_errors[:num_questions]:
                try:
                    # Generate embedding for the error
                    error_text = f"{error.get('error_text', '')} {error.get('error_type', '')}"
                    error_embedding = await self.embedding_service.get_embedding(error_text, domain="english")
                    
                    if not error_embedding:
                        logger.warning(f"Failed to generate embedding for error: {error_text}")
                        continue
                    
                    # Find similar error patterns in vector DB
                    similar_patterns = await self.vector_index_manager.search_similar_content(
                        error_embedding,
                        namespace="english_errors",
                        similarity_threshold=0.7,
                        limit=3
                    )
                    
                    # Generate cloze question using enhanced prompt with similar patterns
                    cloze_question = await self._generate_cloze_for_error_type_enhanced(
                        session, error, similar_patterns, error_embedding
                    )
                    
                    if cloze_question:
                        personalized_questions.append(cloze_question)
                        
                except Exception as e:
                    logger.error(f"Error generating personalized question for error {error.get('id')}: {e}")
                    continue
            
            # 4. If we don't have enough personalized questions, generate generic ones
            if len(personalized_questions) < num_questions:
                remaining_count = num_questions - len(personalized_questions)
                generic_questions = await self._generate_generic_cloze_questions(session, remaining_count)
                personalized_questions.extend(generic_questions)
            
            # 5. Store question embeddings in vector DB
            await self._store_question_embeddings_in_vector_db(personalized_questions, user_id)
            
            logger.info(f"✅ Generated {len(personalized_questions)} personalized cloze questions for user {user_id}")
            return personalized_questions[:num_questions]
            
        except Exception as e:
            logger.error(f"❌ Error generating cloze questions: {e}", exc_info=True)
            # Fallback to generic questions
            return await self._generate_generic_cloze_questions(session, num_questions)

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
                error_pattern, error_embedding
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
        error_pattern, 
        embedding: List[float]
    ):
        """Store error pattern embedding in vector database using standardized metadata"""
        try:
            # Use standardized metadata schema
            metadata = metadata_schema_service.build_error_pattern_metadata(
                domain=Domain.ENGLISH.value,
                error_type=error_pattern.error_type,
                obj_ref=str(error_pattern.id),
                user_id=str(error_pattern.user_id) if hasattr(error_pattern, 'user_id') else None,
                topic_category=getattr(error_pattern, 'topic_category', None),
                skill_tag=error_pattern.skill_tag,
                error_count=getattr(error_pattern, 'error_count', 1)
            )
            
            await self.vector_index_manager.batch_upsert_domain_embeddings_enhanced(
                domain="english",
                content_type="error_patterns",
                items=[{
                    "obj_ref": str(error_pattern.id),
                    "content": f"{error_pattern.error_type}: {error_pattern.error_text}",
                    "embedding": embedding,
                    "metadata": metadata
                }],
                batch_size=1
            )
            
            logger.info(f"✅ Stored error pattern embedding for {error_pattern.id} with standardized metadata")
            
        except Exception as e:
            logger.error(f"❌ Error storing error pattern embedding: {e}")

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
        user_id: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get recent student errors with embeddings"""
        try:
            # Get recent errors from database
            recent_errors = await self.question_repo.get_recent_errors_with_embeddings(
                session, user_id, limit=limit
            )
            
            errors_with_embeddings = []
            
            for error in recent_errors:
                error_data = {
                    "id": error.id,
                    "error_text": error.error_text,
                    "error_type": error.error_type,
                    "skill_tag": error.skill_tag,
                    "created_at": error.created_at,
                    "embedding": None
                }
                
                # Check if embedding already exists
                if hasattr(error, 'embedding') and error.embedding:
                    error_data["embedding"] = error.embedding
                else:
                    # Generate embedding if missing
                    try:
                        error_text = f"{error.error_text} {error.error_type} {error.skill_tag or ''}"
                        embedding = await self.embedding_service.get_embedding(error_text, domain="english")
                        
                        if embedding:
                            error_data["embedding"] = embedding
                            
                            # Store embedding in vector DB
                            await self._store_error_embedding_in_vector_db(error, embedding)
                            
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding for error {error.id}: {e}")
                
                errors_with_embeddings.append(error_data)
            
            return errors_with_embeddings
            
        except Exception as e:
            logger.error(f"Error getting recent student errors: {e}")
            return []

    async def _find_similar_error_patterns_enhanced(
        self, 
        recent_errors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find similar error patterns using enhanced embedding search"""
        try:
            similar_patterns = []
            
            for error in recent_errors:
                if not error.get("embedding"):
                    continue
                
                # Search for similar errors in vector DB
                similar_results = await self.vector_index_manager.search_similar_content(
                    error["embedding"],
                    namespace="english_errors",
                    similarity_threshold=0.7,
                    limit=5,
                    metadata_filters={
                        "domain": "english",
                        "error_type": error.get("error_type", "unknown")
                    }
                )
                
                for result in similar_results:
                    if result.get("obj_ref") != str(error.get("id")):
                        similar_patterns.append({
                            "error_text": result.get("content", ""),
                            "error_type": result.get("metadata", {}).get("error_type", "unknown"),
                            "similarity_score": result.get("similarity", 0.0),
                            "skill_tag": result.get("metadata", {}).get("skill_tag", "unknown")
                        })
            
            # Remove duplicates and sort by similarity
            unique_patterns = {}
            for pattern in similar_patterns:
                key = f"{pattern['error_type']}_{pattern['skill_tag']}"
                if key not in unique_patterns or pattern['similarity_score'] > unique_patterns[key]['similarity_score']:
                    unique_patterns[key] = pattern
            
            return sorted(unique_patterns.values(), key=lambda x: x['similarity_score'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error finding similar error patterns: {e}")
            return []

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
        error: Dict[str, Any],
        similar_patterns: List[Dict[str, Any]],
        error_embedding: List[float]
    ) -> Optional[Question]:
        """Generate cloze question using enhanced prompt with similar patterns"""
        try:
            # Build enhanced prompt with similar error patterns
            enhanced_prompt = self._build_enhanced_error_specific_prompt(error, similar_patterns)
            
            # Generate question using LLM
            llm_response = await self.llm_gateway.generate_json(
                prompt=enhanced_prompt,
                schema=TypeAdapter(ClozeQuestionSchema).json_schema(),
                max_retries=3
            )
            
            if not llm_response:
                logger.warning(f"LLM failed to generate question for error {error.get('id')}")
                return None
            
            # Parse and validate response
            raw_response = llm_response.get("parsed_json") or llm_response.get("data")
            if not raw_response:
                return None
            
            # Validate and repair question data
            validated_data = self._validate_and_repair_question_data(raw_response)
            if not validated_data:
                return None
            
            # Create Question object
            question = Question(
                subject=Subject.ENGLISH,
                question_type=QuestionType.FILL_BLANK,
                content=validated_data["cloze_sentence"],
                correct_answer=validated_data["correct_answer"],
                options=validated_data["distractors"] + [validated_data["correct_answer"]],
                difficulty_level=validated_data.get("difficulty_level", 3),
                explanation=validated_data.get("explanation", ""),
                metadata={
                    "error_type_addressed": error.get("error_type", "unknown"),
                    "skill_tag": error.get("skill_tag", "unknown"),
                    "similarity_score": similar_patterns[0]["similarity_score"] if similar_patterns else 0.0,
                    "generation_method": "embedding_enhanced",
                    "error_id": error.get("id")
                }
            )
            
            # Save question to database
            saved_question = await self.question_repo.create(session, question)
            
            # Store question embedding in vector DB
            question_text = f"{question.content} {question.correct_answer} {' '.join(question.options)}"
            question_embedding = await self.embedding_service.get_embedding(question_text, domain="english")
            
            if question_embedding:
                await self._store_question_embedding_in_vector_db(saved_question, question_embedding)
            
            return saved_question
            
        except Exception as e:
            logger.error(f"Error generating enhanced cloze question: {e}")
            return None

    def _build_enhanced_error_specific_prompt(
        self, 
        error: Dict[str, Any], 
        similar_patterns: List[Dict[str, Any]]
    ) -> str:
        """Build enhanced prompt with similar error patterns"""
        try:
            base_prompt = self._get_enhanced_cloze_system_prompt()
            
            # Add error context
            error_context = f"""
            STUDENT ERROR ANALYSIS:
            - Error Text: {error.get('error_text', 'N/A')}
            - Error Type: {error.get('error_type', 'N/A')}
            - Skill Tag: {error.get('skill_tag', 'N/A')}
            - Error Date: {error.get('created_at', 'N/A')}
            """
            
            # Add similar error patterns
            similar_context = ""
            if similar_patterns:
                similar_context = "\nSIMILAR ERROR PATTERNS FOUND:\n"
                for i, pattern in enumerate(similar_patterns[:3], 1):
                    similar_context += f"""
                    Pattern {i}:
                    - Error: {pattern.get('error_text', 'N/A')}
                    - Type: {pattern.get('error_type', 'N/A')}
                    - Similarity: {pattern.get('similarity_score', 0.0):.2f}
                    """
            
            # Build final prompt
            final_prompt = f"""
            {base_prompt}
            
            {error_context}
            
            {similar_context}
            
            INSTRUCTIONS:
            1. Analyze the student's error and similar patterns
            2. Generate a cloze question that specifically addresses this error type
            3. Use the error context to make the question relevant and challenging
            4. Ensure the distractors are plausible but incorrect
            5. Provide clear explanation of why the correct answer is right
            """
            
            return final_prompt.strip()
            
        except Exception as e:
            logger.error(f"Error building enhanced prompt: {e}")
            return self._get_enhanced_cloze_system_prompt()

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

    async def _store_question_embeddings_in_vector_db(
        self, 
        questions: List[Question], 
        user_id: str
    ):
        """Store question embeddings in vector database"""
        try:
            for question in questions:
                if not question.content:
                    continue
                
                # Generate question embedding
                question_text = f"{question.content} {question.correct_answer} {' '.join(question.options)}"
                question_embedding = await self.embedding_service.get_embedding(question_text, domain="english")
                
                if question_embedding:
                    await self._store_question_embedding_in_vector_db(question, question_embedding)
                    
        except Exception as e:
            logger.error(f"Error storing question embeddings: {e}")

    async def _store_question_embedding_in_vector_db(
        self, 
        question: Question, 
        embedding: List[float]
    ):
        """Store individual question embedding in vector database using standardized metadata"""
        try:
            # Use standardized metadata schema
            metadata = metadata_schema_service.build_cloze_question_metadata(
                domain=Domain.ENGLISH.value,
                obj_ref=str(question.id),
                error_type_addressed=question.metadata.get("error_type_addressed", "unknown"),
                skill_tag=question.metadata.get("skill_tag", "unknown"),
                difficulty_level=question.difficulty_level,
                generation_method=question.metadata.get("generation_method", "unknown"),
                question_id=str(question.id),
                question_type=question.question_type,
                subject=question.subject.value
            )
            
            await self.vector_index_manager.batch_upsert_domain_embeddings_enhanced(
                domain="english",
                content_type="cloze_questions",
                items=[{
                    "obj_ref": str(question.id),
                    "content": question.content,
                    "embedding": embedding,
                    "metadata": metadata
                }],
                batch_size=1
            )
            
            logger.info(f"✅ Stored question embedding for question {question.id} with standardized metadata")
            
        except Exception as e:
            logger.error(f"❌ Error storing question embedding: {e}")

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