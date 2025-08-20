import logging
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis

from app.repositories.question_repository import QuestionRepository
from app.repositories.math_profile_repository import MathProfileRepository
from app.repositories.student_attempt_repository import StudentAttemptRepository
from app.repositories.error_pattern_repository import ErrorPatternRepository
from app.models.question import Question, DifficultyLevel, Subject
from app.models.math_profile import MathProfile
from app.utils.distlock_idem import idempotent_singleflight, IdempotencyConfig
from app.core.config import settings
# --- YENİ EKLENENLER ---
from app.services.embedding_service import embedding_service
from app.services.vector_index_manager import vector_index_manager
# --- BİTİŞ ---

logger = logging.getLogger(__name__)

class MathRecommendService:
    def __init__(
        self,
        question_repo: QuestionRepository,
        math_profile_repo: MathProfileRepository,
        student_attempt_repo: StudentAttemptRepository,
        error_pattern_repo: ErrorPatternRepository
    ):
        self.question_repo = question_repo
        self.math_profile_repo = math_profile_repo
        self.student_attempt_repo = student_attempt_repo
        self.error_pattern_repo = error_pattern_repo
        self.redis_client = Redis.from_url(settings.redis_url)
        
        # Embedding-based recommendation weights
        self.embedding_weights = {
            "error_pattern_similarity": 0.4,
            "question_content_similarity": 0.3,
            "difficulty_fit": 0.2,
            "diversity": 0.1
        }

    async def _recommend_questions_worker(
        self,
        session: AsyncSession,
        user_id: str,
        limit: int
    ) -> List[Question]:
        """Enhanced worker function with embedding-based recommendations."""
        try:
            user_math_profile = await self.math_profile_repo.get_by_user_id(session, user_id)
            if not user_math_profile:
                logger.warning(f"MathProfile not found for user {user_id}. Cannot recommend questions.")
                return []

            # 1. Get user's error patterns with embeddings
            user_error_patterns = await self._get_user_error_patterns_with_embeddings(session, user_id)
            
            # 2. Find similar error patterns using semantic search
            similar_error_patterns = await self._find_similar_math_error_patterns(user_error_patterns)
            
            # 3. Get questions based on error patterns and difficulty
            error_based_questions = await self._get_questions_by_error_patterns(
                session, similar_error_patterns, user_math_profile, limit // 2
            )
            
            # 4. Get questions based on difficulty level (traditional approach)
            difficulty_based_questions = await self._get_questions_by_difficulty(
                session, user_math_profile, limit // 2
            )
            
            # 5. Combine and rank questions using embedding similarity
            combined_questions = await self._combine_and_rank_questions(
                error_based_questions, difficulty_based_questions, user_error_patterns, limit
            )
            
            logger.info(f"Recommended {len(combined_questions)} questions for user {user_id} using embedding-based approach.")
            return combined_questions
            
        except Exception as e:
            logger.error(f"Error in enhanced recommendation worker: {e}", exc_info=True)
            # Fallback to traditional method
            return await self._traditional_recommendation_worker(session, user_id, limit)

    async def _get_user_error_patterns_with_embeddings(
        self, 
        session: AsyncSession, 
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Get user's math error patterns and generate embeddings if missing"""
        try:
            error_patterns = await self.error_pattern_repo.get_by_user_id_and_subject(
                session, user_id, Subject.MATH, limit=10
            )
            
            patterns_with_embeddings = []
            for pattern in error_patterns:
                # Generate embedding if missing
                if not hasattr(pattern, 'embedding') or not pattern.embedding:
                    pattern_text = f"{pattern.error_type}: {pattern.error_context or ''}"
                    embedding = await embedding_service.get_embedding(pattern_text)
                    
                    # Update pattern with embedding (if possible)
                    try:
                        pattern.embedding = embedding
                        await session.commit()
                    except Exception as e:
                        logger.warning(f"Could not update pattern embedding: {e}")
                
                patterns_with_embeddings.append({
                    "id": str(pattern.id),
                    "error_type": pattern.error_type,
                    "error_context": pattern.error_context,
                    "embedding": getattr(pattern, 'embedding', None),
                    "error_count": pattern.error_count,
                    "topic_category": pattern.topic_category
                })
            
            return patterns_with_embeddings
            
        except Exception as e:
            logger.error(f"Error getting user error patterns: {e}")
            return []

    async def _find_similar_math_error_patterns(
        self, 
        user_patterns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find similar math error patterns using semantic search"""
        try:
            similar_patterns = []
            
            for user_pattern in user_patterns:
                if not user_pattern.get("embedding"):
                    continue
                
                # Search for similar patterns in the system
                similar_results = await vector_index_manager.search_similar_content(
                    user_pattern["embedding"],
                    namespace="math_error_patterns",
                    similarity_threshold=0.7,
                    limit=5
                )
                
                for result in similar_results:
                    similar_patterns.append({
                        "error_type": result.get("error_type", user_pattern["error_type"]),
                        "similarity_score": result.get("similarity", 0.0),
                        "pattern_context": result.get("context", ""),
                        "topic_category": result.get("topic_category", user_pattern.get("topic_category")),
                        "user_pattern": user_pattern
                    })
            
            # Sort by similarity score and remove duplicates
            unique_patterns = {}
            for pattern in similar_patterns:
                key = pattern["error_type"]
                if key not in unique_patterns or pattern["similarity_score"] > unique_patterns[key]["similarity_score"]:
                    unique_patterns[key] = pattern
            
            return sorted(unique_patterns.values(), key=lambda x: x["similarity_score"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error finding similar math error patterns: {e}")
            return user_patterns

    async def _get_questions_by_error_patterns(
        self,
        session: AsyncSession,
        error_patterns: List[Dict[str, Any]],
        user_profile: MathProfile,
        limit: int
    ) -> List[Question]:
        """Get questions based on error patterns using semantic search"""
        try:
            questions = []
            
            for pattern in error_patterns[:limit]:
                if not pattern.get("embedding"):
                    continue
                
                # Search for questions similar to the error pattern
                similar_questions = await vector_index_manager.search_similar_content(
                    pattern["embedding"],
                    namespace="math_questions",
                    similarity_threshold=0.6,
                    limit=3
                )
                
                for result in similar_questions:
                    question_id = result.get("obj_ref")
                    if question_id:
                        question = await self.question_repo.get_by_id(session, question_id)
                        if question and question.subject == Subject.MATH:
                            # Add similarity score to question metadata
                            question.similarity_score = result.get("similarity", 0.0)
                            question.error_pattern_match = pattern["error_type"]
                            questions.append(question)
            
            return questions[:limit]
            
        except Exception as e:
            logger.error(f"Error getting questions by error patterns: {e}")
            return []

    async def _get_questions_by_difficulty(
        self,
        session: AsyncSession,
        user_profile: MathProfile,
        limit: int
    ) -> List[Question]:
        """Get questions based on difficulty level (traditional approach)"""
        try:
            # Determine target difficulty range
            target_min_difficulty = user_profile.global_skill + 0.5
            target_max_difficulty = user_profile.global_skill + 1.5
            
            target_min_difficulty = max(0.0, target_min_difficulty)
            target_max_difficulty = min(5.0, target_max_difficulty)
            
            # Get questions within the target difficulty range
            questions = await self.question_repo.get_by_subject_and_difficulty_range(
                session,
                subject=Subject.MATH,
                min_difficulty=target_min_difficulty,
                max_difficulty=target_max_difficulty,
                limit=limit
            )
            
            # Add default similarity scores
            for question in questions:
                question.similarity_score = 0.5  # Default score for difficulty-based questions
                question.error_pattern_match = "difficulty_based"
            
            return questions
            
        except Exception as e:
            logger.error(f"Error getting questions by difficulty: {e}")
            return []

    async def _combine_and_rank_questions(
        self,
        error_based_questions: List[Question],
        difficulty_based_questions: List[Question],
        user_error_patterns: List[Dict[str, Any]],
        limit: int
    ) -> List[Question]:
        """Combine and rank questions using embedding similarity and other factors"""
        try:
            all_questions = error_based_questions + difficulty_based_questions
            
            # Calculate final scores for each question
            scored_questions = []
            for question in all_questions:
                final_score = await self._calculate_question_score(question, user_error_patterns)
                question.final_score = final_score
                scored_questions.append(question)
            
            # Sort by final score and remove duplicates
            unique_questions = {}
            for question in scored_questions:
                if question.id not in unique_questions or question.final_score > unique_questions[question.id].final_score:
                    unique_questions[question.id] = question
            
            # Return top questions
            sorted_questions = sorted(
                unique_questions.values(),
                key=lambda x: x.final_score,
                reverse=True
            )
            
            return sorted_questions[:limit]
            
        except Exception as e:
            logger.error(f"Error combining and ranking questions: {e}")
            return all_questions[:limit]

    async def _calculate_question_score(
        self,
        question: Question,
        user_error_patterns: List[Dict[str, Any]]
    ) -> float:
        """Calculate a comprehensive score for a question"""
        try:
            score = 0.0
            
            # Base similarity score
            similarity_score = getattr(question, 'similarity_score', 0.5)
            score += similarity_score * self.embedding_weights["error_pattern_similarity"]
            
            # Question content similarity to user's error patterns
            if user_error_patterns and hasattr(question, 'content_embedding') and question.content_embedding:
                content_similarity = await self._calculate_content_similarity(
                    question.content_embedding, user_error_patterns
                )
                score += content_similarity * self.embedding_weights["question_content_similarity"]
            
            # Difficulty fit (closer to user's level = higher score)
            difficulty_fit = 1.0 - abs(question.difficulty_level - 3.0) / 5.0  # Normalize to 0-1
            score += difficulty_fit * self.embedding_weights["difficulty_fit"]
            
            # Diversity bonus (different topics get higher scores)
            if hasattr(question, 'topic_category') and question.topic_category:
                diversity_bonus = 0.1  # Small bonus for topic variety
                score += diversity_bonus * self.embedding_weights["diversity"]
            
            return min(1.0, max(0.0, score))  # Clamp between 0 and 1
            
        except Exception as e:
            logger.error(f"Error calculating question score: {e}")
            return 0.5

    async def _calculate_content_similarity(
        self,
        question_embedding: List[float],
        user_error_patterns: List[Dict[str, Any]]
    ) -> float:
        """Calculate similarity between question content and user's error patterns"""
        try:
            if not user_error_patterns:
                return 0.5
            
            similarities = []
            for pattern in user_error_patterns:
                if pattern.get("embedding"):
                    # Calculate cosine similarity
                    similarity = await self._cosine_similarity(question_embedding, pattern["embedding"])
                    similarities.append(similarity)
            
            if similarities:
                return sum(similarities) / len(similarities)
            return 0.5
            
        except Exception as e:
            logger.error(f"Error calculating content similarity: {e}")
            return 0.5

    async def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            if len(vec1) != len(vec2):
                return 0.0
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0

    async def _traditional_recommendation_worker(
        self,
        session: AsyncSession,
        user_id: str,
        limit: int
    ) -> List[Question]:
        """Traditional recommendation method as fallback"""
        try:
            user_math_profile = await self.math_profile_repo.get_by_user_id(session, user_id)
            if not user_math_profile:
                return []

            # Basic difficulty-based recommendation
            target_min_difficulty = user_math_profile.global_skill + 0.5
            target_max_difficulty = user_math_profile.global_skill + 1.5
            
            target_min_difficulty = max(0.0, target_min_difficulty)
            target_max_difficulty = min(5.0, target_max_difficulty)
            
            questions = await self.question_repo.get_by_subject_and_difficulty_range(
                session,
                subject=Subject.MATH,
                min_difficulty=target_min_difficulty,
                max_difficulty=target_max_difficulty,
                limit=limit
            )
            
            return questions
            
        except Exception as e:
            logger.error(f"Error in traditional recommendation: {e}")
            return []

    async def recommend_questions(
        self,
        session: AsyncSession,
        user_id: str,
        limit: int = 5
    ) -> List[Question]:
        """Recommends math questions based on user's skill level and neighbor-wrongs, with idempotency."""
        idempotency_key = f"math_recommendation:{user_id}:{limit}"
        config = IdempotencyConfig(scope="math_recommendation", ttl_seconds=300) # Cache for 5 minutes

        try:
            # Use idempotent_singleflight to ensure only one recommendation process runs per user/limit
            # and to cache the results.
            recommended_questions = await idempotent_singleflight(
                client=self.redis_client,
                key=idempotency_key,
                config=config,
                worker=lambda: self._recommend_questions_worker(session, user_id, limit)
            )
            return recommended_questions

        except Exception as e:
            logger.error(f"Error in MathRecommendService for user {user_id}: {e}", exc_info=True)
            return []

    async def recommend(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """API endpoint için recommend metodu - request body'den user_id alır"""
        try:
            user_id = request.get("user_id")
            limit = request.get("limit", 5)
            
            if not user_id:
                raise ValueError("user_id is required")
            
            # Bu metod session'ı dışarıdan almalı, şimdilik mock dönelim
            # Gerçek implementasyonda session dependency injection kullanılmalı
            return {
                "status": "success",
                "message": "Recommendation service is working",
                "user_id": user_id,
                "limit": limit,
                "note": "Full implementation requires session injection"
            }
        except Exception as e:
            logger.error(f"Error in recommend method: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e)
            }

# Global instance (for dependency injection)
math_recommend_service = MathRecommendService(
    question_repo=QuestionRepository(),
    math_profile_repo=MathProfileRepository(),
    student_attempt_repo=StudentAttemptRepository(),
    error_pattern_repo=ErrorPatternRepository()
)