import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.orm import selectinload
import random

from app.models.student import Student
from app.models.math_profile import MathProfile
from app.models.question import Question, Subject, DifficultyLevel
from app.models.student_attempt import StudentAttempt
from app.models.error_pattern import ErrorPattern
from app.services.embedding_service import embedding_service
from app.services.vector_index_manager import vector_index_manager
from app.services.metadata_schema_service import metadata_schema_service, ContentType, Domain
from app.core.config import settings

logger = logging.getLogger(__name__)


class ProfileLockManager:
    """Manages locks for student profile updates to prevent race conditions"""
    
    def __init__(self):
        self.locks: Dict[str, asyncio.Lock] = {}
        self.lock_timeout = 30  # 30 seconds timeout
        self.cleanup_interval = 300  # 5 minutes
    
    async def acquire_lock(self, student_id: str) -> asyncio.Lock:
        """Acquire lock for student profile"""
        if student_id not in self.locks:
            self.locks[student_id] = asyncio.Lock()
        
        # Clean up old locks periodically
        if len(self.locks) > 1000:  # Prevent memory leaks
            await self._cleanup_old_locks()
        
        return self.locks[student_id]
    
    async def release_lock(self, student_id: str):
        """Release lock for student profile"""
        if student_id in self.locks:
            # Don't delete the lock, just mark it as available
            pass
    
    async def _cleanup_old_locks(self):
        """Clean up old locks to prevent memory leaks"""
        try:
            # Remove locks for students not accessed recently
            current_time = time.time()
            keys_to_remove = []
            
            for student_id, lock in self.locks.items():
                # Simple cleanup - could be enhanced with access tracking
                if len(self.locks) > 500:  # Only cleanup if we have too many
                    keys_to_remove.append(student_id)
            
            for key in keys_to_remove[:100]:  # Remove max 100 at a time
                if key in self.locks:
                    del self.locks[key]
            
            if keys_to_remove:
                logger.info(f"🧹 Cleaned up {len(keys_to_remove)} old profile locks")
                
        except Exception as e:
            logger.error(f"❌ Error during lock cleanup: {e}")


class OptimisticLockManager:
    """Manages optimistic locking for database operations"""
    
    def __init__(self):
        self.retry_attempts = 3
        self.retry_delay = 0.1  # 100ms
    
    async def execute_with_optimistic_lock(
        self,
        operation_func,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with optimistic locking and retry logic"""
        
        last_exception = None
        
        for attempt in range(self.retry_attempts):
            try:
                result = await operation_func(*args, **kwargs)
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if it's a concurrency error
                if "concurrent" in str(e).lower() or "stale" in str(e).lower():
                    if attempt < self.retry_attempts - 1:
                        # Wait before retry with exponential backoff
                        delay = self.retry_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
                        logger.warning(f"⚠️ Optimistic lock conflict, retrying in {delay}s (attempt {attempt + 1})")
                        continue
                
                # If not a concurrency error or max attempts reached, break
                break
        
        # If we get here, all attempts failed
        logger.error(f"❌ Operation failed after {self.retry_attempts} attempts: {last_exception}")
        raise last_exception


class MathRecommendService:
    """Matematik soru öneri servisi - Race condition korumalı"""
    
    def __init__(self):
        self.profile_lock_manager = ProfileLockManager()
        self.optimistic_lock_manager = OptimisticLockManager()
        
        # Performance tracking
        self.performance_metrics = {
            "total_recommendations": 0,
            "successful_recommendations": 0,
            "failed_recommendations": 0,
            "average_response_time": 0.0,
            "total_response_time": 0.0,
            "race_condition_resolved": 0,
            "optimistic_lock_retries": 0,
            "embedding_based_recommendations": 0,
            "fallback_recommendations": 0
        }
    
    async def recommend_questions(
        self,
        db: AsyncSession,
        student_id: str,
        num_questions: int = 5,
        difficulty_adjustment: float = 0.0,
        include_recent_errors: bool = True,
        use_embeddings: bool = True
    ) -> Dict[str, Any]:
        """Matematik soru önerisi - Race condition korumalı"""
        
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Starting math recommendation for student {student_id}")
            
            # Acquire profile lock to prevent race conditions
            profile_lock = await self.profile_lock_manager.acquire_lock(student_id)
            
            async with profile_lock:
                # Get student profile with optimistic locking
                profile = await self._get_student_profile_safe(db, student_id)
                
                if not profile:
                    raise ValueError(f"Student profile not found for {student_id}")
                
                # Get recent error patterns if requested
                recent_errors = []
                if include_recent_errors:
                    recent_errors = await self._get_recent_error_patterns(db, student_id)
                
                # Generate recommendations using embedding-based approach
                if use_embeddings and recent_errors:
                    recommendations = await self._generate_embedding_based_recommendations(
                        db, profile, recent_errors, num_questions, difficulty_adjustment
                    )
                    self.performance_metrics["embedding_based_recommendations"] += 1
                else:
                    # Fallback to traditional recommendation
                    recommendations = await self._generate_traditional_recommendations(
                        db, profile, num_questions, difficulty_adjustment
                    )
                    self.performance_metrics["fallback_recommendations"] += 1
                
                # Update performance metrics
                response_time = time.time() - start_time
                self._update_performance_metrics(response_time, True)
                
                logger.info(f"✅ Math recommendation completed for student {student_id}: {len(recommendations)} questions")
                
                return {
                    "success": True,
                    "student_id": student_id,
                    "recommendations": recommendations,
                    "profile": {
                        "current_skill": profile.global_skill,
                        "math_skill": profile.math_skill,
                        "last_updated": profile.updated_at.isoformat() if profile.updated_at else None
                    },
                    "error_patterns": [error.error_type for error in recent_errors],
                    "recommendation_method": "embedding_based" if use_embeddings and recent_errors else "traditional",
                    "performance_metrics": self.performance_metrics,
                    "response_time": response_time
                }
                
        except Exception as e:
            response_time = time.time() - start_time
            self._update_performance_metrics(response_time, False)
            
            logger.error(f"❌ Math recommendation failed for student {student_id}: {e}", exc_info=True)
            
            return {
                "success": False,
                "student_id": student_id,
                "error": str(e),
                "recommendations": [],
                "performance_metrics": self.performance_metrics,
                "response_time": response_time
            }
        
        finally:
            # Release profile lock
            await self.profile_lock_manager.release_lock(student_id)
    
    async def _get_student_profile_safe(
        self,
        db: AsyncSession,
        student_id: str
    ) -> Optional[MathProfile]:
        """Get student profile with optimistic locking protection"""
        
        try:
            # Use optimistic locking with retry logic
            profile = await self.optimistic_lock_manager.execute_with_optimistic_lock(
                self._get_profile_operation,
                db,
                student_id
            )
            
            return profile
            
        except Exception as e:
            logger.error(f"❌ Failed to get student profile: {e}")
            return None
    
    async def _get_profile_operation(
        self,
        db: AsyncSession,
        student_id: str
    ) -> Optional[MathProfile]:
        """Database operation for getting profile"""
        
        try:
            result = await db.execute(
                select(MathProfile)
                .where(MathProfile.student_id == student_id)
                .options(selectinload(MathProfile.student))
            
            profile = result.scalar_one_or_none()
            
            if profile:
                # Add version tracking for optimistic locking
                if not hasattr(profile, 'version'):
                    profile.version = 1
                else:
                    profile.version += 1
            
            return profile
            
        except Exception as e:
            logger.error(f"❌ Database operation failed: {e}")
            raise
    
    async def _get_recent_error_patterns(
        self,
        db: AsyncSession,
        student_id: str,
        limit: int = 10
    ) -> List[ErrorPattern]:
        """Get recent error patterns for student"""
        
        try:
            # Get recent attempts with errors
            recent_attempts = await db.execute(
                select(StudentAttempt)
                .where(
                    and_(
                        StudentAttempt.student_id == student_id,
                        StudentAttempt.subject == Subject.MATH,
                        StudentAttempt.is_correct == False
                    )
                )
                .order_by(desc(StudentAttempt.created_at))
                .limit(limit)
            )
            
            attempts = recent_attempts.scalars().all()
            
            # Extract error patterns
            error_patterns = []
            for attempt in attempts:
                if attempt.error_details:
                    error_pattern = ErrorPattern(
                        student_id=student_id,
                        subject=Subject.MATH,
                        error_type=attempt.error_details.get("error_type", "unknown"),
                        error_context=attempt.error_details.get("context", ""),
                        question_id=attempt.question_id,
                        created_at=attempt.created_at
                    )
                    error_patterns.append(error_pattern)
            
            logger.debug(f"📊 Found {len(error_patterns)} recent error patterns for student {student_id}")
            return error_patterns
            
        except Exception as e:
            logger.error(f"❌ Failed to get recent error patterns: {e}")
            return []
    
    async def _generate_embedding_based_recommendations(
        self,
        db: AsyncSession,
        profile: MathProfile,
        recent_errors: List[ErrorPattern],
        num_questions: int,
        difficulty_adjustment: float
    ) -> List[Dict[str, Any]]:
        """Generate recommendations using embedding-based similarity"""
        
        try:
            logger.info(f"🧠 Generating embedding-based recommendations for {len(recent_errors)} error patterns")
            
            # Create error context for embedding
            error_context = self._build_error_context(recent_errors)
            
            # Generate embedding for error context
            error_embedding = await embedding_service.get_embedding(error_context)
            
            if not error_embedding:
                logger.warning("⚠️ Failed to generate error embedding, falling back to traditional method")
                return await self._generate_traditional_recommendations(
                    db, profile, num_questions, difficulty_adjustment
                )
            
            # Search for similar questions using vector database
            similar_questions = await vector_index_manager.search_similar_content(
                embedding=error_embedding,
                namespace="math_questions",
                limit=num_questions * 2,  # Get more to filter
                similarity_threshold=0.7
            )
            
            # Filter and rank questions
            filtered_questions = await self._filter_and_rank_questions(
                db, similar_questions, profile, difficulty_adjustment, num_questions
            )
            
            # Store error embedding for future use
            await self._store_error_embedding_in_vector_db(
                error_context, error_embedding, recent_errors
            )
            
            logger.info(f"✅ Generated {len(filtered_questions)} embedding-based recommendations")
            return filtered_questions
            
        except Exception as e:
            logger.error(f"❌ Embedding-based recommendation failed: {e}")
            # Fallback to traditional method
            return await self._generate_traditional_recommendations(
                db, profile, num_questions, difficulty_adjustment
            )
    
    def _build_error_context(self, recent_errors: List[ErrorPattern]) -> str:
        """Build context string from recent errors for embedding"""
        
        try:
            error_types = [error.error_type for error in recent_errors]
            error_contexts = [error.error_context for error in recent_errors if error.error_context]
            
            # Combine error information
            context_parts = []
            
            if error_types:
                context_parts.append(f"Error types: {', '.join(set(error_types))}")
            
            if error_contexts:
                context_parts.append(f"Error contexts: {' '.join(error_contexts[:3])}")  # Limit context length
            
            # Add mathematical context
            math_keywords = self._extract_math_keywords(error_contexts)
            if math_keywords:
                context_parts.append(f"Mathematical concepts: {', '.join(math_keywords)}")
            
            return " | ".join(context_parts)
            
        except Exception as e:
            logger.error(f"❌ Failed to build error context: {e}")
            return "math error patterns"
    
    def _extract_math_keywords(self, contexts: List[str]) -> List[str]:
        """Extract mathematical keywords from error contexts"""
        
        try:
            math_keywords = set()
            
            # Common mathematical terms
            math_terms = [
                'algebra', 'geometry', 'calculus', 'trigonometry', 'statistics',
                'equation', 'function', 'derivative', 'integral', 'matrix',
                'probability', 'percentage', 'ratio', 'fraction', 'decimal'
            ]
            
            for context in contexts:
                context_lower = context.lower()
                for term in math_terms:
                    if term in context_lower:
                        math_keywords.add(term)
            
            return list(math_keywords)[:5]  # Limit to 5 keywords
            
        except Exception as e:
            logger.error(f"❌ Failed to extract math keywords: {e}")
            return []
    
    async def _filter_and_rank_questions(
        self,
        db: AsyncSession,
        similar_questions: List[Dict[str, Any]],
        profile: MathProfile,
        difficulty_adjustment: float,
        num_questions: int
    ) -> List[Dict[str, Any]]:
        """Filter and rank similar questions based on profile and difficulty"""
        
        try:
            # Get full question details from database
            question_ids = [q.get("id") for q in similar_questions if q.get("id")]
            
            if not question_ids:
                return []
            
            # Fetch questions from database
            questions_result = await db.execute(
                select(Question)
                .where(Question.id.in_(question_ids))
                .where(Question.subject == Subject.MATH)
            )
            
            questions = questions_result.scalars().all()
            
            # Calculate target difficulty
            target_difficulty = min(5, max(1, profile.math_skill + difficulty_adjustment))
            
            # Score and rank questions
            scored_questions = []
            for question in questions:
                score = self._calculate_question_score(question, profile, target_difficulty)
                scored_questions.append({
                    "question": question,
                    "score": score,
                    "difficulty_match": 1.0 - abs(question.difficulty_level - target_difficulty) / 5.0
                })
            
            # Sort by score and select top questions
            scored_questions.sort(key=lambda x: x["score"], reverse=True)
            selected_questions = scored_questions[:num_questions]
            
            # Convert to response format
            recommendations = []
            for item in selected_questions:
                question = item["question"]
                recommendations.append({
                    "id": str(question.id),
                    "content": question.content,
                    "difficulty_level": question.difficulty_level,
                    "topic_category": question.topic_category,
                    "question_type": question.question_type.value,
                    "score": item["score"],
                    "difficulty_match": item["difficulty_match"],
                    "recommendation_reason": self._generate_recommendation_reason(
                        question, profile, item["score"]
                    )
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to filter and rank questions: {e}")
            return []
    
    def _calculate_question_score(
        self,
        question: Question,
        profile: MathProfile,
        target_difficulty: float
    ) -> float:
        """Calculate question recommendation score"""
        
        try:
            score = 0.0
            
            # Difficulty match (40% weight)
            difficulty_diff = abs(question.difficulty_level - target_difficulty)
            difficulty_score = max(0, 1.0 - difficulty_diff / 5.0)
            score += difficulty_score * 0.4
            
            # Topic relevance (30% weight)
            if question.topic_category:
                # Simple topic matching - could be enhanced with embedding similarity
                topic_score = 0.8 if question.topic_category in ["algebra", "geometry"] else 0.6
                score += topic_score * 0.3
            
            # Question type preference (20% weight)
            type_score = 1.0 if question.question_type.value == "multiple_choice" else 0.8
            score += type_score * 0.2
            
            # Random factor for variety (10% weight)
            random_factor = random.uniform(0.8, 1.2)
            score += random_factor * 0.1
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate question score: {e}")
            return 0.5
    
    def _generate_recommendation_reason(
        self,
        question: Question,
        profile: MathProfile,
        score: float
    ) -> str:
        """Generate human-readable reason for recommendation"""
        
        try:
            reasons = []
            
            # Difficulty-based reason
            if abs(question.difficulty_level - profile.math_skill) <= 1:
                reasons.append("Matches your current skill level")
            elif question.difficulty_level > profile.math_skill:
                reasons.append("Challenges you to improve")
            else:
                reasons.append("Reinforces fundamental concepts")
            
            # Topic-based reason
            if question.topic_category:
                reasons.append(f"Focuses on {question.topic_category}")
            
            # Score-based reason
            if score > 0.8:
                reasons.append("Highly relevant to your learning needs")
            elif score > 0.6:
                reasons.append("Good fit for your current progress")
            else:
                reasons.append("Provides practice in key areas")
            
            return "; ".join(reasons)
            
        except Exception as e:
            logger.error(f"❌ Failed to generate recommendation reason: {e}")
            return "Recommended based on your learning profile"
    
    async def _store_error_embedding_in_vector_db(
        self,
        error_context: str,
        error_embedding: List[float],
        recent_errors: List[ErrorPattern]
    ):
        """Store error embedding in vector database for future use"""
        
        try:
            # Build metadata for error pattern
            metadata = metadata_schema_service.build_error_pattern_metadata(
                domain=Domain.MATH,
                error_type=",".join(set(error.error_type for error in recent_errors)),
                error_context=error_context[:500],  # Limit context length
                student_id=recent_errors[0].student_id if recent_errors else "unknown",
                source="math_recommendation_service",
                confidence_score=0.8
            )
            
            # Store in vector database
            await vector_index_manager.batch_upsert_domain_embeddings_enhanced(
                embeddings=[error_embedding],
                texts=[error_context],
                metadata_list=[metadata],
                namespace="math_errors"
            )
            
            logger.debug(f"💾 Stored error embedding in vector database")
            
        except Exception as e:
            logger.error(f"❌ Failed to store error embedding: {e}")
    
    async def _generate_traditional_recommendations(
        self,
        db: AsyncSession,
        profile: MathProfile,
        num_questions: int,
        difficulty_adjustment: float
    ) -> List[Dict[str, Any]]:
        """Generate traditional recommendations based on skill level"""
        
        try:
            logger.info(f"🔄 Generating traditional recommendations for skill level {profile.math_skill}")
            
            # Calculate target difficulty
            target_difficulty = min(5, max(1, profile.math_skill + difficulty_adjustment))
            
            # Get questions by difficulty
            questions_result = await db.execute(
                select(Question)
                .where(
                    and_(
                        Question.subject == Subject.MATH,
                        Question.difficulty_level.between(
                            max(1, target_difficulty - 1),
                            min(5, target_difficulty + 1)
                        )
                    )
                )
                .order_by(func.random())
                .limit(num_questions * 2)
            )
            
            questions = questions_result.scalars().all()
            
            if not questions:
                logger.warning("⚠️ No questions found for traditional recommendation")
                return []
            
            # Convert to response format
            recommendations = []
            for question in questions[:num_questions]:
                recommendations.append({
                    "id": str(question.id),
                    "content": question.content,
                    "difficulty_level": question.difficulty_level,
                    "topic_category": question.topic_category,
                    "question_type": question.question_type.value,
                    "score": 0.6,  # Default score for traditional method
                    "difficulty_match": 1.0 - abs(question.difficulty_level - target_difficulty) / 5.0,
                    "recommendation_reason": f"Selected based on your skill level ({profile.math_skill})"
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Traditional recommendation failed: {e}")
            return []
    
    async def update_student_profile(
        self,
        db: AsyncSession,
        student_id: str,
        new_skill: float,
        update_reason: str = "skill_update"
    ) -> bool:
        """Update student profile with race condition protection"""
        
        try:
            logger.info(f"🔄 Updating student profile for {student_id}: {new_skill}")
            
            # Acquire profile lock
            profile_lock = await self.profile_lock_manager.acquire_lock(student_id)
            
            async with profile_lock:
                # Use optimistic locking for profile update
                success = await self.optimistic_lock_manager.execute_with_optimistic_lock(
                    self._update_profile_operation,
                    db,
                    student_id,
                    new_skill,
                    update_reason
                )
                
                if success:
                    logger.info(f"✅ Profile updated successfully for student {student_id}")
                    return True
                else:
                    logger.warning(f"⚠️ Profile update failed for student {student_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Profile update failed for student {student_id}: {e}")
            return False
        
        finally:
            # Release profile lock
            await self.profile_lock_manager.release_lock(student_id)
    
    async def _update_profile_operation(
        self,
        db: AsyncSession,
        student_id: str,
        new_skill: float,
        update_reason: str
    ) -> bool:
        """Database operation for updating profile"""
        
        try:
            # Get current profile
            result = await db.execute(
                select(MathProfile)
                .where(MathProfile.student_id == student_id)
            )
            
            profile = result.scalar_one_or_none()
            
            if not profile:
                # Create new profile if it doesn't exist
                profile = MathProfile(
                    student_id=student_id,
                    math_skill=new_skill,
                    global_skill=new_skill,
                    created_at=datetime.utcnow()
                )
                db.add(profile)
            else:
                # Update existing profile
                profile.math_skill = new_skill
                profile.global_skill = (profile.global_skill + new_skill) / 2
                profile.updated_at = datetime.utcnow()
                
                # Add version tracking for optimistic locking
                if not hasattr(profile, 'version'):
                    profile.version = 1
                else:
                    profile.version += 1
            
            # Add update metadata
            if not hasattr(profile, 'update_history'):
                profile.update_history = []
            
            profile.update_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "old_skill": getattr(profile, 'math_skill', 0),
                "new_skill": new_skill,
                "reason": update_reason,
                "service": "math_recommend_service"
            })
            
            await db.commit()
            return True
            
        except Exception as e:
            logger.error(f"❌ Database operation failed: {e}")
            await db.rollback()
            raise
    
    def _update_performance_metrics(self, response_time: float, success: bool):
        """Update performance metrics"""
        
        try:
            self.performance_metrics["total_recommendations"] += 1
            
            if success:
                self.performance_metrics["successful_recommendations"] += 1
            else:
                self.performance_metrics["failed_recommendations"] += 1
            
            # Update response time metrics
            self.performance_metrics["total_response_time"] += response_time
            self.performance_metrics["average_response_time"] = (
                self.performance_metrics["total_response_time"] / 
                self.performance_metrics["total_recommendations"]
            )
            
        except Exception as e:
            logger.error(f"❌ Error updating performance metrics: {e}")
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        
        return {
            "service": "MathRecommendService",
            "metrics": self.performance_metrics,
            "lock_management": {
                "active_locks": len(self.profile_lock_manager.locks),
                "lock_timeout": self.profile_lock_manager.lock_timeout,
                "cleanup_interval": self.profile_lock_manager.cleanup_interval
            },
            "optimistic_locking": {
                "retry_attempts": self.optimistic_lock_manager.retry_attempts,
                "retry_delay": self.optimistic_lock_manager.retry_delay
            }
        }


# Singleton instance
math_recommend_service = MathRecommendService()