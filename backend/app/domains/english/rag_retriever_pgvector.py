from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, text
from sqlalchemy.orm import joinedload
import numpy as np
from datetime import datetime, timedelta
import logging

from app.models.question import Question, Subject, DifficultyLevel
from app.models.student_attempt import StudentAttempt
from app.models.user import User
from app.core.cache import cache_service
from app.core.database import database

logger = logging.getLogger(__name__)


class RAGRetrieverPGVector:
    """RAG (Retrieval-Augmented Generation) retriever using PostgreSQL with pgvector"""
    
    def __init__(self):
        self.embedding_dimension = 1536  # OpenAI embedding dimension
        self.similarity_threshold = 0.7
        self.max_results = 20
        self.cache_ttl = 1800  # 30 minutes
        
    async def retrieve_similar_questions(
        self,
        db: AsyncSession,
        query_text: str,
        user: User,
        topic: Optional[str] = None,
        difficulty: Optional[DifficultyLevel] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve similar questions using vector similarity search"""
        
        cache_key = f"rag_retrieve:{hash(query_text)}:{user.id}:{topic}:{difficulty}:{limit}"
        
        # Try to get from cache first
        cached_result = await cache_service.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # Generate embedding for query text
            query_embedding = await self._generate_embedding(query_text)
            
            if not query_embedding:
                logger.warning("Failed to generate embedding for query")
                return await self._fallback_retrieval(db, user, topic, difficulty, limit)
            
            # Perform vector similarity search
            similar_questions = await self._vector_similarity_search(
                db, query_embedding, user, topic, difficulty, limit
            )
            
            # Cache results
            await cache_service.set(cache_key, similar_questions, self.cache_ttl)
            
            return similar_questions
            
        except Exception as e:
            logger.error(f"Error in RAG retrieval: {e}")
            return await self._fallback_retrieval(db, user, topic, difficulty, limit)
    
    async def retrieve_context_for_generation(
        self,
        db: AsyncSession,
        topic: str,
        difficulty_level: int,
        question_type: str,
        user_error_patterns: List[str],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve context for question generation"""
        
        cache_key = f"rag_context:{topic}:{difficulty_level}:{question_type}:{hash(str(user_error_patterns))}"
        
        # Try to get from cache first
        cached_result = await cache_service.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # Build context query
            context_query = f"{topic} {question_type} difficulty {difficulty_level}"
            if user_error_patterns:
                context_query += f" {' '.join(user_error_patterns)}"
            
            # Generate embedding for context query
            context_embedding = await self._generate_embedding(context_query)
            
            if not context_embedding:
                return await self._fallback_context_retrieval(
                    db, topic, difficulty_level, question_type, limit
                )
            
            # Search for similar questions
            similar_questions = await self._vector_similarity_search(
                db, context_embedding, None, topic, difficulty_level, limit
            )
            
            # Cache results
            await cache_service.set(cache_key, similar_questions, self.cache_ttl)
            
            return similar_questions
            
        except Exception as e:
            logger.error(f"Error in context retrieval: {e}")
            return await self._fallback_context_retrieval(
                db, topic, difficulty_level, question_type, limit
            )
    
    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using embedding service"""
        
        try:
            from app.services.embedding_service import embedding_service
            return await embedding_service.get_embedding(text)
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    async def _vector_similarity_search(
        self,
        db: AsyncSession,
        query_embedding: List[float],
        user: Optional[User],
        topic: Optional[str],
        difficulty: Optional[DifficultyLevel],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search using pgvector"""
        
        try:
            # Convert embedding to PostgreSQL array format
            embedding_array = f"[{','.join(map(str, query_embedding))}]"
            
            # Build the similarity search query
            similarity_query = f"""
                SELECT 
                    q.id,
                    q.content,
                    q.topic,
                    q.difficulty_level,
                    q.question_type,
                    q.options,
                    q.correct_answer,
                    q.explanation,
                    q.tags,
                    q.embedding <-> '{embedding_array}'::vector as distance
                FROM questions q
                WHERE q.subject = 'english' 
                AND q.is_active = true
            """
            
            # Add filters
            if topic:
                similarity_query += f" AND q.topic ILIKE '%{topic}%'"
            
            if difficulty:
                similarity_query += f" AND q.difficulty_level = {difficulty.value}"
            elif user:
                similarity_query += f" AND q.difficulty_level = {user.current_english_level}"
            
            # Add similarity ordering and limit
            similarity_query += f"""
                ORDER BY distance ASC
                LIMIT {limit}
            """
            
            # Execute query
            result = await database.fetch_all(similarity_query)
            
            # Convert to response format
            similar_questions = []
            for row in result:
                similar_questions.append({
                    "id": str(row["id"]),
                    "content": row["content"],
                    "topic": row["topic"],
                    "difficulty_level": row["difficulty_level"],
                    "question_type": row["question_type"],
                    "options": row["options"],
                    "correct_answer": row["correct_answer"],
                    "explanation": row["explanation"],
                    "tags": row["tags"],
                    "similarity_score": 1.0 - float(row["distance"]),
                    "distance": float(row["distance"])
                })
            
            return similar_questions
            
        except Exception as e:
            logger.error(f"Error in vector similarity search: {e}")
            return []
    
    async def _fallback_retrieval(
        self,
        db: AsyncSession,
        user: User,
        topic: Optional[str],
        difficulty: Optional[DifficultyLevel],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Fallback retrieval when vector search fails"""
        
        try:
            # Build base query
            query = select(Question).where(
                and_(
                    Question.subject == Subject.ENGLISH,
                    Question.is_active == True
                )
            )
            
            # Add filters
            if topic:
                query = query.where(Question.topic.ilike(f"%{topic}%"))
            
            if difficulty:
                query = query.where(Question.difficulty_level == difficulty)
            else:
                query = query.where(Question.difficulty_level == user.current_english_level)
            
            # Order by creation date and limit
            query = query.order_by(desc(Question.created_at)).limit(limit)
            
            result = await db.execute(query)
            questions = result.scalars().all()
            
            # Convert to response format
            fallback_questions = []
            for question in questions:
                fallback_questions.append({
                    "id": str(question.id),
                    "content": question.content,
                    "topic": question.topic,
                    "difficulty_level": question.difficulty_level,
                    "question_type": question.question_type,
                    "options": question.options,
                    "correct_answer": question.correct_answer,
                    "explanation": question.explanation,
                    "tags": question.tags,
                    "similarity_score": 0.5,  # Default fallback score
                    "distance": 1.0  # Default fallback distance
                })
            
            return fallback_questions
            
        except Exception as e:
            logger.error(f"Error in fallback retrieval: {e}")
            return []
    
    async def _fallback_context_retrieval(
        self,
        db: AsyncSession,
        topic: str,
        difficulty_level: int,
        question_type: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Fallback context retrieval when vector search fails"""
        
        try:
            # Build base query
            query = select(Question).where(
                and_(
                    Question.subject == Subject.ENGLISH,
                    Question.is_active == True,
                    Question.topic.ilike(f"%{topic}%"),
                    Question.difficulty_level == difficulty_level
                )
            )
            
            # Order by usage count and limit
            query = query.order_by(desc(Question.usage_count)).limit(limit)
            
            result = await db.execute(query)
            questions = result.scalars().all()
            
            # Convert to response format
            context_questions = []
            for question in questions:
                context_questions.append({
                    "id": str(question.id),
                    "content": question.content,
                    "topic": question.topic,
                    "difficulty_level": question.difficulty_level,
                    "question_type": question.question_type,
                    "options": question.options,
                    "correct_answer": question.correct_answer,
                    "explanation": question.explanation,
                    "tags": question.tags,
                    "similarity_score": 0.5,
                    "distance": 1.0
                })
            
            return context_questions
            
        except Exception as e:
            logger.error(f"Error in fallback context retrieval: {e}")
            return []
    
    async def store_question_embedding(
        self,
        question_id: str,
        question_content: str,
        topic: str
    ) -> bool:
        """Store question embedding in the database"""
        
        try:
            # Generate embedding
            embedding = await self._generate_embedding(question_content)
            
            if not embedding:
                logger.error(f"Failed to generate embedding for question {question_id}")
                return False
            
            # Convert to PostgreSQL array format
            embedding_array = f"[{','.join(map(str, embedding))}]"
            
            # Update question with embedding
            update_query = f"""
                UPDATE questions 
                SET embedding = '{embedding_array}'::vector
                WHERE id = '{question_id}'
            """
            
            await database.execute(update_query)
            
            logger.info(f"Stored embedding for question {question_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing question embedding: {e}")
            return False
    
    async def get_question_clusters(
        self,
        db: AsyncSession,
        topic: Optional[str] = None,
        difficulty: Optional[DifficultyLevel] = None,
        min_cluster_size: int = 3
    ) -> List[Dict[str, Any]]:
        """Get question clusters based on similarity"""
        
        try:
            # This would typically use clustering algorithms
            # For now, we'll group by topic and difficulty
            
            query = select(Question).where(
                and_(
                    Question.subject == Subject.ENGLISH,
                    Question.is_active == True
                )
            )
            
            if topic:
                query = query.where(Question.topic.ilike(f"%{topic}%"))
            
            if difficulty:
                query = query.where(Question.difficulty_level == difficulty)
            
            result = await db.execute(query)
            questions = result.scalars().all()
            
            # Group questions by topic and difficulty
            clusters = {}
            for question in questions:
                cluster_key = f"{question.topic}_{question.difficulty_level}"
                
                if cluster_key not in clusters:
                    clusters[cluster_key] = {
                        "cluster_id": cluster_key,
                        "topic": question.topic,
                        "difficulty_level": question.difficulty_level,
                        "question_count": 0,
                        "questions": []
                    }
                
                clusters[cluster_key]["question_count"] += 1
                clusters[cluster_key]["questions"].append({
                    "id": str(question.id),
                    "content": question.content,
                    "question_type": question.question_type
                })
            
            # Filter clusters by minimum size
            filtered_clusters = [
                cluster for cluster in clusters.values()
                if cluster["question_count"] >= min_cluster_size
            ]
            
            return filtered_clusters
            
        except Exception as e:
            logger.error(f"Error getting question clusters: {e}")
            return []


# Global instance
rag_retriever = RAGRetrieverPGVector()