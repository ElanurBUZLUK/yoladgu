from typing import List, Dict, Any, Optional, Tuple
import logging
import asyncio
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, text
from sqlalchemy.orm import joinedload

from app.models.question import Question, Subject, DifficultyLevel
from app.models.student_attempt import StudentAttempt
from app.models.user import User
from app.database import database_manager
from app.core.cache import cache_service
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever combining vector search and traditional database queries"""
    
    def __init__(self):
        self.semantic_weight = 0.6
        self.keyword_weight = 0.4
        self.max_results = 50
        self.cache_ttl = 3600  # 1 hour
        self.similarity_threshold = 0.7
        
    async def retrieve(
        self,
        query: str,
        user: User,
        subject: Subject = Subject.ENGLISH,
        topic: Optional[str] = None,
        difficulty: Optional[DifficultyLevel] = None,
        limit: int = 10,
        exclude_attempted: bool = True,
        db: Optional[AsyncSession] = None
    ) -> List[Dict[str, Any]]:
        """Main retrieval method combining vector and keyword search"""
        
        cache_key = f"retrieve:{hash(query)}:{user.id}:{subject}:{topic}:{difficulty}:{limit}:{exclude_attempted}"
        
        # Try to get from cache first
        cached_result = await cache_service.get(cache_key)
        if cached_result:
            logger.debug(f"Cache hit for retrieval: {query[:50]}...")
            return cached_result
        
        try:
            # Generate query embedding
            query_embeddings = await embedding_service.embed_texts([query])
            query_embedding = query_embeddings[0] if query_embeddings else []
            
            # Perform vector search
            vector_results = await self._vector_search(
                query_embedding, user, subject, topic, difficulty, limit * 2, db
            )
            
            # Perform keyword search
            keyword_results = await self._keyword_search(
                query, user, subject, topic, difficulty, limit * 2, db
            )
            
            # Combine and rank results
            combined_results = await self._combine_results(
                vector_results, keyword_results, limit
            )
            
            # Filter out attempted questions if requested
            if exclude_attempted:
                combined_results = await self._filter_attempted_questions(
                    user, combined_results, db
                )
            
            # Cache results
            await cache_service.set(cache_key, combined_results, self.cache_ttl)
            
            logger.info(f"Retrieved {len(combined_results)} questions for query: {query[:50]}...")
            return combined_results
            
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            # Fallback to simple keyword search
            return await self._fallback_search(query, user, subject, topic, difficulty, limit, db)
    
    async def _vector_search(
        self,
        query_embedding: List[float],
        user: User,
        subject: Subject,
        topic: Optional[str],
        difficulty: Optional[DifficultyLevel],
        limit: int,
        db: Optional[AsyncSession]
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
                    q.content_embedding <-> '{embedding_array}'::vector as distance,
                    1 - (q.content_embedding <=> '{embedding_array}'::vector) as similarity
                FROM questions q
                WHERE q.subject = '{subject.value}'
                AND q.is_active = true
                AND q.content_embedding IS NOT NULL
            """
            
            # Add filters
            if topic:
                similarity_query += f" AND q.topic ILIKE '%{topic}%'"
            
            if difficulty:
                similarity_query += f" AND q.difficulty_level = {difficulty.value}"
            else:
                # Use user's current level
                user_level = getattr(user, f'current_{subject.value}_level', 3)
                similarity_query += f" AND q.difficulty_level = {user_level}"
            
            # Add similarity ordering and limit
            similarity_query += f"""
                ORDER BY distance ASC
                LIMIT {limit}
            """
            
            # Execute query
            result = await database.fetch_all(similarity_query)
            
            # Convert to response format
            vector_results = []
            for row in result:
                vector_results.append({
                    "id": str(row["id"]),
                    "content": row["content"],
                    "topic": row["topic"],
                    "difficulty_level": row["difficulty_level"],
                    "question_type": row["question_type"],
                    "options": row["options"],
                    "correct_answer": row["correct_answer"],
                    "explanation": row["explanation"],
                    "tags": row["tags"],
                    "similarity_score": float(row["similarity"]),
                    "distance": float(row["distance"]),
                    "search_method": "vector"
                })
            
            logger.debug(f"Vector search returned {len(vector_results)} results")
            return vector_results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    async def _keyword_search(
        self,
        query: str,
        user: User,
        subject: Subject,
        topic: Optional[str],
        difficulty: Optional[DifficultyLevel],
        limit: int,
        db: Optional[AsyncSession]
    ) -> List[Dict[str, Any]]:
        """Perform keyword-based search using traditional SQL"""
        
        try:
            # Build base query
            base_query = f"""
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
                    q.usage_count,
                    q.created_at
                FROM questions q
                WHERE q.subject = '{subject.value}'
                AND q.is_active = true
            """
            
            # Add topic filter
            if topic:
                base_query += f" AND q.topic ILIKE '%{topic}%'"
            
            # Add difficulty filter
            if difficulty:
                base_query += f" AND q.difficulty_level = {difficulty.value}"
            else:
                user_level = getattr(user, f'current_{subject.value}_level', 3)
                base_query += f" AND q.difficulty_level = {user_level}"
            
            # Add keyword search
            if query:
                keywords = query.split()
                keyword_conditions = []
                for keyword in keywords:
                    if len(keyword) > 2:  # Only search for meaningful keywords
                        keyword_conditions.append(f"q.content ILIKE '%{keyword}%'")
                        keyword_conditions.append(f"q.topic ILIKE '%{keyword}%'")
                
                if keyword_conditions:
                    base_query += f" AND ({' OR '.join(keyword_conditions)})"
            
            # Add ordering by relevance (usage count and recency)
            base_query += f"""
                ORDER BY q.usage_count DESC, q.created_at DESC
                LIMIT {limit}
            """
            
            # Execute query
            result = await database.fetch_all(base_query)
            
            # Convert to response format
            keyword_results = []
            for i, row in enumerate(result):
                # Calculate keyword relevance score based on position
                keyword_score = 1.0 - (i * 0.05)  # Decreasing score based on position
                
                keyword_results.append({
                    "id": str(row["id"]),
                    "content": row["content"],
                    "topic": row["topic"],
                    "difficulty_level": row["difficulty_level"],
                    "question_type": row["question_type"],
                    "options": row["options"],
                    "correct_answer": row["correct_answer"],
                    "explanation": row["explanation"],
                    "tags": row["tags"],
                    "similarity_score": keyword_score,
                    "usage_count": row["usage_count"],
                    "search_method": "keyword"
                })
            
            logger.debug(f"Keyword search returned {len(keyword_results)} results")
            return keyword_results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    async def _combine_results(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Combine and rank results from both search methods"""
        
        # Create a combined score for each question
        question_scores = {}
        
        # Process vector results
        for result in vector_results:
            question_id = result["id"]
            if question_id not in question_scores:
                question_scores[question_id] = {
                    "question": result,
                    "vector_score": result["similarity_score"],
                    "keyword_score": 0.0,
                    "combined_score": 0.0
                }
            else:
                question_scores[question_id]["vector_score"] = result["similarity_score"]
        
        # Process keyword results
        for result in keyword_results:
            question_id = result["id"]
            if question_id not in question_scores:
                question_scores[question_id] = {
                    "question": result,
                    "vector_score": 0.0,
                    "keyword_score": result["similarity_score"],
                    "combined_score": 0.0
                }
            else:
                question_scores[question_id]["keyword_score"] = result["similarity_score"]
        
        # Calculate combined scores
        for question_data in question_scores.values():
            combined_score = (
                self.semantic_weight * question_data["vector_score"] +
                self.keyword_weight * question_data["keyword_score"]
            )
            question_data["combined_score"] = combined_score
        
        # Sort by combined score
        sorted_results = sorted(
            question_scores.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )
        
        # Return top results
        final_results = []
        for result in sorted_results[:limit]:
            question = result["question"].copy()
            question["combined_score"] = result["combined_score"]
            question["vector_score"] = result["vector_score"]
            question["keyword_score"] = result["keyword_score"]
            final_results.append(question)
        
        return final_results
    
    async def _filter_attempted_questions(
        self,
        user: User,
        questions: List[Dict[str, Any]],
        db: Optional[AsyncSession]
    ) -> List[Dict[str, Any]]:
        """Filter out questions that the user has already attempted"""
        
        try:
            # Get user's attempted question IDs
            attempted_query = f"""
                SELECT DISTINCT question_id 
                FROM student_attempts 
                WHERE student_id = '{user.id}'
            """
            
            result = await database.fetch_all(attempted_query)
            attempted_ids = {str(row["question_id"]) for row in result}
            
            # Filter out attempted questions
            filtered_questions = [
                q for q in questions if q["id"] not in attempted_ids
            ]
            
            logger.debug(f"Filtered {len(questions) - len(filtered_questions)} attempted questions")
            return filtered_questions
            
        except Exception as e:
            logger.error(f"Error filtering attempted questions: {e}")
            return questions  # Return original list if filtering fails
    
    async def _fallback_search(
        self,
        query: str,
        user: User,
        subject: Subject,
        topic: Optional[str],
        difficulty: Optional[DifficultyLevel],
        limit: int,
        db: Optional[AsyncSession]
    ) -> List[Dict[str, Any]]:
        """Fallback search when main retrieval fails"""
        
        try:
            # Simple keyword search as fallback
            return await self._keyword_search(
                query, user, subject, topic, difficulty, limit, db
            )
            
        except Exception as e:
            logger.error(f"Fallback search also failed: {e}")
            return []
    
    async def retrieve_grammar_rules(self, pattern: str) -> List[str]:
        """Retrieve grammar rules related to a given error pattern."""
        try:
            # Use the main retrieve method to find relevant "grammar rule" questions
            # Assuming grammar rules are stored as questions with specific tags or topics
            query = f"grammar rules for {pattern}"
            # We might need a dedicated subject or tag for grammar rules if they are not questions
            # For now, let's search within English subject and look for "grammar" in topic/tags
            results = await self.retrieve(
                query=query,
                user=User(id="system", username="system"), # Use a system user for knowledge retrieval
                subject=Subject.ENGLISH,
                topic="grammar", # Assuming a "grammar" topic exists for rules
                limit=5,
                exclude_attempted=False # Don't exclude, as these are knowledge, not practice questions
            )
            
            # Extract explanations or content as grammar rules
            grammar_rules = [r.get("explanation", r.get("content", "")) for r in results if r.get("explanation") or r.get("content")]
            return grammar_rules
        except Exception as e:
            logger.error(f"Error retrieving grammar rules for pattern {pattern}: {e}")
            return []

    async def retrieve_vocabulary_context(self, error_patterns: List[str]) -> Optional[str]:
        """Retrieve vocabulary context related to given error patterns."""
        try:
            # Combine error patterns into a single query for vocabulary
            query = f"vocabulary related to {', '.join(error_patterns)}"
            results = await self.retrieve(
                query=query,
                user=User(id="system", username="system"),
                subject=Subject.ENGLISH,
                topic="vocabulary", # Assuming a "vocabulary" topic exists
                limit=3,
                exclude_attempted=False
            )
            
            # Concatenate relevant content/explanations
            vocabulary_context = " ".join([r.get("explanation", r.get("content", "")) for r in results if r.get("explanation") or r.get("content")])
            return vocabulary_context if vocabulary_context else None
        except Exception as e:
            logger.error(f"Error retrieving vocabulary context for patterns {error_patterns}: {e}")
            return None

    async def retrieve_topic_context(self, error_patterns: List[str]) -> Optional[str]:
        """Retrieve broader topic context related to given error patterns."""
        try:
            query = f"topics related to {', '.join(error_patterns)}"
            results = await self.retrieve(
                query=query,
                user=User(id="system", username="system"),
                subject=Subject.ENGLISH,
                limit=2, # Broader topics, fewer results needed
                exclude_attempted=False
            )
            topic_context = " ".join([r.get("topic", r.get("content", "")) for r in results if r.get("topic") or r.get("content")])
            return topic_context if topic_context else None
        except Exception as e:
            logger.error(f"Error retrieving topic context for patterns {error_patterns}: {e}")
            return None

    async def get_similar_questions(
        self,
        question_id: str,
        user: User,
        subject: Subject = Subject.ENGLISH,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get similar questions to a given question"""
        
        try:
            # Get the question content
            question_query = f"""
                SELECT content, topic, difficulty_level
                FROM questions
                WHERE id = '{question_id}'
            """
            
            result = await database.fetch_one(question_query)
            if not result:
                return []
            
            question_content = result["content"]
            
            # Use the content to find similar questions
            return await self.retrieve(
                query=question_content,
                user=user,
                subject=subject,
                limit=limit,
                exclude_attempted=True
            )
            
        except Exception as e:
            logger.error(f"Error getting similar questions: {e}")
            return []
    
    async def get_question_statistics(self) -> Dict[str, Any]:
        """Get statistics about the retriever"""
        
        try:
            # Get total questions count
            total_questions = await database.fetch_one("""
                SELECT COUNT(*) as count FROM questions WHERE is_active = true
            """)
            
            # Get questions with embeddings
            questions_with_embeddings = await database.fetch_one("""
                SELECT COUNT(*) as count FROM questions 
                WHERE is_active = true AND content_embedding IS NOT NULL
            """)
            
            return {
                "total_questions": total_questions["count"] if total_questions else 0,
                "questions_with_embeddings": questions_with_embeddings["count"] if questions_with_embeddings else 0,
                "semantic_weight": self.semantic_weight,
                "keyword_weight": self.keyword_weight,
                "similarity_threshold": self.similarity_threshold,
                "cache_ttl": self.cache_ttl
            }
            
        except Exception as e:
            logger.error(f"Error getting retriever statistics: {e}")
            return {}


# Global instance
hybrid_retriever = HybridRetriever()
