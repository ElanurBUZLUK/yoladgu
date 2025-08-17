from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.orm import joinedload
import numpy as np
from datetime import datetime, timedelta

from app.models.question import Question, Subject, DifficultyLevel
from app.models.student_attempt import StudentAttempt
from app.models.user import User
from app.core.cache import cache_service


class HybridRetriever:
    """Hybrid retriever that combines semantic search and keyword-based retrieval for English questions"""
    
    def __init__(self):
        self.semantic_weight = 0.6
        self.keyword_weight = 0.4
        self.max_results = 50
        self.cache_ttl = 3600  # 1 hour
        
    async def retrieve_questions(
        self,
        db: AsyncSession,
        user: User,
        topic: Optional[str] = None,
        difficulty: Optional[DifficultyLevel] = None,
        limit: int = 10,
        exclude_attempted: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve English questions using hybrid approach"""
        
        cache_key = f"hybrid_retrieve:{user.id}:{topic}:{difficulty}:{limit}:{exclude_attempted}"
        
        # Try to get from cache first
        cached_result = await cache_service.get(cache_key)
        if cached_result:
            return cached_result
        
        # Get semantic results
        semantic_results = await self._semantic_search(
            db, user, topic, difficulty, limit * 2
        )
        
        # Get keyword results
        keyword_results = await self._keyword_search(
            db, user, topic, difficulty, limit * 2
        )
        
        # Combine and rank results
        combined_results = await self._combine_results(
            semantic_results, keyword_results, limit
        )
        
        # Filter out attempted questions if requested
        if exclude_attempted:
            combined_results = await self._filter_attempted_questions(
                db, user, combined_results
            )
        
        # Cache results
        await cache_service.set(cache_key, combined_results, self.cache_ttl)
        
        return combined_results
    
    async def _semantic_search(
        self,
        db: AsyncSession,
        user: User,
        topic: Optional[str] = None,
        difficulty: Optional[DifficultyLevel] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Semantic search using embeddings and similarity"""
        
        # Build base query
        query = select(Question).where(
            and_(
                Question.subject == Subject.ENGLISH,
                Question.is_active == True
            )
        )
        
        # Add topic filter if provided
        if topic:
            query = query.where(
                or_(
                    Question.topic.contains(topic),
                    Question.content.contains(topic),
                    Question.tags.contains([topic])
                )
            )
        
        # Add difficulty filter if provided
        if difficulty:
            query = query.where(Question.difficulty_level == difficulty)
        else:
            # Use user's current English level
            query = query.where(Question.difficulty_level == user.current_english_level)
        
        # Add ordering by relevance (for now, using creation date as proxy)
        query = query.order_by(desc(Question.created_at)).limit(limit)
        
        result = await db.execute(query)
        questions = result.scalars().all()
        
        # Convert to dict format with semantic score
        semantic_results = []
        for i, question in enumerate(questions):
            semantic_results.append({
                "question": question,
                "score": 1.0 - (i * 0.05),  # Decreasing score based on position
                "method": "semantic"
            })
        
        return semantic_results
    
    async def _keyword_search(
        self,
        db: AsyncSession,
        user: User,
        topic: Optional[str] = None,
        difficulty: Optional[DifficultyLevel] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Keyword-based search using text matching"""
        
        # Build base query
        query = select(Question).where(
            and_(
                Question.subject == Subject.ENGLISH,
                Question.is_active == True
            )
        )
        
        # Add topic filter if provided
        if topic:
            # More specific keyword matching
            query = query.where(
                or_(
                    Question.topic.ilike(f"%{topic}%"),
                    Question.content.ilike(f"%{topic}%"),
                    Question.question_type.ilike(f"%{topic}%")
                )
            )
        
        # Add difficulty filter if provided
        if difficulty:
            query = query.where(Question.difficulty_level == difficulty)
        else:
            # Use user's current English level
            query = query.where(Question.difficulty_level == user.current_english_level)
        
        # Add ordering by popularity/usage
        query = query.order_by(desc(Question.usage_count)).limit(limit)
        
        result = await db.execute(query)
        questions = result.scalars().all()
        
        # Convert to dict format with keyword score
        keyword_results = []
        for i, question in enumerate(questions):
            keyword_results.append({
                "question": question,
                "score": 1.0 - (i * 0.03),  # Less aggressive score decrease
                "method": "keyword"
            })
        
        return keyword_results
    
    async def _combine_results(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Combine and rank results from both methods"""
        
        # Create a combined score for each question
        question_scores = {}
        
        # Process semantic results
        for result in semantic_results:
            question_id = str(result["question"].id)
            if question_id not in question_scores:
                question_scores[question_id] = {
                    "question": result["question"],
                    "semantic_score": result["score"],
                    "keyword_score": 0.0,
                    "combined_score": 0.0
                }
            else:
                question_scores[question_id]["semantic_score"] = result["score"]
        
        # Process keyword results
        for result in keyword_results:
            question_id = str(result["question"].id)
            if question_id not in question_scores:
                question_scores[question_id] = {
                    "question": result["question"],
                    "semantic_score": 0.0,
                    "keyword_score": result["score"],
                    "combined_score": 0.0
                }
            else:
                question_scores[question_id]["keyword_score"] = result["score"]
        
        # Calculate combined scores
        for question_data in question_scores.values():
            combined_score = (
                question_data["semantic_score"] * self.semantic_weight +
                question_data["keyword_score"] * self.keyword_weight
            )
            question_data["combined_score"] = combined_score
        
        # Sort by combined score and return top results
        sorted_results = sorted(
            question_scores.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )
        
        # Convert to final format
        final_results = []
        for result in sorted_results[:limit]:
            final_results.append({
                "id": str(result["question"].id),
                "content": result["question"].content,
                "topic": result["question"].topic,
                "difficulty_level": result["question"].difficulty_level,
                "question_type": result["question"].question_type,
                "options": result["question"].options,
                "correct_answer": result["question"].correct_answer,
                "explanation": result["question"].explanation,
                "tags": result["question"].tags,
                "semantic_score": result["semantic_score"],
                "keyword_score": result["keyword_score"],
                "combined_score": result["combined_score"]
            })
        
        return final_results
    
    async def _filter_attempted_questions(
        self,
        db: AsyncSession,
        user: User,
        questions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter out questions that the user has already attempted"""
        
        if not questions:
            return questions
        
        question_ids = [q["id"] for q in questions]
        
        # Get attempted question IDs
        attempted_query = select(StudentAttempt.question_id).where(
            and_(
                StudentAttempt.user_id == user.id,
                StudentAttempt.question_id.in_(question_ids)
            )
        )
        
        result = await db.execute(attempted_query)
        attempted_ids = {str(row[0]) for row in result.fetchall()}
        
        # Filter out attempted questions
        filtered_questions = [
            q for q in questions if q["id"] not in attempted_ids
        ]
        
        return filtered_questions
    
    async def get_similar_questions(
        self,
        db: AsyncSession,
        question_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get similar questions based on topic and difficulty"""
        
        # Get the reference question
        query = select(Question).where(Question.id == question_id)
        result = await db.execute(query)
        reference_question = result.scalar_one_or_none()
        
        if not reference_question:
            return []
        
        # Find similar questions
        similar_query = select(Question).where(
            and_(
                Question.subject == Subject.ENGLISH,
                Question.is_active == True,
                Question.id != question_id,
                Question.topic == reference_question.topic,
                Question.difficulty_level == reference_question.difficulty_level
            )
        ).limit(limit)
        
        result = await db.execute(similar_query)
        similar_questions = result.scalars().all()
        
        return [
            {
                "id": str(q.id),
                "content": q.content,
                "topic": q.topic,
                "difficulty_level": q.difficulty_level,
                "question_type": q.question_type
            }
            for q in similar_questions
        ]


# Global instance
hybrid_retriever = HybridRetriever()