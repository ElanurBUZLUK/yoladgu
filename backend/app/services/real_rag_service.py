"""
Real RAG Service - pgvector based semantic search and retrieval
"""
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import logging
from datetime import datetime
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.question import VECTOR

from app.models.question import Question
from app.services.embedding_service import embedding_service
from app.core.config import settings

logger = logging.getLogger(__name__)


class RealRAGService:
    """Real RAG service using pgvector for semantic search"""
    
    def __init__(self):
        self.embedding_service = embedding_service
        self.similarity_threshold = 0.7
        self.max_results = 10
        self.context_window_size = 5
        
    async def search_similar_questions(
        self,
        query: str,
        subject: str,
        difficulty_range: Optional[Tuple[float, float]] = None,
        limit: int = 10,
        db: AsyncSession = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar questions using semantic similarity
        """
        try:
            # Generate embedding for query
            query_embedding = await self.embedding_service.get_embedding(
                text=query,
                domain=subject,
                content_type="question"
            )
            
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
            
            # Build search query with vector similarity
            search_query = select(Question).add_columns(
                func.cosine_similarity(Question.embedding, query_embedding).label('similarity')
            ).where(
                and_(
                    Question.subject == subject,
                    Question.embedding.isnot(None)
                )
            )
            
            # Add difficulty filter if provided
            if difficulty_range:
                min_diff, max_diff = difficulty_range
                search_query = search_query.where(
                    Question.difficulty_level.between(min_diff, max_diff)
                )
            
            # Order by similarity and limit results
            search_query = search_query.order_by(
                func.cosine_similarity(Question.embedding, query_embedding).desc()
            ).limit(limit)
            
            # Execute query
            result = await db.execute(search_query)
            rows = result.fetchall()
            
            # Process results
            similar_questions = []
            for row in rows:
                question, similarity = row
                if similarity >= self.similarity_threshold:
                    similar_questions.append({
                        "question": question,
                        "similarity_score": float(similarity),
                        "metadata": {
                            "subject": question.subject,
                            "difficulty": question.difficulty_level,
                            "topic": question.topic_category,
                            "type": question.question_type
                        }
                    })
            
            logger.info(f"Found {len(similar_questions)} similar questions for query: {query[:50]}...")
            return similar_questions
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    async def hybrid_search(
        self,
        query: str,
        subject: str,
        difficulty_range: Optional[Tuple[float, float]] = None,
        limit: int = 10,
        db: AsyncSession = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic and keyword search
        """
        try:
            # Semantic search
            semantic_results = await self.search_similar_questions(
                query=query,
                subject=subject,
                difficulty_range=difficulty_range,
                limit=limit // 2,
                db=db
            )
            
            # Keyword search using full-text search
            keyword_results = await self._keyword_search(
                query=query,
                subject=subject,
                difficulty_range=difficulty_range,
                limit=limit // 2,
                db=db
            )
            
            # Combine and rank results
            combined_results = self._combine_and_rank_results(
                semantic_results=semantic_results,
                keyword_results=keyword_results,
                query=query
            )
            
            return combined_results[:limit]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    async def _keyword_search(
        self,
        query: str,
        subject: str,
        difficulty_range: Optional[Tuple[float, float]] = None,
        limit: int = 5,
        db: AsyncSession = None
    ) -> List[Dict[str, Any]]:
        """
        Keyword-based search using PostgreSQL full-text search
        """
        try:
            # Build keyword search query
            search_query = select(Question).where(
                and_(
                    Question.subject == subject,
                    Question.search_vector.op('@@')(func.plainto_tsquery('english', query))
                )
            )
            
            # Add difficulty filter if provided
            if difficulty_range:
                min_diff, max_diff = difficulty_range
                search_query = search_query.where(
                    Question.difficulty_level.between(min_diff, max_diff)
                )
            
            # Order by relevance and limit
            search_query = search_query.order_by(
                func.ts_rank(Question.search_vector, func.plainto_tsquery('english', query)).desc()
            ).limit(limit)
            
            # Execute query
            result = await db.execute(search_query)
            questions = result.scalars().all()
            
            # Process results
            keyword_results = []
            for question in questions:
                keyword_results.append({
                    "question": question,
                    "similarity_score": 0.6,  # Default score for keyword matches
                    "search_type": "keyword",
                    "metadata": {
                        "subject": question.subject,
                        "difficulty": question.difficulty_level,
                        "topic": question.topic_category,
                        "type": question.question_type
                    }
                })
            
            return keyword_results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def _combine_and_rank_results(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Combine semantic and keyword results with intelligent ranking
        """
        try:
            # Create a map of question IDs to avoid duplicates
            seen_questions = {}
            
            # Process semantic results (higher weight)
            for result in semantic_results:
                question_id = str(result["question"].id)
                seen_questions[question_id] = {
                    **result,
                    "search_type": "semantic",
                    "combined_score": result["similarity_score"] * 0.7  # 70% weight for semantic
                }
            
            # Process keyword results
            for result in keyword_results:
                question_id = str(result["question"].id)
                if question_id in seen_questions:
                    # Question found in both searches - boost score
                    existing_score = seen_questions[question_id]["combined_score"]
                    keyword_score = result["similarity_score"] * 0.3  # 30% weight for keyword
                    seen_questions[question_id]["combined_score"] = existing_score + keyword_score
                    seen_questions[question_id]["search_type"] = "hybrid"
                else:
                    # New question from keyword search
                    seen_questions[question_id] = {
                        **result,
                        "combined_score": result["similarity_score"] * 0.3
                    }
            
            # Sort by combined score
            combined_results = list(seen_questions.values())
            combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error combining search results: {e}")
            return semantic_results + keyword_results
    
    async def generate_question_embeddings(
        self,
        question_ids: List[str],
        db: AsyncSession = None
    ) -> Dict[str, bool]:
        """
        Generate embeddings for questions that don't have them
        """
        try:
            # Find questions without embeddings
            query = select(Question).where(
                and_(
                    Question.id.in_(question_ids),
                    or_(
                        Question.embedding.is_(None),
                        Question.embedding_generated_at.is_(None)
                    )
                )
            )
            
            result = await db.execute(query)
            questions = result.scalars().all()
            
            success_count = 0
            for question in questions:
                try:
                    # Generate embedding
                    embedding = await self.embedding_service.get_embedding(
                        text=question.content,
                        domain=question.subject,
                        content_type="question"
                    )
                    
                    if embedding:
                        # Update question with embedding
                        question.embedding = embedding
                        question.embedding_model = "text-embedding-3-small"
                        question.embedding_generated_at = datetime.utcnow()
                        
                        # Update search vector for full-text search
                        question.search_vector = func.to_tsvector('english', question.content)
                        
                        success_count += 1
                        logger.debug(f"Generated embedding for question {question.id}")
                    else:
                        logger.warning(f"Failed to generate embedding for question {question.id}")
                        
                except Exception as e:
                    logger.error(f"Error generating embedding for question {question.id}: {e}")
            
            await db.commit()
            logger.info(f"Generated embeddings for {success_count}/{len(questions)} questions")
            
            return {"success": True, "processed": success_count, "total": len(questions)}
            
        except Exception as e:
            logger.error(f"Error in batch embedding generation: {e}")
            await db.rollback()
            return {"success": False, "error": str(e)}
    
    async def update_search_vectors(self, db: AsyncSession = None) -> Dict[str, Any]:
        """
        Update search vectors for all questions
        """
        try:
            # Update all questions without search vectors
            update_query = """
                UPDATE questions 
                SET search_vector = to_tsvector('english', content)
                WHERE search_vector IS NULL
            """
            
            result = await db.execute(update_query)
            await db.commit()
            
            logger.info(f"Updated search vectors for questions")
            return {"success": True, "updated": result.rowcount}
            
        except Exception as e:
            logger.error(f"Error updating search vectors: {e}")
            await db.rollback()
            return {"success": False, "error": str(e)}
    
    async def get_rag_statistics(self, db: AsyncSession = None) -> Dict[str, Any]:
        """
        Get RAG system statistics
        """
        try:
            # Count questions with embeddings
            embedding_query = select(func.count(Question.id)).where(Question.embedding.isnot(None))
            result = await db.execute(embedding_query)
            questions_with_embeddings = result.scalar()
            
            # Count total questions
            total_query = select(func.count(Question.id))
            result = await db.execute(total_query)
            total_questions = result.scalar()
            
            # Count by subject
            subject_query = select(Question.subject, func.count(Question.id)).group_by(Question.subject)
            result = await db.execute(subject_query)
            subject_counts = dict(result.fetchall())
            
            return {
                "total_questions": total_questions,
                "questions_with_embeddings": questions_with_embeddings,
                "embedding_coverage": round(questions_with_embeddings / total_questions * 100, 2) if total_questions > 0 else 0,
                "subject_distribution": subject_counts,
                "similarity_threshold": self.similarity_threshold,
                "max_results": self.max_results
            }
            
        except Exception as e:
            logger.error(f"Error getting RAG statistics: {e}")
            return {"error": str(e)}


# Global instance
real_rag_service = RealRAGService()
