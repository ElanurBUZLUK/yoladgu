from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import json
import asyncio
from collections import defaultdict
from datetime import datetime
import numpy as np

from app.core.cache import cache_service
from app.services.embedding_service import embedding_service
from app.core.database import database

logger = logging.getLogger(__name__)


class AdvancedRetrievalService:
    """Advanced retrieval service with reranking, MMR diversification, and hybrid optimization"""
    
    def __init__(self):
        self.cache_ttl = 1800  # 30 minutes
        self.mmr_lambda = 0.7  # Diversity vs relevance balance
        self.max_rerank_candidates = 100
        self.min_similarity_threshold = 0.3
        
        # Reranking weights
        self.semantic_weight = 0.4
        self.keyword_weight = 0.2
        self.diversity_weight = 0.2
        self.freshness_weight = 0.1
        self.quality_weight = 0.1
        
        # MMR settings
        self.mmr_diversity_threshold = 0.5
        self.mmr_max_iterations = 50
        
        # Query expansion settings
        self.expansion_threshold = 0.8
        self.max_expansion_terms = 5
    
    async def advanced_retrieve(
        self,
        query: str,
        user_id: str,
        subject: str,
        topic: Optional[str] = None,
        difficulty_level: Optional[int] = None,
        limit: int = 10,
        use_reranking: bool = True,
        use_mmr: bool = True,
        use_query_expansion: bool = True
    ) -> Dict[str, Any]:
        """Advanced retrieval with multiple optimization techniques"""
        
        try:
            start_time = datetime.now()
            
            # Step 1: Query expansion
            if use_query_expansion:
                expanded_query = await self._expand_query(query, topic, subject)
            else:
                expanded_query = query
            
            # Step 2: Initial retrieval
            initial_results = await self._initial_retrieval(
                expanded_query, user_id, subject, topic, difficulty_level, limit * 3
            )
            
            # Step 3: Reranking
            if use_reranking and len(initial_results) > 1:
                reranked_results = await self._rerank_results(
                    initial_results, query, user_id, subject
                )
            else:
                reranked_results = initial_results
            
            # Step 4: MMR diversification
            if use_mmr and len(reranked_results) > 1:
                diversified_results = await self._apply_mmr_diversification(
                    reranked_results, query, limit
                )
            else:
                diversified_results = reranked_results[:limit]
            
            # Step 5: Final processing
            final_results = await self._process_final_results(diversified_results, user_id)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            return {
                "results": final_results,
                "total_candidates": len(initial_results),
                "reranked_candidates": len(reranked_results),
                "final_count": len(final_results),
                "processing_time": processing_time,
                "techniques_used": {
                    "query_expansion": use_query_expansion,
                    "reranking": use_reranking,
                    "mmr_diversification": use_mmr
                },
                "query_info": {
                    "original_query": query,
                    "expanded_query": expanded_query,
                    "subject": subject,
                    "topic": topic,
                    "difficulty_level": difficulty_level
                }
            }
            
        except Exception as e:
            logger.error(f"Error in advanced retrieval: {e}")
            return {
                "results": [],
                "error": str(e),
                "processing_time": 0,
                "techniques_used": {}
            }
    
    async def _expand_query(
        self,
        query: str,
        topic: Optional[str],
        subject: str
    ) -> str:
        """Expand query with related terms"""
        
        try:
            # Get query embedding
            query_embedding = await embedding_service.get_embedding(query)
            
            # Find similar terms from existing questions
            similar_terms = await self._find_similar_terms(query_embedding, topic, subject)
            
            # Build expanded query
            expanded_terms = [query]
            
            # Add topic-specific terms
            if topic:
                expanded_terms.append(topic)
            
            # Add similar terms
            for term, similarity in similar_terms:
                if similarity >= self.expansion_threshold:
                    expanded_terms.append(term)
            
            # Limit expansion terms
            expanded_terms = expanded_terms[:self.max_expansion_terms]
            
            expanded_query = " ".join(expanded_terms)
            logger.debug(f"Query expanded: '{query}' -> '{expanded_query}'")
            
            return expanded_query
            
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return query
    
    async def _find_similar_terms(
        self,
        query_embedding: List[float],
        topic: Optional[str],
        subject: str
    ) -> List[Tuple[str, float]]:
        """Find terms similar to the query"""
        
        try:
            # Convert embedding to PostgreSQL array format
            embedding_array = f"[{','.join(map(str, query_embedding))}]"
            
            # Query for similar content
            query = f"""
                SELECT DISTINCT 
                    unnest(string_to_array(content, ' ')) as term,
                    content_embedding <-> '{embedding_array}'::vector as distance
                FROM questions
                WHERE is_active = true
                AND subject = %s
                AND content_embedding IS NOT NULL
                AND length(content) > 20
            """
            
            params = [subject]
            if topic:
                query += " AND topic ILIKE %s"
                params.append(f"%{topic}%")
            
            query += """
                ORDER BY distance ASC
                LIMIT 50
            """
            
            result = await database.fetch_all(query, params)
            
            # Process terms
            term_similarities = defaultdict(list)
            for row in result:
                term = row["term"].lower().strip()
                distance = float(row["distance"])
                similarity = 1 - distance
                
                # Filter out common words and short terms
                if len(term) > 3 and term not in ['the', 'and', 'for', 'with', 'this', 'that']:
                    term_similarities[term].append(similarity)
            
            # Calculate average similarity for each term
            term_avg_similarities = []
            for term, similarities in term_similarities.items():
                avg_similarity = sum(similarities) / len(similarities)
                term_avg_similarities.append((term, avg_similarity))
            
            # Sort by similarity and return top terms
            term_avg_similarities.sort(key=lambda x: x[1], reverse=True)
            return term_avg_similarities[:10]
            
        except Exception as e:
            logger.error(f"Error finding similar terms: {e}")
            return []
    
    async def _initial_retrieval(
        self,
        query: str,
        user_id: str,
        subject: str,
        topic: Optional[str],
        difficulty_level: Optional[int],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Perform initial retrieval using hybrid search"""
        
        try:
            # Generate query embedding
            query_embedding = await embedding_service.get_embedding(query)
            
            # Build search query
            embedding_array = f"[{','.join(map(str, query_embedding))}]"
            
            search_query = f"""
                SELECT 
                    id, content, topic, difficulty_level, question_type,
                    options, correct_answer, explanation, tags,
                    content_embedding <-> '{embedding_array}'::vector as distance,
                    1 - (content_embedding <=> '{embedding_array}'::vector) as similarity,
                    usage_count, created_at,
                    CASE 
                        WHEN content ILIKE %s THEN 1.0
                        WHEN content ILIKE %s THEN 0.8
                        WHEN content ILIKE %s THEN 0.6
                        ELSE 0.0
                    END as keyword_score
                FROM questions
                WHERE is_active = true
                AND subject = %s
                AND content_embedding IS NOT NULL
            """
            
            # Add filters
            params = [f"%{query}%", f"%{query.split()[0]}%", f"%{query.split()[-1]}%", subject]
            
            if topic:
                search_query += " AND topic ILIKE %s"
                params.append(f"%{topic}%")
            
            if difficulty_level:
                search_query += " AND difficulty_level = %s"
                params.append(difficulty_level)
            
            # Add ordering and limit
            search_query += f"""
                ORDER BY (similarity * 0.7 + keyword_score * 0.3) DESC
                LIMIT {limit}
            """
            
            result = await database.fetch_all(search_query, params)
            
            # Convert to response format
            results = []
            for row in result:
                results.append({
                    "id": str(row["id"]),
                    "content": row["content"],
                    "topic": row["topic"],
                    "difficulty_level": row["difficulty_level"],
                    "question_type": row["question_type"],
                    "options": row["options"],
                    "correct_answer": row["correct_answer"],
                    "explanation": row["explanation"],
                    "tags": row["tags"],
                    "similarity": float(row["similarity"]),
                    "keyword_score": float(row["keyword_score"]),
                    "usage_count": row["usage_count"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "combined_score": float(row["similarity"]) * 0.7 + float(row["keyword_score"]) * 0.3
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in initial retrieval: {e}")
            return []
    
    async def _rerank_results(
        self,
        results: List[Dict[str, Any]],
        original_query: str,
        user_id: str,
        subject: str
    ) -> List[Dict[str, Any]]:
        """Rerank results using multiple factors"""
        
        try:
            if not results:
                return results
            
            # Get user preferences and history
            user_preferences = await self._get_user_preferences(user_id, subject)
            
            # Calculate reranking scores
            reranked_results = []
            for result in results:
                rerank_score = await self._calculate_rerank_score(
                    result, original_query, user_preferences
                )
                
                result_copy = result.copy()
                result_copy["rerank_score"] = rerank_score
                reranked_results.append(result_copy)
            
            # Sort by rerank score
            reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            logger.debug(f"Reranked {len(riscored_results)} results")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return results
    
    async def _calculate_rerank_score(
        self,
        result: Dict[str, Any],
        query: str,
        user_preferences: Dict[str, Any]
    ) -> float:
        """Calculate comprehensive reranking score"""
        
        try:
            # Base semantic similarity
            semantic_score = result.get("similarity", 0.0) * self.semantic_weight
            
            # Keyword relevance
            keyword_score = result.get("keyword_score", 0.0) * self.keyword_weight
            
            # Content diversity (based on topic)
            diversity_score = self._calculate_diversity_score(result, user_preferences) * self.diversity_weight
            
            # Content freshness
            freshness_score = self._calculate_freshness_score(result) * self.freshness_weight
            
            # Content quality
            quality_score = self._calculate_quality_score(result) * self.quality_weight
            
            # User preference alignment
            preference_score = self._calculate_preference_score(result, user_preferences)
            
            # Combine all scores
            total_score = (
                semantic_score +
                keyword_score +
                diversity_score +
                freshness_score +
                quality_score +
                preference_score
            )
            
            return total_score
            
        except Exception as e:
            logger.error(f"Error calculating rerank score: {e}")
            return result.get("similarity", 0.0)
    
    def _calculate_diversity_score(
        self,
        result: Dict[str, Any],
        user_preferences: Dict[str, Any]
    ) -> float:
        """Calculate diversity score based on user history"""
        
        try:
            # Get user's recent topics
            recent_topics = user_preferences.get("recent_topics", [])
            current_topic = result.get("topic", "")
            
            # Higher score for topics not recently seen
            if current_topic not in recent_topics:
                return 1.0
            elif len(recent_topics) > 0:
                # Lower score for recently seen topics
                topic_index = recent_topics.index(current_topic)
                return max(0.1, 1.0 - (topic_index * 0.2))
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating diversity score: {e}")
            return 0.5
    
    def _calculate_freshness_score(self, result: Dict[str, Any]) -> float:
        """Calculate freshness score based on creation date"""
        
        try:
            created_at = result.get("created_at")
            if not created_at:
                return 0.5
            
            # Parse creation date
            if isinstance(created_at, str):
                created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            else:
                created_date = created_at
            
            # Calculate days since creation
            days_old = (datetime.now() - created_date).days
            
            # Higher score for newer content
            if days_old <= 1:
                return 1.0
            elif days_old <= 7:
                return 0.8
            elif days_old <= 30:
                return 0.6
            elif days_old <= 90:
                return 0.4
            else:
                return 0.2
                
        except Exception as e:
            logger.error(f"Error calculating freshness score: {e}")
            return 0.5
    
    def _calculate_quality_score(self, result: Dict[str, Any]) -> float:
        """Calculate quality score based on content characteristics"""
        
        try:
            content = result.get("content", "")
            explanation = result.get("explanation", "")
            
            # Content length score
            length_score = min(1.0, len(content) / 100)
            
            # Explanation quality score
            explanation_score = min(1.0, len(explanation) / 50) if explanation else 0.0
            
            # Usage count score (popular content might be better)
            usage_count = result.get("usage_count", 0)
            usage_score = min(1.0, usage_count / 100)
            
            # Combine scores
            quality_score = (length_score * 0.4 + explanation_score * 0.4 + usage_score * 0.2)
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.5
    
    def _calculate_preference_score(
        self,
        result: Dict[str, Any],
        user_preferences: Dict[str, Any]
    ) -> float:
        """Calculate preference alignment score"""
        
        try:
            # Get user preferences
            preferred_difficulty = user_preferences.get("preferred_difficulty", 3)
            preferred_topics = user_preferences.get("preferred_topics", [])
            preferred_formats = user_preferences.get("preferred_formats", [])
            
            # Difficulty preference
            current_difficulty = result.get("difficulty_level", 3)
            difficulty_score = 1.0 - abs(current_difficulty - preferred_difficulty) / 5
            
            # Topic preference
            current_topic = result.get("topic", "")
            topic_score = 1.0 if current_topic in preferred_topics else 0.5
            
            # Format preference
            current_format = result.get("question_type", "")
            format_score = 1.0 if current_format in preferred_formats else 0.5
            
            # Combine preference scores
            preference_score = (difficulty_score * 0.4 + topic_score * 0.3 + format_score * 0.3)
            
            return preference_score * 0.2  # Lower weight for preferences
            
        except Exception as e:
            logger.error(f"Error calculating preference score: {e}")
            return 0.0
    
    async def _apply_mmr_diversification(
        self,
        results: List[Dict[str, Any]],
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Apply Maximal Marginal Relevance diversification"""
        
        try:
            if len(results) <= limit:
                return results[:limit]
            
            # Get query embedding
            query_embedding = await embedding_service.get_embedding(query)
            
            # Initialize selected and remaining sets
            selected = []
            remaining = results.copy()
            
            # Select first item (highest relevance)
            if remaining:
                selected.append(remaining.pop(0))
            
            # Apply MMR selection
            for _ in range(min(limit - 1, len(remaining), self.mmr_max_iterations)):
                if not remaining:
                    break
                
                best_mmr_score = -1
                best_index = 0
                
                # Calculate MMR score for each remaining item
                for i, item in enumerate(remaining):
                    # Relevance to query
                    relevance = item.get("rerank_score", item.get("similarity", 0.0))
                    
                    # Diversity from selected items
                    diversity = self._calculate_mmr_diversity(item, selected)
                    
                    # MMR score
                    mmr_score = self.mmr_lambda * relevance + (1 - self.mmr_lambda) * diversity
                    
                    if mmr_score > best_mmr_score:
                        best_mmr_score = mmr_score
                        best_index = i
                
                # Select item with best MMR score
                selected.append(remaining.pop(best_index))
            
            logger.debug(f"MMR diversification selected {len(selected)} items from {len(results)} candidates")
            return selected
            
        except Exception as e:
            logger.error(f"Error in MMR diversification: {e}")
            return results[:limit]
    
    def _calculate_mmr_diversity(
        self,
        item: Dict[str, Any],
        selected_items: List[Dict[str, Any]]
    ) -> float:
        """Calculate diversity score for MMR"""
        
        try:
            if not selected_items:
                return 1.0
            
            # Calculate average similarity to selected items
            similarities = []
            for selected_item in selected_items:
                # Use topic and format for diversity calculation
                topic_similarity = 1.0 if item.get("topic") == selected_item.get("topic") else 0.0
                format_similarity = 1.0 if item.get("question_type") == selected_item.get("question_type") else 0.0
                difficulty_similarity = 1.0 - abs(item.get("difficulty_level", 3) - selected_item.get("difficulty_level", 3)) / 5
                
                # Combined similarity
                combined_similarity = (topic_similarity * 0.4 + format_similarity * 0.3 + difficulty_similarity * 0.3)
                similarities.append(combined_similarity)
            
            # Diversity is inverse of average similarity
            avg_similarity = sum(similarities) / len(similarities)
            diversity = 1.0 - avg_similarity
            
            return max(0.0, diversity)
            
        except Exception as e:
            logger.error(f"Error calculating MMR diversity: {e}")
            return 0.5
    
    async def _get_user_preferences(self, user_id: str, subject: str) -> Dict[str, Any]:
        """Get user preferences and history"""
        
        try:
            # This would query user preferences from database
            # For now, return default preferences
            return {
                "preferred_difficulty": 3,
                "preferred_topics": [],
                "preferred_formats": ["mcq", "fill_blank"],
                "recent_topics": [],
                "learning_style": "visual"
            }
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return {}
    
    async def _process_final_results(
        self,
        results: List[Dict[str, Any]],
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Process final results for response"""
        
        try:
            processed_results = []
            
            for result in results:
                # Remove internal scoring fields
                processed_result = {
                    "id": result["id"],
                    "content": result["content"],
                    "topic": result["topic"],
                    "difficulty_level": result["difficulty_level"],
                    "question_type": result["question_type"],
                    "options": result["options"],
                    "correct_answer": result["correct_answer"],
                    "explanation": result["explanation"],
                    "tags": result["tags"],
                    "similarity_score": result.get("similarity", 0.0),
                    "rerank_score": result.get("rerank_score", 0.0),
                    "usage_count": result.get("usage_count", 0),
                    "created_at": result.get("created_at")
                }
                
                processed_results.append(processed_result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error processing final results: {e}")
            return results
    
    async def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get retrieval service statistics"""
        
        return {
            "mmr_lambda": self.mmr_lambda,
            "max_rerank_candidates": self.max_rerank_candidates,
            "min_similarity_threshold": self.min_similarity_threshold,
            "reranking_weights": {
                "semantic": self.semantic_weight,
                "keyword": self.keyword_weight,
                "diversity": self.diversity_weight,
                "freshness": self.freshness_weight,
                "quality": self.quality_weight
            },
            "mmr_settings": {
                "diversity_threshold": self.mmr_diversity_threshold,
                "max_iterations": self.mmr_max_iterations
            },
            "query_expansion": {
                "expansion_threshold": self.expansion_threshold,
                "max_expansion_terms": self.max_expansion_terms
            }
        }


# Global instance
advanced_retrieval_service = AdvancedRetrievalService()
