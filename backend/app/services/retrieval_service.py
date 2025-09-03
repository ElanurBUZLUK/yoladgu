"""
Hybrid retrieval service combining dense and sparse search with RRF fusion.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.vector_service import vector_service
from app.services.search_service import search_service
from app.db.repositories.user import user_repository


class RetrievalService:
    """Hybrid retrieval service with caching and fusion."""
    
    def __init__(self):
        self.redis_client = None
        self.cache_ttl = 24 * 3600  # 24 hours
        self.cache_prefix = "retrieval:"
        
        # Fusion weights
        self.dense_weight = 0.6
        self.sparse_weight = 0.4
        
        # RRF parameters
        self.rrf_k = 60  # RRF constant
    
    async def _get_redis_client(self) -> redis.Redis:
        """Get Redis client for caching."""
        if self.redis_client is None:
            self.redis_client = redis.from_url(settings.REDIS_URL)
        return self.redis_client
    
    def _get_cache_key(self, query_hash: str) -> str:
        """Generate cache key for retrieval results."""
        return f"{self.cache_prefix}{query_hash}"
    
    def _hash_query(self, query_params: Dict[str, Any]) -> str:
        """Create hash for query parameters."""
        import hashlib
        query_str = json.dumps(query_params, sort_keys=True)
        return hashlib.md5(query_str.encode()).hexdigest()
    
    async def _get_from_cache(self, query_hash: str) -> Optional[List[Dict[str, Any]]]:
        """Get results from cache."""
        try:
            redis_client = await self._get_redis_client()
            cached_data = await redis_client.get(self._get_cache_key(query_hash))
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            print(f"Cache get error: {e}")
        return None
    
    async def _set_cache(self, query_hash: str, results: List[Dict[str, Any]]) -> None:
        """Set results in cache."""
        try:
            redis_client = await self._get_redis_client()
            await redis_client.setex(
                self._get_cache_key(query_hash),
                self.cache_ttl,
                json.dumps(results, default=str)
            )
        except Exception as e:
            print(f"Cache set error: {e}")
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combine dense and sparse results using Reciprocal Rank Fusion (RRF).
        
        RRF Score = sum(1 / (k + rank)) for each ranking
        
        Args:
            dense_results: Results from vector search
            sparse_results: Results from BM25 search
            k: RRF constant (typically 60)
            
        Returns:
            Fused and ranked results
        """
        # Create item score maps
        item_scores = {}
        
        # Process dense results
        for rank, result in enumerate(dense_results):
            item_id = result["item_id"]
            rrf_score = 1.0 / (k + rank + 1)  # +1 because rank is 0-indexed
            
            if item_id not in item_scores:
                item_scores[item_id] = {
                    "item_id": item_id,
                    "dense_score": result["score"],
                    "sparse_score": 0.0,
                    "dense_rank": rank + 1,
                    "sparse_rank": None,
                    "rrf_score": 0.0,
                    "metadata": result.get("metadata", {})
                }
            
            item_scores[item_id]["rrf_score"] += self.dense_weight * rrf_score
        
        # Process sparse results
        for rank, result in enumerate(sparse_results):
            item_id = result["item_id"]
            rrf_score = 1.0 / (k + rank + 1)
            
            if item_id not in item_scores:
                item_scores[item_id] = {
                    "item_id": item_id,
                    "dense_score": 0.0,
                    "sparse_score": result["score"],
                    "dense_rank": None,
                    "sparse_rank": rank + 1,
                    "rrf_score": 0.0,
                    "metadata": result.get("metadata", {})
                }
            else:
                item_scores[item_id]["sparse_score"] = result["score"]
                item_scores[item_id]["sparse_rank"] = rank + 1
            
            item_scores[item_id]["rrf_score"] += self.sparse_weight * rrf_score
        
        # Sort by RRF score and return
        fused_results = sorted(
            item_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )
        
        return fused_results
    
    def _apply_metadata_filters(
        self,
        results: List[Dict[str, Any]],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply metadata filters to results."""
        if not filters:
            return results
        
        filtered_results = []
        
        for result in results:
            metadata = result.get("metadata", {})
            include_item = True
            
            for key, value in filters.items():
                if key in metadata:
                    if isinstance(value, list):
                        # Check if any value matches
                        if metadata[key] not in value:
                            include_item = False
                            break
                    else:
                        # Exact match
                        if metadata[key] != value:
                            include_item = False
                            break
                else:
                    # Required field missing
                    include_item = False
                    break
            
            if include_item:
                filtered_results.append(result)
        
        return filtered_results
    
    async def hybrid_search(
        self,
        query: str,
        item_type: Optional[str] = None,
        lang: str = "tr",
        skills: Optional[List[str]] = None,
        difficulty_range: Optional[Tuple[float, float]] = None,
        limit: int = 200,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense and sparse retrieval.
        
        Args:
            query: Search query text
            item_type: "math" or "english"
            lang: Language code
            skills: Skill tags to filter by
            difficulty_range: (min, max) difficulty for math items
            limit: Maximum results to return
            use_cache: Whether to use caching
            
        Returns:
            Hybrid search results with RRF scores
        """
        # Create query parameters for caching
        query_params = {
            "query": query,
            "item_type": item_type,
            "lang": lang,
            "skills": skills,
            "difficulty_range": difficulty_range,
            "limit": limit
        }
        
        query_hash = self._hash_query(query_params)
        
        # Try cache first
        if use_cache:
            cached_results = await self._get_from_cache(query_hash)
            if cached_results:
                return cached_results[:limit]
        
        # Perform parallel dense and sparse searches
        dense_task = vector_service.search_by_skills(
            skills=skills or [],
            item_type=item_type or "math",
            lang=lang,
            difficulty_range=difficulty_range,
            limit=limit
        )
        
        sparse_task = search_service.search_by_skills(
            skills=skills or [],
            item_type=item_type or "math",
            lang=lang,
            difficulty_range=difficulty_range,
            limit=limit
        )
        
        # Execute searches in parallel
        dense_results, sparse_results = await asyncio.gather(
            dense_task, sparse_task, return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(dense_results, Exception):
            print(f"Dense search error: {dense_results}")
            dense_results = []
        
        if isinstance(sparse_results, Exception):
            print(f"Sparse search error: {sparse_results}")
            sparse_results = []
        
        # Fuse results using RRF
        fused_results = self._reciprocal_rank_fusion(
            dense_results, sparse_results, self.rrf_k
        )
        
        # Apply additional filters
        filters = {"lang": lang, "status": "active"}
        if item_type:
            filters["type"] = item_type
        
        filtered_results = self._apply_metadata_filters(fused_results, filters)
        
        # Limit results
        final_results = filtered_results[:limit]
        
        # Cache results
        if use_cache and final_results:
            await self._set_cache(query_hash, final_results)
        
        return final_results
    
    async def search_for_user(
        self,
        session: AsyncSession,
        user_id: str,
        target_skills: Optional[List[str]] = None,
        item_type: str = "math",
        limit: int = 200,
        use_personalization: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search items personalized for a specific user.
        
        Args:
            session: Database session
            user_id: User ID
            target_skills: Skills to focus on
            item_type: "math" or "english"
            limit: Maximum results
            use_personalization: Whether to use user profile for personalization
            
        Returns:
            Personalized search results
        """
        # Get user profile for personalization
        user = await user_repository.get(session, user_id)
        if not user:
            return []
        
        # Determine search parameters based on user profile
        lang = user.lang
        query_skills = target_skills or []
        
        if use_personalization:
            # Get user's weak skills for targeting
            error_profile = (
                user.error_profile_math if item_type == "math"
                else user.error_profile_en
            ) or {}
            
            # Add weak skills to search if no specific skills provided
            if not target_skills and error_profile:
                weak_skills = [
                    skill for skill, error_rate in error_profile.items()
                    if error_rate >= 0.3  # High error rate threshold
                ]
                query_skills.extend(weak_skills[:3])  # Top 3 weak skills
            
            # Get optimal difficulty range for math items
            if item_type == "math":
                theta = user.theta_math or 0.0
                # Target slightly easier items for practice
                difficulty_range = (theta - 0.5, theta + 0.3)
            else:
                difficulty_range = None
        else:
            difficulty_range = None
        
        # Create search query from skills
        query = " ".join(query_skills) if query_skills else ""
        
        # Perform hybrid search
        results = await self.hybrid_search(
            query=query,
            item_type=item_type,
            lang=lang,
            skills=query_skills,
            difficulty_range=difficulty_range,
            limit=limit
        )
        
        return results
    
    async def get_similar_items(
        self,
        item_id: str,
        item_type: str,
        limit: int = 10,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Get items similar to a given item."""
        # Try vector similarity first (more accurate for content similarity)
        similar_items = await vector_service.get_similar_items(
            item_id=item_id,
            limit=limit,
            filters={"type": item_type, "status": "active"}
        )
        
        return similar_items
    
    async def search_by_content(
        self,
        content: str,
        item_type: str,
        lang: str = "tr",
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search for items with similar content."""
        return await self.hybrid_search(
            query=content,
            item_type=item_type,
            lang=lang,
            limit=limit
        )
    
    async def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval service statistics."""
        try:
            # Get cache stats
            redis_client = await self._get_redis_client()
            cache_info = await redis_client.info("memory")
            
            # Get search engine stats
            search_stats = await search_service.get_index_stats()
            
            # Get vector DB stats
            vector_stats = await vector_service.get_collection_info()
            
            return {
                "cache": {
                    "memory_used": cache_info.get("used_memory_human", "N/A"),
                    "hit_rate": "N/A"  # Would need to track this separately
                },
                "search_engine": search_stats,
                "vector_db": vector_stats,
                "fusion_weights": {
                    "dense": self.dense_weight,
                    "sparse": self.sparse_weight
                }
            }
        except Exception as e:
            print(f"Error getting retrieval stats: {e}")
            return {}
    
    async def invalidate_cache(self, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries matching pattern."""
        try:
            redis_client = await self._get_redis_client()
            
            if pattern:
                # Delete keys matching pattern
                keys = await redis_client.keys(f"{self.cache_prefix}{pattern}*")
            else:
                # Delete all retrieval cache
                keys = await redis_client.keys(f"{self.cache_prefix}*")
            
            if keys:
                deleted = await redis_client.delete(*keys)
                return deleted
            return 0
        except Exception as e:
            print(f"Error invalidating cache: {e}")
            return 0


# Create service instance
retrieval_service = RetrievalService()