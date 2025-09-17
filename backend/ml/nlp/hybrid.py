"""
Hybrid Search with Reciprocal Rank Fusion (RRF)
Combines BM25 and dense search results using RRF algorithm
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import structlog
from .embedder import e5_embedder
from .index import HybridSearchIndex, SearchResult

logger = structlog.get_logger()

class ReciprocalRankFusion:
    """Reciprocal Rank Fusion for combining search results"""
    
    def __init__(self, k: int = 60):
        self.k = k  # RRF parameter
        
    def fuse(self, ranked_lists: List[List[str]]) -> List[Tuple[str, float]]:
        """
        Fuse multiple ranked lists using RRF
        
        Args:
            ranked_lists: List of ranked document ID lists
            
        Returns:
            List of (doc_id, score) tuples sorted by score
        """
        scores = defaultdict(float)
        
        for ranked_list in ranked_lists:
            for rank, doc_id in enumerate(ranked_list, 1):
                # RRF formula: 1 / (k + rank)
                scores[doc_id] += 1.0 / (self.k + rank)
        
        # Sort by score (descending)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

class PersonalizedReranker:
    """Personalized reranking using user features"""
    
    def __init__(self):
        self.feature_weights = {
            "bm25_score": 0.3,
            "dense_score": 0.3,
            "recency": 0.2,
            "user_topic_similarity": 0.1,
            "difficulty_match": 0.1
        }
        
    def extract_features(self, result: SearchResult, user_context: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for reranking"""
        features = {}
        
        # BM25 and dense scores (normalized)
        features["bm25_score"] = getattr(result, "bm25_score", 0.0)
        features["dense_score"] = getattr(result, "dense_score", 0.0)
        
        # Recency (based on timestamp if available)
        if "timestamp" in result.metadata:
            import time
            current_time = time.time()
            doc_time = result.metadata["timestamp"]
            features["recency"] = 1.0 / (1.0 + (current_time - doc_time) / (24 * 3600))  # Daily decay
        else:
            features["recency"] = 0.5
        
        # User topic similarity (simplified)
        if "user_topics" in user_context and "topics" in result.metadata:
            user_topics = set(user_context["user_topics"])
            doc_topics = set(result.metadata["topics"])
            intersection = len(user_topics.intersection(doc_topics))
            union = len(user_topics.union(doc_topics))
            features["user_topic_similarity"] = intersection / union if union > 0 else 0.0
        else:
            features["user_topic_similarity"] = 0.0
        
        # Difficulty match
        if "user_level" in user_context and "difficulty" in result.metadata:
            user_level = user_context["user_level"]
            doc_difficulty = result.metadata["difficulty"]
            # Simple difficulty matching
            if user_level == doc_difficulty:
                features["difficulty_match"] = 1.0
            elif abs(user_level - doc_difficulty) == 1:
                features["difficulty_match"] = 0.7
            else:
                features["difficulty_match"] = 0.3
        else:
            features["difficulty_match"] = 0.5
        
        return features
    
    def rerank(self, results: List[SearchResult], user_context: Dict[str, Any]) -> List[SearchResult]:
        """Rerank results based on user features"""
        if not results:
            return results
        
        # Extract features and calculate scores
        scored_results = []
        for result in results:
            features = self.extract_features(result, user_context)
            score = sum(self.feature_weights[feature] * value 
                       for feature, value in features.items())
            
            # Create new result with personalized score
            personalized_result = SearchResult(
                doc_id=result.doc_id,
                score=score,
                text=result.text,
                metadata=result.metadata
            )
            # Keep original scores for reference
            personalized_result.bm25_score = features["bm25_score"]
            personalized_result.dense_score = features["dense_score"]
            
            scored_results.append(personalized_result)
        
        # Sort by personalized score
        return sorted(scored_results, key=lambda x: x.score, reverse=True)

class HybridSearchEngine:
    """Main hybrid search engine combining BM25, dense search, and RRF"""
    
    def __init__(self, rrf_k: int = 60, enable_reranking: bool = True):
        self.index = HybridSearchIndex()
        self.e5_embedder = e5_embedder
        self.rrf = ReciprocalRankFusion(k=rrf_k)
        self.reranker = PersonalizedReranker() if enable_reranking else None
        self._initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the hybrid search engine"""
        try:
            # Initialize E5 embedder
            embedder_success = await self.e5_embedder.initialize()
            if not embedder_success:
                logger.error("Failed to initialize E5 embedder")
                return False
            
            # Initialize hybrid index
            index_success = await self.index.initialize()
            if not index_success:
                logger.error("Failed to initialize hybrid index")
                return False
            
            self._initialized = True
            logger.info("Hybrid search engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to initialize hybrid search engine: %s", e)
            return False
    
    async def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the hybrid search index"""
        if not self._initialized:
            raise RuntimeError("Hybrid search engine not initialized")
        
        try:
            # Extract texts for embedding
            texts = [doc["text"] for doc in documents]
            
            # Generate embeddings using E5
            embeddings = self.e5_embedder.encode_passages(texts)
            
            # Add to hybrid index
            self.index.add_documents(documents, embeddings)
            
            logger.info("Added %d documents to hybrid search index", len(documents))
            
        except Exception as e:
            logger.error("Failed to add documents: %s", e)
            raise
    
    async def search(self, query: str, k: int = 10, user_context: Optional[Dict[str, Any]] = None,
                    search_type: str = "hybrid") -> List[SearchResult]:
        """
        Search using hybrid approach
        
        Args:
            query: Search query
            k: Number of results to return
            user_context: User context for personalization
            search_type: "bm25", "dense", "hybrid", or "rrf"
        """
        if not self._initialized:
            raise RuntimeError("Hybrid search engine not initialized")
        
        try:
            if search_type == "bm25":
                return self.index.search_bm25(query, k)
            
            elif search_type == "dense":
                query_vector = self.e5_embedder.encode_single_query(query)
                return self.index.search_dense(query_vector, k)
            
            elif search_type == "hybrid":
                query_vector = self.e5_embedder.encode_single_query(query)
                return self.index.search_hybrid(query, query_vector, k)
            
            elif search_type == "rrf":
                return await self._search_with_rrf(query, k, user_context)
            
            else:
                raise ValueError(f"Unknown search type: {search_type}")
                
        except Exception as e:
            logger.error("Search failed: %s", e)
            return []
    
    async def _search_with_rrf(self, query: str, k: int, user_context: Optional[Dict[str, Any]]) -> List[SearchResult]:
        """Search using Reciprocal Rank Fusion"""
        try:
            # Get BM25 results
            bm25_results = self.index.search_bm25(query, k * 2)
            bm25_doc_ids = [r.doc_id for r in bm25_results]
            
            # Get dense results
            query_vector = self.e5_embedder.encode_single_query(query)
            dense_results = self.index.search_dense(query_vector, k * 2)
            dense_doc_ids = [r.doc_id for r in dense_results]
            
            # Fuse using RRF
            fused_doc_ids = self.rrf.fuse([bm25_doc_ids, dense_doc_ids])
            
            # Create result mapping
            all_results = {r.doc_id: r for r in bm25_results + dense_results}
            
            # Build fused results
            fused_results = []
            for doc_id, rrf_score in fused_doc_ids[:k]:
                if doc_id in all_results:
                    result = all_results[doc_id]
                    # Update score with RRF score
                    result.score = rrf_score
                    fused_results.append(result)
            
            # Apply personalized reranking if enabled and user context provided
            if self.reranker and user_context:
                fused_results = self.reranker.rerank(fused_results, user_context)
            
            return fused_results
            
        except Exception as e:
            logger.error("RRF search failed: %s", e)
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        return {
            "initialized": self._initialized,
            "embedder_initialized": self.e5_embedder.is_initialized,
            "index_stats": self.index.get_stats(),
            "rrf_k": self.rrf.k,
            "reranking_enabled": self.reranker is not None
        }

# Global instance
hybrid_search_engine = HybridSearchEngine()
