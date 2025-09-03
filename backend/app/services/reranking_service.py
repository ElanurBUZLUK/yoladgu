"""
Re-ranking service with cross-encoder model for relevance scoring.

This service implements cross-encoder based re-ranking to improve
the relevance of retrieved candidates for personalized recommendations.
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import json

import numpy as np
from sentence_transformers import CrossEncoder
import torch

from app.core.config import settings

logger = logging.getLogger(__name__)


class RerankingService:
    """Cross-encoder based re-ranking service."""
    
    def __init__(self):
        self.model = None
        self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 32
        self.max_length = 512
        
        # Cache configuration
        self.cache = {}
        self.cache_ttl = timedelta(hours=1)
        self.max_cache_size = 10000
        
        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "avg_latency_ms": 0.0,
            "model_load_time": None
        }
        
        # Initialize model asynchronously
        self._model_loaded = False
        self._loading_lock = asyncio.Lock()
    
    async def _load_model(self):
        """Load the cross-encoder model."""
        if self._model_loaded:
            return
        
        async with self._loading_lock:
            if self._model_loaded:
                return
            
            try:
                start_time = time.time()
                logger.info(f"Loading cross-encoder model: {self.model_name}")
                
                # Load model in a thread to avoid blocking
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(
                    None, 
                    lambda: CrossEncoder(self.model_name, device=self.device)
                )
                
                load_time = time.time() - start_time
                self.stats["model_load_time"] = load_time
                self._model_loaded = True
                
                logger.info(f"Cross-encoder model loaded in {load_time:.2f}s on {self.device}")
                
            except Exception as e:
                logger.error(f"Failed to load cross-encoder model: {e}")
                raise
    
    async def rerank_candidates(
        self,
        query_context: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        max_k: int = 40,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Re-rank candidates using cross-encoder model.
        
        Args:
            query_context: User context and query information
            candidates: List of candidate items to re-rank
            max_k: Maximum number of candidates to return
            use_cache: Whether to use caching
            
        Returns:
            Re-ranked candidates with scores
        """
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        try:
            # Ensure model is loaded
            await self._load_model()
            
            if not candidates:
                return []
            
            # Build query representation
            query_text = self._build_query_text(query_context)
            
            # Check cache
            cache_key = None
            if use_cache:
                cache_key = self._generate_cache_key(query_text, candidates)
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    self.stats["cache_hits"] += 1
                    return cached_result[:max_k]
            
            # Prepare query-candidate pairs
            pairs = []
            candidate_texts = []
            
            for candidate in candidates:
                candidate_text = self._build_candidate_text(candidate)
                candidate_texts.append(candidate_text)
                pairs.append([query_text, candidate_text])
            
            # Batch inference
            scores = await self._batch_predict(pairs)
            
            # Combine with heuristic scores
            enhanced_candidates = self._combine_scores(
                candidates, scores, query_context
            )
            
            # Sort by combined score
            enhanced_candidates.sort(
                key=lambda x: x.get("rerank_score", 0), 
                reverse=True
            )
            
            # Take top-k
            result = enhanced_candidates[:max_k]
            
            # Cache result
            if use_cache and cache_key:
                self._add_to_cache(cache_key, result)
            
            # Update stats
            latency = (time.time() - start_time) * 1000
            self._update_latency_stats(latency)
            
            logger.debug(f"Re-ranked {len(candidates)} candidates to {len(result)} in {latency:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            # Fallback to original candidates with heuristic scoring
            return self._fallback_ranking(candidates, query_context, max_k)
    
    async def _batch_predict(self, pairs: List[List[str]]) -> List[float]:
        """Perform batch inference with the cross-encoder model."""
        try:
            if not pairs:
                return []
            
            # Process in batches to manage memory
            all_scores = []
            
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i:i + self.batch_size]
                
                # Run inference in executor to avoid blocking
                loop = asyncio.get_event_loop()
                batch_scores = await loop.run_in_executor(
                    None,
                    lambda: self.model.predict(batch_pairs, convert_to_numpy=True)
                )
                
                all_scores.extend(batch_scores.tolist())
            
            return all_scores
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            # Return neutral scores as fallback
            return [0.5] * len(pairs)
    
    def _build_query_text(self, query_context: Dict[str, Any]) -> str:
        """Build query text from user context."""
        parts = []
        
        # Add explicit query if available
        if query_context.get("query"):
            parts.append(query_context["query"])
        
        # Add target skills
        target_skills = query_context.get("target_skills", [])
        if target_skills:
            parts.append(f"Skills: {', '.join(target_skills)}")
        
        # Add difficulty level
        theta_math = query_context.get("theta_math", 0.0)
        if theta_math > 0.5:
            parts.append("Advanced level")
        elif theta_math < -0.5:
            parts.append("Basic level")
        else:
            parts.append("Intermediate level")
        
        # Add language preference
        language = query_context.get("language", "tr")
        if language == "tr":
            parts.append("Turkish language")
        elif language == "en":
            parts.append("English language")
        
        # Add error profile context
        error_profiles = query_context.get("error_profiles", {})
        if error_profiles:
            error_areas = []
            for subject, errors in error_profiles.items():
                high_error_types = [
                    error_type for error_type, rate in errors.items() 
                    if rate > 0.3
                ]
                if high_error_types:
                    error_areas.extend(high_error_types)
            
            if error_areas:
                parts.append(f"Focus areas: {', '.join(error_areas[:3])}")
        
        return " | ".join(parts) if parts else "General math question"
    
    def _build_candidate_text(self, candidate: Dict[str, Any]) -> str:
        """Build candidate text for cross-encoder input."""
        parts = []
        
        metadata = candidate.get("metadata", {})
        
        # Add item type
        item_type = metadata.get("type", "unknown")
        parts.append(f"Type: {item_type}")
        
        # Add skills
        skills = metadata.get("skills", [])
        if skills:
            parts.append(f"Skills: {', '.join(skills[:3])}")
        
        # Add difficulty
        difficulty = metadata.get("difficulty", 0.0)
        if difficulty > 0.5:
            parts.append("Advanced difficulty")
        elif difficulty < -0.5:
            parts.append("Basic difficulty")
        else:
            parts.append("Intermediate difficulty")
        
        # Add language
        lang = metadata.get("lang", "tr")
        parts.append(f"Language: {lang}")
        
        # Add bloom level if available
        bloom_level = metadata.get("bloom_level")
        if bloom_level:
            parts.append(f"Bloom: {bloom_level}")
        
        # Add CEFR level for English items
        cefr_level = metadata.get("cefr_level")
        if cefr_level:
            parts.append(f"CEFR: {cefr_level}")
        
        # Add item content if available (truncated)
        item_content = ""
        if "stem" in metadata:
            item_content = metadata["stem"][:200]
        elif "passage" in metadata:
            item_content = metadata["passage"][:200]
        
        if item_content:
            parts.append(f"Content: {item_content}")
        
        return " | ".join(parts)
    
    def _combine_scores(
        self,
        candidates: List[Dict[str, Any]],
        cross_encoder_scores: List[float],
        query_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Combine cross-encoder scores with heuristic scores."""
        enhanced_candidates = []
        
        for i, candidate in enumerate(candidates):
            # Get cross-encoder score
            ce_score = cross_encoder_scores[i] if i < len(cross_encoder_scores) else 0.5
            
            # Get existing retrieval scores
            retrieval_scores = candidate.get("retriever_scores", {})
            max_retrieval_score = max(retrieval_scores.values()) if retrieval_scores else 0.0
            
            # Calculate heuristic scores
            skill_match_score = self._calculate_skill_match(candidate, query_context)
            difficulty_match_score = self._calculate_difficulty_match(candidate, query_context)
            freshness_score = self._calculate_freshness_score(candidate)
            
            # Weighted combination
            weights = {
                "cross_encoder": 0.4,
                "retrieval": 0.25,
                "skill_match": 0.15,
                "difficulty_match": 0.15,
                "freshness": 0.05
            }
            
            combined_score = (
                weights["cross_encoder"] * ce_score +
                weights["retrieval"] * max_retrieval_score +
                weights["skill_match"] * skill_match_score +
                weights["difficulty_match"] * difficulty_match_score +
                weights["freshness"] * freshness_score
            )
            
            # Add scores to candidate
            enhanced_candidate = candidate.copy()
            enhanced_candidate.update({
                "cross_encoder_score": ce_score,
                "skill_match_score": skill_match_score,
                "difficulty_match_score": difficulty_match_score,
                "freshness_score": freshness_score,
                "rerank_score": combined_score
            })
            
            enhanced_candidates.append(enhanced_candidate)
        
        return enhanced_candidates
    
    def _calculate_skill_match(
        self, 
        candidate: Dict[str, Any], 
        query_context: Dict[str, Any]
    ) -> float:
        """Calculate skill match score."""
        candidate_skills = set(candidate.get("metadata", {}).get("skills", []))
        target_skills = set(query_context.get("target_skills", []))
        
        if not target_skills:
            return 0.5  # Neutral if no target skills
        
        if not candidate_skills:
            return 0.0
        
        # Jaccard similarity
        intersection = len(candidate_skills & target_skills)
        union = len(candidate_skills | target_skills)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_difficulty_match(
        self, 
        candidate: Dict[str, Any], 
        query_context: Dict[str, Any]
    ) -> float:
        """Calculate difficulty match score."""
        candidate_difficulty = candidate.get("metadata", {}).get("difficulty", 0.0)
        user_theta = query_context.get("theta_math", 0.0)
        
        # Optimal difficulty is close to user's theta
        difficulty_diff = abs(candidate_difficulty - user_theta)
        
        # Score decreases as difference increases
        return max(0.0, 1.0 - difficulty_diff)
    
    def _calculate_freshness_score(self, candidate: Dict[str, Any]) -> float:
        """Calculate freshness score based on creation time."""
        # Placeholder implementation - would use actual timestamps
        # For now, return a neutral score
        return 0.5
    
    def _fallback_ranking(
        self,
        candidates: List[Dict[str, Any]],
        query_context: Dict[str, Any],
        max_k: int
    ) -> List[Dict[str, Any]]:
        """Fallback ranking using heuristic scores only."""
        logger.warning("Using fallback ranking due to cross-encoder failure")
        
        scored_candidates = []
        for candidate in candidates:
            # Use retrieval scores as base
            retrieval_scores = candidate.get("retriever_scores", {})
            base_score = max(retrieval_scores.values()) if retrieval_scores else 0.0
            
            # Add heuristic adjustments
            skill_score = self._calculate_skill_match(candidate, query_context)
            difficulty_score = self._calculate_difficulty_match(candidate, query_context)
            
            # Simple weighted combination
            fallback_score = 0.5 * base_score + 0.3 * skill_score + 0.2 * difficulty_score
            
            candidate_copy = candidate.copy()
            candidate_copy["rerank_score"] = fallback_score
            candidate_copy["fallback_ranking"] = True
            
            scored_candidates.append(candidate_copy)
        
        # Sort and return top-k
        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[:max_k]
    
    def _generate_cache_key(
        self, 
        query_text: str, 
        candidates: List[Dict[str, Any]]
    ) -> str:
        """Generate cache key for query and candidates."""
        # Create a hash of query and candidate IDs
        candidate_ids = [c.get("item_id", str(i)) for i, c in enumerate(candidates)]
        cache_input = {
            "query": query_text,
            "candidates": sorted(candidate_ids)  # Sort for consistency
        }
        
        cache_str = json.dumps(cache_input, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Get result from cache if available and not expired."""
        if cache_key not in self.cache:
            return None
        
        cached_item = self.cache[cache_key]
        if datetime.utcnow() - cached_item["timestamp"] > self.cache_ttl:
            # Expired, remove from cache
            del self.cache[cache_key]
            return None
        
        return cached_item["result"]
    
    def _add_to_cache(self, cache_key: str, result: List[Dict[str, Any]]):
        """Add result to cache."""
        # Clean cache if too large
        if len(self.cache) >= self.max_cache_size:
            self._clean_cache()
        
        self.cache[cache_key] = {
            "result": result,
            "timestamp": datetime.utcnow()
        }
    
    def _clean_cache(self):
        """Remove expired items from cache."""
        current_time = datetime.utcnow()
        expired_keys = [
            key for key, item in self.cache.items()
            if current_time - item["timestamp"] > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        # If still too large, remove oldest items
        if len(self.cache) >= self.max_cache_size:
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: x[1]["timestamp"]
            )
            
            # Keep only the newest half
            keep_count = self.max_cache_size // 2
            keys_to_remove = [item[0] for item in sorted_items[:-keep_count]]
            
            for key in keys_to_remove:
                del self.cache[key]
    
    def _update_latency_stats(self, latency_ms: float):
        """Update latency statistics."""
        current_avg = self.stats["avg_latency_ms"]
        total_requests = self.stats["total_requests"]
        
        # Running average
        self.stats["avg_latency_ms"] = (
            (current_avg * (total_requests - 1) + latency_ms) / total_requests
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        cache_hit_rate = (
            self.stats["cache_hits"] / self.stats["total_requests"]
            if self.stats["total_requests"] > 0 else 0.0
        )
        
        return {
            "model_loaded": self._model_loaded,
            "model_name": self.model_name,
            "device": self.device,
            "total_requests": self.stats["total_requests"],
            "cache_hits": self.stats["cache_hits"],
            "cache_hit_rate": cache_hit_rate,
            "avg_latency_ms": self.stats["avg_latency_ms"],
            "model_load_time": self.stats["model_load_time"],
            "cache_size": len(self.cache)
        }
    
    async def warmup(self):
        """Warm up the service by loading the model."""
        await self._load_model()
        logger.info("Re-ranking service warmed up successfully")


# Create service instance
reranking_service = RerankingService()