"""
Recommendation orchestration service that coordinates all components.

This service implements the main recommendation pipeline:
retrieval → re-ranking → diversification → bandit selection
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

from app.services.retrieval_service import retrieval_service
from app.services.profile_service import profile_service
from app.services.bandit_service import bandit_service
from app.services.math_generation_service import math_generation_service
from app.services.reranking_service import reranking_service
from app.db.repositories.user import UserRepository
from app.db.repositories.item import EnglishItemRepository
from app.db.repositories.attempt import AttemptRepository
from app.core.config import settings

logger = logging.getLogger(__name__)


class RecommendationPipeline:
    """Main recommendation pipeline orchestrator."""
    
    def __init__(self):
        self.user_repo = UserRepository()
        self.item_repo = EnglishItemRepository()
        self.attempt_repo = AttemptRepository()
        
        # Pipeline configuration
        self.config = {
            "retrieval_k": 200,
            "rerank_k": 40,
            "diversification_k": 20,
            "final_k": 5,
            "generation_fallback": True,
            "cache_ttl": 300,  # 5 minutes
            "timeout_seconds": 10
        }
    
    async def recommend_next_questions(
        self,
        user_id: str,
        target_skills: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        personalization: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main recommendation pipeline entry point.
        
        Args:
            user_id: User identifier
            target_skills: Skills to focus on
            constraints: Filtering constraints
            personalization: Personalization parameters
            request_id: Request tracking ID
            
        Returns:
            Recommendation results with items and metadata
        """
        if not request_id:
            request_id = str(uuid.uuid4())
        
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Starting recommendation pipeline for user {user_id}, request {request_id}")
            
            # Step 1: Get user profile and context
            user_context = await self._get_user_context(user_id, target_skills)
            
            # Step 2: Retrieval phase
            retrieval_results = await self._retrieval_phase(
                user_context, constraints, request_id
            )
            
            # Step 3: Re-ranking phase (placeholder for now)
            reranked_results = await self._reranking_phase(
                retrieval_results, user_context, request_id
            )
            
            # Step 4: Diversification phase
            diversified_results = await self._diversification_phase(
                reranked_results, user_context, request_id
            )
            
            # Step 5: Bandit selection phase
            final_recommendations = await self._bandit_selection_phase(
                diversified_results, user_context, request_id
            )
            
            # Step 6: Generation fallback if needed
            if len(final_recommendations) < self.config["final_k"] and self.config["generation_fallback"]:
                final_recommendations = await self._generation_fallback(
                    final_recommendations, user_context, request_id
                )
            
            # Calculate pipeline metrics
            end_time = datetime.utcnow()
            pipeline_time = (end_time - start_time).total_seconds() * 1000
            
            # Calculate phase-specific metrics
            phase_metrics = self._calculate_phase_metrics(
                retrieval_results, reranked_results, 
                diversified_results, final_recommendations
            )
            
            result = {
                "items": final_recommendations,
                "metadata": {
                    "request_id": request_id,
                    "user_id": user_id,
                    "pipeline_time_ms": pipeline_time,
                    "retrieval_count": len(retrieval_results),
                    "reranked_count": len(reranked_results),
                    "diversified_count": len(diversified_results),
                    "final_count": len(final_recommendations),
                    "generated_at": end_time.isoformat(),
                    "policy_id": "main_pipeline_v1",
                    "bandit_version": "linucb_v1.3",
                    "phase_metrics": phase_metrics,
                    "success": True
                }
            }
            
            logger.info(f"Pipeline completed for request {request_id} in {pipeline_time:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed for request {request_id}: {e}")
            
            # Fallback to generation-only
            fallback_result = await self._emergency_fallback(user_id, request_id)
            return fallback_result
    
    async def _get_user_context(
        self,
        user_id: str,
        target_skills: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get comprehensive user context for recommendations."""
        try:
            # Get user profile
            profile = await profile_service.get_profile(user_id)
            
            # Get recent attempts for context
            recent_attempts = await self.attempt_repo.get_recent_attempts(
                user_id=user_id,
                limit=20
            )
            
            # Calculate skill gaps and preferences
            skill_analysis = self._analyze_skill_gaps(profile, recent_attempts)
            
            context = {
                "user_id": user_id,
                "profile": profile,
                "target_skills": target_skills or skill_analysis.get("priority_skills", []),
                "theta_math": profile.get("theta_math", 0.0),
                "theta_en": profile.get("theta_en", 0.0),
                "error_profiles": profile.get("error_profiles", {}),
                "recent_attempts": recent_attempts,
                "skill_gaps": skill_analysis.get("gaps", {}),
                "preferences": profile.get("preferences", {}),
                "language": profile.get("lang", "tr")
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get user context for {user_id}: {e}")
            # Return minimal context
            return {
                "user_id": user_id,
                "profile": {},
                "target_skills": target_skills or [],
                "theta_math": 0.0,
                "theta_en": 0.0,
                "error_profiles": {},
                "recent_attempts": [],
                "skill_gaps": {},
                "preferences": {},
                "language": "tr"
            }
    
    async def _retrieval_phase(
        self,
        user_context: Dict[str, Any],
        constraints: Optional[Dict[str, Any]],
        request_id: str
    ) -> List[Dict[str, Any]]:
        """Retrieval phase using hybrid search."""
        try:
            logger.debug(f"Starting retrieval phase for request {request_id}")
            
            # Build search query from user context
            search_query = self._build_search_query(user_context, constraints)
            
            # Add timeout for retrieval
            retrieval_timeout = self.config.get("timeout_seconds", 10)
            
            # Perform hybrid retrieval with timeout
            results = await asyncio.wait_for(
                retrieval_service.hybrid_search(
                    query=search_query.get("query"),
                    goals=search_query.get("goals"),
                    lang=user_context["language"],
                    k=self.config["retrieval_k"]
                ),
                timeout=retrieval_timeout
            )
            
            logger.debug(f"Retrieved {len(results)} candidates for request {request_id}")
            return results
            
        except asyncio.TimeoutError:
            logger.error(f"Retrieval phase timed out for request {request_id}")
            return []
        except Exception as e:
            logger.error(f"Retrieval phase failed for request {request_id}: {e}")
            return []
    
    async def _reranking_phase(
        self,
        candidates: List[Dict[str, Any]],
        user_context: Dict[str, Any],
        request_id: str
    ) -> List[Dict[str, Any]]:
        """Re-ranking phase using cross-encoder model."""
        try:
            logger.debug(f"Starting re-ranking phase for request {request_id}")
            
            if not candidates:
                return []
            
            # Build query context for re-ranking
            query_context = {
                "query": self._build_rerank_query(user_context),
                "target_skills": user_context.get("target_skills", []),
                "theta_math": user_context.get("theta_math", 0.0),
                "theta_en": user_context.get("theta_en", 0.0),
                "language": user_context.get("language", "tr"),
                "error_profiles": user_context.get("error_profiles", {})
            }
            
            # Add timeout for re-ranking
            rerank_timeout = self.config.get("timeout_seconds", 10)
            
            # Perform cross-encoder re-ranking with timeout
            reranked_candidates = await asyncio.wait_for(
                reranking_service.rerank_candidates(
                    query_context=query_context,
                    candidates=candidates,
                    max_k=self.config["rerank_k"],
                    use_cache=True
                ),
                timeout=rerank_timeout
            )
            
            logger.debug(f"Re-ranked {len(candidates)} to {len(reranked_candidates)} candidates for request {request_id}")
            return reranked_candidates
            
        except asyncio.TimeoutError:
            logger.error(f"Re-ranking phase timed out for request {request_id}")
            return self._fallback_reranking(candidates, user_context)
        except Exception as e:
            logger.error(f"Re-ranking phase failed for request {request_id}: {e}")
            return self._fallback_reranking(candidates, user_context)
    
    def _build_rerank_query(self, user_context: Dict[str, Any]) -> str:
        """Build query string for re-ranking."""
        query_parts = []
        
        # Add target skills
        target_skills = user_context.get("target_skills", [])
        if target_skills:
            query_parts.extend(target_skills)
        
        # Add difficulty level description
        theta_math = user_context.get("theta_math", 0.0)
        if theta_math > 0.5:
            query_parts.append("advanced mathematics")
        elif theta_math < -0.5:
            query_parts.append("basic mathematics")
        else:
            query_parts.append("intermediate mathematics")
        
        # Add error focus areas
        error_profiles = user_context.get("error_profiles", {})
        if error_profiles:
            for subject, errors in error_profiles.items():
                high_error_areas = [
                    error_type for error_type, rate in errors.items()
                    if rate > 0.3
                ]
                if high_error_areas:
                    query_parts.extend(high_error_areas[:2])  # Top 2 error areas
        
        return " ".join(query_parts) if query_parts else "mathematics question"
    
    def _fallback_reranking(
        self,
        candidates: List[Dict[str, Any]],
        user_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Fallback re-ranking using heuristic scoring."""
        logger.warning("Using fallback re-ranking due to cross-encoder failure")
        
        scored_candidates = []
        for candidate in candidates[:self.config["rerank_k"]]:
            score = self._calculate_heuristic_score(candidate, user_context)
            candidate["rerank_score"] = score
            candidate["fallback_reranking"] = True
            scored_candidates.append(candidate)
        
        # Sort by re-rank score
        scored_candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        return scored_candidates
    
    async def _diversification_phase(
        self,
        candidates: List[Dict[str, Any]],
        user_context: Dict[str, Any],
        request_id: str
    ) -> List[Dict[str, Any]]:
        """Diversification phase using MMR (Maximal Marginal Relevance)."""
        try:
            logger.debug(f"Starting diversification phase for request {request_id}")
            
            if not candidates:
                return []
            
            # MMR parameters
            lambda_param = 0.7  # Balance between relevance and diversity
            
            # Initialize with highest scoring candidate
            selected = [candidates[0]]
            remaining = candidates[1:]
            
            # MMR selection
            while len(selected) < self.config["diversification_k"] and remaining:
                best_candidate = None
                best_mmr_score = -float('inf')
                
                for candidate in remaining:
                    # Relevance score (from re-ranking)
                    relevance = candidate.get("rerank_score", 0)
                    
                    # Diversity score (minimum similarity to selected items)
                    diversity = self._calculate_diversity_score(candidate, selected)
                    
                    # MMR score
                    mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
                    
                    if mmr_score > best_mmr_score:
                        best_mmr_score = mmr_score
                        best_candidate = candidate
                
                if best_candidate:
                    selected.append(best_candidate)
                    remaining.remove(best_candidate)
                else:
                    break
            
            # Add diversity metadata
            for item in selected:
                item["diversity_score"] = self._calculate_diversity_score(item, selected)
            
            logger.debug(f"Diversified to {len(selected)} candidates for request {request_id}")
            return selected
            
        except Exception as e:
            logger.error(f"Diversification phase failed for request {request_id}: {e}")
            return candidates[:self.config["diversification_k"]]
    
    async def _bandit_selection_phase(
        self,
        candidates: List[Dict[str, Any]],
        user_context: Dict[str, Any],
        request_id: str
    ) -> List[Dict[str, Any]]:
        """Bandit selection phase using LinUCB."""
        try:
            logger.debug(f"Starting bandit selection phase for request {request_id}")
            
            if not candidates:
                logger.warning(f"No candidates for bandit selection in request {request_id}")
                return []
            
            # Prepare bandit context
            bandit_context = self._build_bandit_context(user_context)
            
            # Convert candidates to bandit arms
            arms = []
            for i, candidate in enumerate(candidates):
                try:
                    arm = {
                        "arm_id": f"item_{candidate.get('item_id', i)}",
                        "item_id": candidate.get("item_id"),
                        "features": self._extract_item_features(candidate, user_context),
                        "metadata": candidate
                    }
                    arms.append(arm)
                except Exception as e:
                    logger.warning(f"Failed to create arm for candidate {i}: {e}")
                    continue
            
            if not arms:
                logger.error(f"No valid arms created for request {request_id}")
                return self._create_fallback_recommendations(candidates, user_context)
            
            # Add timeout for bandit service
            bandit_timeout = self.config.get("timeout_seconds", 10)
            
            # Get bandit recommendations with timeout
            bandit_result = await asyncio.wait_for(
                bandit_service.select_arms(
                    context=bandit_context,
                    arms=arms,
                    k=self.config["final_k"]
                ),
                timeout=bandit_timeout
            )
            
            # Convert back to recommendation format
            recommendations = []
            selected_arms = bandit_result.get("selected_arms", [])
            
            for selection in selected_arms:
                try:
                    arm_metadata = selection.get("metadata", {})
                    
                    recommendation = {
                        "item_id": selection.get("item_id"),
                        "reason_tags": self._generate_reason_tags(arm_metadata, user_context),
                        "propensity": selection.get("propensity", 0.0),
                        "bandit_score": selection.get("score", 0.0),
                        "metadata": arm_metadata
                    }
                    recommendations.append(recommendation)
                except Exception as e:
                    logger.warning(f"Failed to process bandit selection: {e}")
                    continue
            
            logger.debug(f"Selected {len(recommendations)} items via bandit for request {request_id}")
            return recommendations
            
        except asyncio.TimeoutError:
            logger.error(f"Bandit selection timed out for request {request_id}")
            return self._create_fallback_recommendations(candidates, user_context)
        except Exception as e:
            logger.error(f"Bandit selection phase failed for request {request_id}: {e}")
            return self._create_fallback_recommendations(candidates, user_context)
    
    def _create_fallback_recommendations(
        self,
        candidates: List[Dict[str, Any]],
        user_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create fallback recommendations when bandit fails."""
        fallback_recommendations = []
        
        for candidate in candidates[:self.config["final_k"]]:
            try:
                recommendation = {
                    "item_id": candidate.get("item_id"),
                    "reason_tags": ["fallback", "system_error"],
                    "propensity": 0.5,
                    "bandit_score": candidate.get("rerank_score", 0.0),
                    "metadata": candidate
                }
                fallback_recommendations.append(recommendation)
            except Exception as e:
                logger.warning(f"Failed to create fallback recommendation: {e}")
                continue
        
        return fallback_recommendations
    
    async def _generation_fallback(
        self,
        current_recommendations: List[Dict[str, Any]],
        user_context: Dict[str, Any],
        request_id: str
    ) -> List[Dict[str, Any]]:
        """Generate new questions if not enough recommendations."""
        try:
            needed_count = self.config["final_k"] - len(current_recommendations)
            if needed_count <= 0:
                return current_recommendations
            
            logger.debug(f"Generating {needed_count} fallback questions for request {request_id}")
            
            # Determine what to generate based on user context
            target_skills = user_context.get("target_skills", [])
            language = user_context.get("language", "tr")
            
            # Generate math questions (English generation not implemented yet)
            generated_items = []
            
            # Choose appropriate templates based on skills and difficulty
            templates_to_use = self._select_generation_templates(user_context)
            
            for i in range(needed_count):
                template_id = templates_to_use[i % len(templates_to_use)]
                
                try:
                    # Generate question
                    question_data = math_generation_service.generate_question(
                        template_id=template_id,
                        target_difficulty=user_context.get("theta_math", 0.0),
                        language=language
                    )
                    
                    # Convert to recommendation format
                    generated_item = {
                        "item_id": None,  # Generated items don't have IDs
                        "generated_item": question_data,
                        "reason_tags": ["generated", "skill_gap", template_id],
                        "propensity": 0.8,  # High propensity for generated content
                        "bandit_score": 0.5,
                        "metadata": {
                            "type": "generated",
                            "template_id": template_id,
                            "skills": question_data["item"]["skills"]
                        }
                    }
                    generated_items.append(generated_item)
                    
                except Exception as e:
                    logger.error(f"Failed to generate question {i+1}: {e}")
                    continue
            
            # Combine with existing recommendations
            all_recommendations = current_recommendations + generated_items
            
            logger.debug(f"Added {len(generated_items)} generated items for request {request_id}")
            return all_recommendations
            
        except Exception as e:
            logger.error(f"Generation fallback failed for request {request_id}: {e}")
            return current_recommendations
    
    async def _emergency_fallback(
        self,
        user_id: str,
        request_id: str
    ) -> Dict[str, Any]:
        """Emergency fallback when entire pipeline fails."""
        try:
            logger.warning(f"Using emergency fallback for request {request_id}")
            
            # Generate a single basic math question
            question_data = math_generation_service.generate_question(
                template_id="linear_equation_v1",
                language="tr"
            )
            
            fallback_item = {
                "item_id": None,
                "generated_item": question_data,
                "reason_tags": ["emergency_fallback"],
                "propensity": 1.0,
                "bandit_score": 0.0,
                "metadata": {"type": "emergency_fallback"}
            }
            
            return {
                "items": [fallback_item],
                "metadata": {
                    "request_id": request_id,
                    "user_id": user_id,
                    "pipeline_time_ms": 0,
                    "retrieval_count": 0,
                    "reranked_count": 0,
                    "diversified_count": 0,
                    "final_count": 1,
                    "generated_at": datetime.utcnow().isoformat(),
                    "policy_id": "emergency_fallback",
                    "bandit_version": "none",
                    "fallback_reason": "pipeline_failure"
                }
            }
            
        except Exception as e:
            logger.error(f"Emergency fallback failed for request {request_id}: {e}")
            # Return empty result as last resort
            return {
                "items": [],
                "metadata": {
                    "request_id": request_id,
                    "user_id": user_id,
                    "error": "complete_failure",
                    "generated_at": datetime.utcnow().isoformat()
                }
            }
    
    def _build_search_query(
        self,
        user_context: Dict[str, Any],
        constraints: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build search query from user context and constraints."""
        target_skills = user_context.get("target_skills", [])
        language = user_context.get("language", "tr")
        
        # Build query string
        query_parts = []
        if target_skills:
            query_parts.extend(target_skills)
        
        # Add difficulty-related terms
        theta_math = user_context.get("theta_math", 0.0)
        if theta_math > 0.5:
            query_parts.append("advanced")
        elif theta_math < -0.5:
            query_parts.append("basic")
        else:
            query_parts.append("intermediate")
        
        query = " ".join(query_parts) if query_parts else "math"
        
        # Build goals
        goals = {
            "target_skills": target_skills,
            "difficulty_range": [theta_math - 0.3, theta_math + 0.3],
            "language": language
        }
        
        # Apply constraints
        if constraints:
            goals.update(constraints)
        
        return {"query": query, "goals": goals}
    
    def _calculate_heuristic_score(
        self,
        candidate: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> float:
        """Calculate heuristic relevance score for candidate."""
        score = 0.0
        
        # Base retrieval score
        retrieval_scores = candidate.get("retriever_scores", {})
        base_score = max(retrieval_scores.values()) if retrieval_scores else 0.0
        score += base_score * 0.4
        
        # Skill match score
        candidate_skills = candidate.get("metadata", {}).get("skills", [])
        target_skills = user_context.get("target_skills", [])
        
        if target_skills and candidate_skills:
            skill_overlap = len(set(candidate_skills) & set(target_skills))
            skill_score = skill_overlap / len(target_skills)
            score += skill_score * 0.3
        
        # Difficulty match score
        candidate_difficulty = candidate.get("metadata", {}).get("difficulty", 0.0)
        user_theta = user_context.get("theta_math", 0.0)
        
        difficulty_diff = abs(candidate_difficulty - user_theta)
        difficulty_score = max(0, 1.0 - difficulty_diff)
        score += difficulty_score * 0.2
        
        # Freshness score (prefer newer content)
        # TODO: Add timestamp-based freshness when available
        score += 0.1
        
        return score
    
    def _calculate_diversity_score(
        self,
        candidate: Dict[str, Any],
        selected_items: List[Dict[str, Any]]
    ) -> float:
        """Calculate diversity score (minimum similarity to selected items)."""
        if not selected_items:
            return 1.0
        
        candidate_skills = set(candidate.get("metadata", {}).get("skills", []))
        
        min_similarity = 1.0
        for selected in selected_items:
            selected_skills = set(selected.get("metadata", {}).get("skills", []))
            
            # Jaccard similarity
            if candidate_skills or selected_skills:
                intersection = len(candidate_skills & selected_skills)
                union = len(candidate_skills | selected_skills)
                similarity = intersection / union if union > 0 else 0.0
            else:
                similarity = 1.0  # Both empty
            
            min_similarity = min(min_similarity, similarity)
        
        # Diversity is inverse of similarity
        return 1.0 - min_similarity
    
    def _build_bandit_context(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Build context vector for bandit algorithm."""
        return {
            "user_id": user_context["user_id"],
            "theta_math": user_context.get("theta_math", 0.0),
            "theta_en": user_context.get("theta_en", 0.0),
            "recent_performance": self._calculate_recent_performance(
                user_context.get("recent_attempts", [])
            ),
            "skill_gaps": user_context.get("skill_gaps", {}),
            "session_context": {
                "time_of_day": datetime.utcnow().hour,
                "day_of_week": datetime.utcnow().weekday()
            }
        }
    
    def _extract_item_features(
        self,
        candidate: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract feature vector for bandit algorithm."""
        metadata = candidate.get("metadata", {})
        
        features = {
            "difficulty": metadata.get("difficulty", 0.0),
            "retrieval_score": max(candidate.get("retriever_scores", {}).values() or [0.0]),
            "rerank_score": candidate.get("rerank_score", 0.0),
            "skill_match": self._calculate_skill_match_score(candidate, user_context),
            "type_math": 1.0 if metadata.get("type") == "math" else 0.0,
            "type_english": 1.0 if metadata.get("type") == "english" else 0.0,
            "freshness": 0.5  # Placeholder
        }
        
        return features
    
    def _calculate_skill_match_score(
        self,
        candidate: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> float:
        """Calculate how well candidate matches target skills."""
        candidate_skills = set(candidate.get("metadata", {}).get("skills", []))
        target_skills = set(user_context.get("target_skills", []))
        
        if not target_skills:
            return 0.5  # Neutral if no target skills
        
        if not candidate_skills:
            return 0.0
        
        # Jaccard similarity
        intersection = len(candidate_skills & target_skills)
        union = len(candidate_skills | target_skills)
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_reason_tags(
        self,
        item_metadata: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> List[str]:
        """Generate reason tags explaining why item was recommended."""
        tags = []
        
        # Skill-based reasons
        item_skills = item_metadata.get("metadata", {}).get("skills", [])
        target_skills = user_context.get("target_skills", [])
        
        skill_overlap = set(item_skills) & set(target_skills)
        if skill_overlap:
            tags.extend([f"skill_{skill}" for skill in list(skill_overlap)[:2]])
        
        # Difficulty-based reasons
        item_difficulty = item_metadata.get("metadata", {}).get("difficulty", 0.0)
        user_theta = user_context.get("theta_math", 0.0)
        
        if abs(item_difficulty - user_theta) < 0.2:
            tags.append("optimal_difficulty")
        elif item_difficulty > user_theta:
            tags.append("challenge")
        else:
            tags.append("practice")
        
        # Algorithm-based reasons
        if item_metadata.get("rerank_score", 0) > 0.8:
            tags.append("high_relevance")
        
        if item_metadata.get("diversity_score", 0) > 0.7:
            tags.append("diverse_content")
        
        return tags[:4]  # Limit to 4 tags
    
    def _analyze_skill_gaps(
        self,
        profile: Dict[str, Any],
        recent_attempts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze skill gaps and priority areas."""
        # Placeholder implementation
        # TODO: Implement sophisticated skill gap analysis
        
        error_profiles = profile.get("error_profiles", {})
        
        # Find skills with high error rates
        priority_skills = []
        gaps = {}
        
        for subject, errors in error_profiles.items():
            for error_type, rate in errors.items():
                if rate > 0.3:  # High error rate threshold
                    priority_skills.append(error_type)
                    gaps[error_type] = rate
        
        # Default skills if no specific gaps identified
        if not priority_skills:
            priority_skills = ["linear_equation", "algebra"]
        
        return {
            "priority_skills": priority_skills[:3],  # Top 3 priority skills
            "gaps": gaps
        }
    
    def _calculate_recent_performance(self, recent_attempts: List[Dict[str, Any]]) -> float:
        """Calculate recent performance score."""
        if not recent_attempts:
            return 0.5  # Neutral performance
        
        correct_count = sum(1 for attempt in recent_attempts if attempt.get("correct", False))
        return correct_count / len(recent_attempts)
    
    def _select_generation_templates(self, user_context: Dict[str, Any]) -> List[str]:
        """Select appropriate templates for generation based on user context."""
        target_skills = user_context.get("target_skills", [])
        theta_math = user_context.get("theta_math", 0.0)
        
        # Map skills to templates
        skill_template_map = {
            "linear_equation": "linear_equation_v1",
            "algebra": "linear_equation_v1",
            "quadratic_equation": "quadratic_equation_v1",
            "factoring": "quadratic_equation_v1",
            "ratio": "ratio_proportion_v1",
            "proportion": "ratio_proportion_v1"
        }
        
        # Select templates based on skills
        templates = []
        for skill in target_skills:
            if skill in skill_template_map:
                template = skill_template_map[skill]
                if template not in templates:
                    templates.append(template)
        
        # Default templates if none selected
        if not templates:
            if theta_math > 0.2:
                templates = ["quadratic_equation_v1", "linear_equation_v1"]
            else:
                templates = ["linear_equation_v1", "ratio_proportion_v1"]
        
        return templates
    
    def _calculate_phase_metrics(
        self,
        retrieval_results: List[Dict[str, Any]],
        reranked_results: List[Dict[str, Any]],
        diversified_results: List[Dict[str, Any]],
        final_recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate metrics for each pipeline phase."""
        metrics = {}
        
        # Retrieval phase metrics
        if retrieval_results:
            retrieval_scores = []
            for result in retrieval_results:
                scores = result.get("retriever_scores", {})
                if scores:
                    retrieval_scores.append(max(scores.values()))
            
            if retrieval_scores:
                metrics["retrieval"] = {
                    "avg_score": sum(retrieval_scores) / len(retrieval_scores),
                    "max_score": max(retrieval_scores),
                    "min_score": min(retrieval_scores)
                }
        
        # Re-ranking phase metrics
        if reranked_results:
            rerank_scores = [r.get("rerank_score", 0.0) for r in reranked_results]
            if rerank_scores:
                metrics["reranking"] = {
                    "avg_score": sum(rerank_scores) / len(rerank_scores),
                    "max_score": max(rerank_scores),
                    "min_score": min(rerank_scores)
                }
        
        # Diversification metrics
        if diversified_results:
            diversity_scores = [r.get("diversity_score", 0.0) for r in diversified_results]
            if diversity_scores:
                metrics["diversification"] = {
                    "avg_diversity": sum(diversity_scores) / len(diversity_scores),
                    "max_diversity": max(diversity_scores),
                    "min_diversity": min(diversity_scores)
                }
        
        # Bandit metrics
        if final_recommendations:
            propensities = [r.get("propensity", 0.0) for r in final_recommendations]
            bandit_scores = [r.get("bandit_score", 0.0) for r in final_recommendations]
            
            if propensities:
                metrics["bandit"] = {
                    "avg_propensity": sum(propensities) / len(propensities),
                    "max_propensity": max(propensities),
                    "min_propensity": min(propensities)
                }
            
            if bandit_scores:
                metrics["bandit"]["avg_bandit_score"] = sum(bandit_scores) / len(bandit_scores)
        
        # Pipeline efficiency metrics
        metrics["efficiency"] = {
            "retrieval_to_rerank_ratio": len(reranked_results) / len(retrieval_results) if retrieval_results else 0,
            "rerank_to_diverse_ratio": len(diversified_results) / len(reranked_results) if reranked_results else 0,
            "diverse_to_final_ratio": len(final_recommendations) / len(diversified_results) if diversified_results else 0
        }
        
        return metrics


# Create service instance
orchestration_service = RecommendationPipeline()