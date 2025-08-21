from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import json
import asyncio
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import statistics

from app.core.cache import cache_service
from app.database import database_manager

logger = logging.getLogger(__name__)


class PersonalizationService:
    """Comprehensive personalization service with learning and adaptation"""
    
    def __init__(self):
        self.cache_ttl = 3600  # 1 hour
        self.learning_rate = 0.1
        self.decay_factor = 0.95
        self.min_confidence = 0.3
        
        # Preference weights
        self.implicit_weight = 0.7
        self.explicit_weight = 0.3
        self.recent_weight = 0.8
        self.historical_weight = 0.2
        
        # Learning settings
        self.min_interactions = 5
        self.max_history_days = 90
        self.adaptation_threshold = 0.6
        
        # Recommendation settings
        self.max_recommendations = 10
        self.diversity_factor = 0.3
        self.serendipity_factor = 0.1
    
    async def learn_user_preferences(
        self,
        user_id: str,
        interaction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Learn user preferences from interaction data"""
        
        try:
            # Extract interaction information
            question_id = interaction_data.get("question_id")
            subject = interaction_data.get("subject")
            topic = interaction_data.get("topic")
            difficulty_level = interaction_data.get("difficulty_level")
            question_type = interaction_data.get("question_type")
            user_answer = interaction_data.get("user_answer")
            correct_answer = interaction_data.get("correct_answer")
            time_spent = interaction_data.get("time_spent", 0)
            is_correct = user_answer == correct_answer
            
            # Update user interaction history
            await self._update_interaction_history(
                user_id, question_id, subject, topic, difficulty_level,
                question_type, is_correct, time_spent
            )
            
            # Learn implicit preferences
            implicit_preferences = await self._learn_implicit_preferences(
                user_id, subject, topic, difficulty_level, question_type,
                is_correct, time_spent
            )
            
            # Update explicit preferences if provided
            explicit_preferences = await self._update_explicit_preferences(
                user_id, interaction_data.get("explicit_feedback", {})
            )
            
            # Combine preferences
            combined_preferences = await self._combine_preferences(
                implicit_preferences, explicit_preferences
            )
            
            # Update user profile
            await self._update_user_profile(user_id, combined_preferences)
            
            # Generate learning insights
            insights = await self._generate_learning_insights(
                user_id, interaction_data, combined_preferences
            )
            
            return {
                "success": True,
                "preferences_updated": True,
                "implicit_preferences": implicit_preferences,
                "explicit_preferences": explicit_preferences,
                "combined_preferences": combined_preferences,
                "insights": insights,
                "learning_progress": await self._get_learning_progress(user_id, subject)
            }
            
        except Exception as e:
            logger.error(f"Error learning user preferences: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_personalized_recommendations(
        self,
        user_id: str,
        subject: str,
        topic: Optional[str] = None,
        limit: int = 10,
        include_diversity: bool = True,
        include_serendipity: bool = True
    ) -> Dict[str, Any]:
        """Get personalized question recommendations"""
        
        try:
            # Get user preferences
            user_preferences = await self._get_user_preferences(user_id, subject)
            
            # Get user's learning progress
            learning_progress = await self._get_learning_progress(user_id, subject)
            
            # Get recommended questions
            recommended_questions = await self._get_recommended_questions(
                user_id, subject, topic, user_preferences, learning_progress, limit
            )
            
            # Apply diversity if requested
            if include_diversity and len(recommended_questions) > 1:
                diversified_questions = await self._apply_diversity_filtering(
                    recommended_questions, user_preferences
                )
            else:
                diversified_questions = recommended_questions
            
            # Add serendipity if requested
            if include_serendipity and len(diversified_questions) < limit:
                serendipitous_questions = await self._get_serendipitous_questions(
                    user_id, subject, topic, diversified_questions, limit - len(diversified_questions)
                )
                diversified_questions.extend(serendipitous_questions)
            
            # Generate recommendation explanations
            explanations = await self._generate_recommendation_explanations(
                diversified_questions, user_preferences, learning_progress
            )
            
            return {
                "recommendations": diversified_questions,
                "explanations": explanations,
                "user_preferences": user_preferences,
                "learning_progress": learning_progress,
                "recommendation_metadata": {
                    "total_recommendations": len(diversified_questions),
                    "diversity_applied": include_diversity,
                    "serendipity_applied": include_serendipity,
                    "confidence_scores": [q.get("confidence", 0.0) for q in diversified_questions]
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting personalized recommendations: {e}")
            return {
                "recommendations": [],
                "error": str(e)
            }
    
    async def adapt_content_generation(
        self,
        user_id: str,
        subject: str,
        base_content: Dict[str, Any],
        adaptation_type: str = "difficulty"
    ) -> Dict[str, Any]:
        """Adapt content generation based on user preferences"""
        
        try:
            # Get user preferences and progress
            user_preferences = await self._get_user_preferences(user_id, subject)
            learning_progress = await self._get_learning_progress(user_id, subject)
            
            # Apply adaptations based on type
            if adaptation_type == "difficulty":
                adapted_content = await self._adapt_difficulty(
                    base_content, user_preferences, learning_progress
                )
            elif adaptation_type == "topic":
                adapted_content = await self._adapt_topic(
                    base_content, user_preferences, learning_progress
                )
            elif adaptation_type == "format":
                adapted_content = await self._adapt_format(
                    base_content, user_preferences, learning_progress
                )
            elif adaptation_type == "comprehensive":
                adapted_content = await self._adapt_comprehensive(
                    base_content, user_preferences, learning_progress
                )
            else:
                adapted_content = base_content
            
            # Add adaptation metadata
            adapted_content["adaptation_metadata"] = {
                "user_id": user_id,
                "subject": subject,
                "adaptation_type": adaptation_type,
                "original_content": base_content,
                "adaptation_factors": {
                    "user_preferences": user_preferences,
                    "learning_progress": learning_progress
                }
            }
            
            return adapted_content
            
        except Exception as e:
            logger.error(f"Error adapting content generation: {e}")
            return base_content
    
    async def get_learning_path_recommendations(
        self,
        user_id: str,
        subject: str,
        current_level: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get personalized learning path recommendations"""
        
        try:
            # Get user's current progress
            if not current_level:
                current_level = await self._get_current_level(user_id, subject)
            
            # Get learning progress
            learning_progress = await self._get_learning_progress(user_id, subject)
            
            # Generate learning path
            learning_path = await self._generate_learning_path(
                user_id, subject, current_level, learning_progress
            )
            
            # Get next steps
            next_steps = await self._get_next_learning_steps(
                user_id, subject, current_level, learning_progress
            )
            
            # Calculate progress metrics
            progress_metrics = await self._calculate_progress_metrics(
                user_id, subject, learning_progress
            )
            
            return {
                "current_level": current_level,
                "learning_path": learning_path,
                "next_steps": next_steps,
                "progress_metrics": progress_metrics,
                "recommendations": {
                    "focus_areas": await self._get_focus_areas(user_id, subject),
                    "practice_topics": await self._get_practice_topics(user_id, subject),
                    "challenge_areas": await self._get_challenge_areas(user_id, subject)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting learning path recommendations: {e}")
            return {
                "error": str(e),
                "current_level": current_level or 1
            }
    
    async def _update_interaction_history(
        self,
        user_id: str,
        question_id: str,
        subject: str,
        topic: str,
        difficulty_level: int,
        question_type: str,
        is_correct: bool,
        time_spent: float
    ) -> None:
        """Update user interaction history"""
        
        try:
            # Store interaction in database
            interaction_data = {
                "user_id": user_id,
                "question_id": question_id,
                "subject": subject,
                "topic": topic,
                "difficulty_level": difficulty_level,
                "question_type": question_type,
                "is_correct": is_correct,
                "time_spent": time_spent,
                "timestamp": datetime.now().isoformat()
            }
            
            # This would insert into user_interactions table
            # For now, cache the interaction
            cache_key = f"interaction:{user_id}:{question_id}"
            await cache_service.set(cache_key, interaction_data, self.cache_ttl)
            
        except Exception as e:
            logger.error(f"Error updating interaction history: {e}")
    
    async def _learn_implicit_preferences(
        self,
        user_id: str,
        subject: str,
        topic: str,
        difficulty_level: int,
        question_type: str,
        is_correct: bool,
        time_spent: float
    ) -> Dict[str, Any]:
        """Learn implicit preferences from user behavior"""
        
        try:
            # Get recent interactions
            recent_interactions = await self._get_recent_interactions(user_id, subject, days=30)
            
            # Analyze topic preferences
            topic_preferences = self._analyze_topic_preferences(recent_interactions, topic, is_correct)
            
            # Analyze difficulty preferences
            difficulty_preferences = self._analyze_difficulty_preferences(
                recent_interactions, difficulty_level, is_correct, time_spent
            )
            
            # Analyze format preferences
            format_preferences = self._analyze_format_preferences(
                recent_interactions, question_type, is_correct, time_spent
            )
            
            # Analyze learning patterns
            learning_patterns = self._analyze_learning_patterns(recent_interactions)
            
            return {
                "topic_preferences": topic_preferences,
                "difficulty_preferences": difficulty_preferences,
                "format_preferences": format_preferences,
                "learning_patterns": learning_patterns,
                "confidence": self._calculate_preference_confidence(recent_interactions)
            }
            
        except Exception as e:
            logger.error(f"Error learning implicit preferences: {e}")
            return {}
    
    def _analyze_topic_preferences(
        self,
        interactions: List[Dict[str, Any]],
        current_topic: str,
        is_correct: bool
    ) -> Dict[str, Any]:
        """Analyze user's topic preferences"""
        
        try:
            topic_performance = defaultdict(lambda: {"correct": 0, "total": 0, "avg_time": 0})
            
            for interaction in interactions:
                topic = interaction.get("topic", "")
                is_correct_interaction = interaction.get("is_correct", False)
                time_spent = interaction.get("time_spent", 0)
                
                topic_performance[topic]["total"] += 1
                if is_correct_interaction:
                    topic_performance[topic]["correct"] += 1
                topic_performance[topic]["avg_time"] += time_spent
            
            # Calculate performance metrics
            topic_preferences = {}
            for topic, stats in topic_performance.items():
                if stats["total"] >= self.min_interactions:
                    accuracy = stats["correct"] / stats["total"]
                    avg_time = stats["avg_time"] / stats["total"]
                    
                    # Preference score based on accuracy and engagement
                    preference_score = accuracy * 0.7 + (1 - min(avg_time / 300, 1.0)) * 0.3
                    
                    topic_preferences[topic] = {
                        "accuracy": accuracy,
                        "avg_time": avg_time,
                        "preference_score": preference_score,
                        "interaction_count": stats["total"]
                    }
            
            # Update current topic preference
            if current_topic in topic_preferences:
                current_pref = topic_preferences[current_topic]
                # Adjust based on current performance
                if is_correct:
                    current_pref["preference_score"] += self.learning_rate
                else:
                    current_pref["preference_score"] -= self.learning_rate * 0.5
                current_pref["preference_score"] = max(0.0, min(1.0, current_pref["preference_score"]))
            
            return topic_preferences
            
        except Exception as e:
            logger.error(f"Error analyzing topic preferences: {e}")
            return {}
    
    def _analyze_difficulty_preferences(
        self,
        interactions: List[Dict[str, Any]],
        current_difficulty: int,
        is_correct: bool,
        time_spent: float
    ) -> Dict[str, Any]:
        """Analyze user's difficulty preferences"""
        
        try:
            difficulty_performance = defaultdict(lambda: {"correct": 0, "total": 0, "avg_time": 0})
            
            for interaction in interactions:
                difficulty = interaction.get("difficulty_level", 3)
                is_correct_interaction = interaction.get("is_correct", False)
                time_spent_interaction = interaction.get("time_spent", 0)
                
                difficulty_performance[difficulty]["total"] += 1
                if is_correct_interaction:
                    difficulty_performance[difficulty]["correct"] += 1
                difficulty_performance[difficulty]["avg_time"] += time_spent_interaction
            
            # Calculate optimal difficulty range
            difficulty_preferences = {}
            for difficulty, stats in difficulty_performance.items():
                if stats["total"] >= self.min_interactions:
                    accuracy = stats["correct"] / stats["total"]
                    avg_time = stats["avg_time"] / stats["total"]
                    
                    # Optimal difficulty: high accuracy with reasonable time
                    optimal_score = accuracy * 0.8 + (1 - min(avg_time / 180, 1.0)) * 0.2
                    
                    difficulty_preferences[difficulty] = {
                        "accuracy": accuracy,
                        "avg_time": avg_time,
                        "optimal_score": optimal_score,
                        "interaction_count": stats["total"]
                    }
            
            # Find optimal difficulty range
            optimal_difficulties = []
            for difficulty, stats in difficulty_preferences.items():
                if stats["optimal_score"] >= self.adaptation_threshold:
                    optimal_difficulties.append(difficulty)
            
            return {
                "difficulty_preferences": difficulty_preferences,
                "optimal_difficulty_range": sorted(optimal_difficulties) if optimal_difficulties else [3],
                "current_difficulty_performance": {
                    "accuracy": difficulty_preferences.get(current_difficulty, {}).get("accuracy", 0.5),
                    "avg_time": difficulty_preferences.get(current_difficulty, {}).get("avg_time", 120)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing difficulty preferences: {e}")
            return {"optimal_difficulty_range": [3]}
    
    def _analyze_format_preferences(
        self,
        interactions: List[Dict[str, Any]],
        current_format: str,
        is_correct: bool,
        time_spent: float
    ) -> Dict[str, Any]:
        """Analyze user's question format preferences"""
        
        try:
            format_performance = defaultdict(lambda: {"correct": 0, "total": 0, "avg_time": 0})
            
            for interaction in interactions:
                format_type = interaction.get("question_type", "")
                is_correct_interaction = interaction.get("is_correct", False)
                time_spent_interaction = interaction.get("time_spent", 0)
                
                format_performance[format_type]["total"] += 1
                if is_correct_interaction:
                    format_performance[format_type]["correct"] += 1
                format_performance[format_type]["avg_time"] += time_spent_interaction
            
            # Calculate format preferences
            format_preferences = {}
            for format_type, stats in format_performance.items():
                if stats["total"] >= self.min_interactions:
                    accuracy = stats["correct"] / stats["total"]
                    avg_time = stats["avg_time"] / stats["total"]
                    
                    # Preference score based on accuracy and efficiency
                    preference_score = accuracy * 0.6 + (1 - min(avg_time / 240, 1.0)) * 0.4
                    
                    format_preferences[format_type] = {
                        "accuracy": accuracy,
                        "avg_time": avg_time,
                        "preference_score": preference_score,
                        "interaction_count": stats["total"]
                    }
            
            return format_preferences
            
        except Exception as e:
            logger.error(f"Error analyzing format preferences: {e}")
            return {}
    
    def _analyze_learning_patterns(
        self,
        interactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze user's learning patterns"""
        
        try:
            if not interactions:
                return {}
            
            # Time-based patterns
            time_patterns = defaultdict(int)
            for interaction in interactions:
                timestamp = interaction.get("timestamp")
                if timestamp:
                    hour = datetime.fromisoformat(timestamp).hour
                    time_patterns[hour] += 1
            
            # Performance trends
            recent_interactions = sorted(interactions, key=lambda x: x.get("timestamp", ""))[-20:]
            recent_accuracy = sum(1 for i in recent_interactions if i.get("is_correct", False)) / len(recent_interactions)
            
            # Learning speed
            avg_time = statistics.mean([i.get("time_spent", 0) for i in interactions])
            
            # Consistency
            accuracy_scores = [1 if i.get("is_correct", False) else 0 for i in interactions]
            consistency = 1 - statistics.stdev(accuracy_scores) if len(accuracy_scores) > 1 else 0
            
            return {
                "peak_learning_hours": sorted(time_patterns.items(), key=lambda x: x[1], reverse=True)[:3],
                "recent_accuracy": recent_accuracy,
                "avg_response_time": avg_time,
                "consistency": consistency,
                "learning_style": self._determine_learning_style(interactions)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing learning patterns: {e}")
            return {}
    
    def _determine_learning_style(self, interactions: List[Dict[str, Any]]) -> str:
        """Determine user's learning style based on interactions"""
        
        try:
            if not interactions:
                return "balanced"
            
            # Analyze response times
            response_times = [i.get("time_spent", 0) for i in interactions]
            avg_time = statistics.mean(response_times)
            
            # Analyze accuracy patterns
            accuracies = [1 if i.get("is_correct", False) else 0 for i in interactions]
            avg_accuracy = statistics.mean(accuracies)
            
            # Determine style
            if avg_time < 60 and avg_accuracy > 0.8:
                return "fast_learner"
            elif avg_time > 180 and avg_accuracy > 0.7:
                return "thorough_learner"
            elif avg_accuracy < 0.6:
                return "needs_support"
            else:
                return "balanced"
                
        except Exception as e:
            logger.error(f"Error determining learning style: {e}")
            return "balanced"
    
    async def _update_explicit_preferences(
        self,
        user_id: str,
        explicit_feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update explicit user preferences"""
        
        try:
            # This would update user_preferences table
            # For now, return the feedback as preferences
            return {
                "explicit_topics": explicit_feedback.get("preferred_topics", []),
                "explicit_difficulty": explicit_feedback.get("preferred_difficulty", 3),
                "explicit_formats": explicit_feedback.get("preferred_formats", []),
                "learning_goals": explicit_feedback.get("learning_goals", []),
                "feedback_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating explicit preferences: {e}")
            return {}
    
    async def _combine_preferences(
        self,
        implicit_preferences: Dict[str, Any],
        explicit_preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine implicit and explicit preferences"""
        
        try:
            combined = {
                "topic_preferences": {},
                "difficulty_preferences": {},
                "format_preferences": {},
                "learning_patterns": {},
                "explicit_preferences": explicit_preferences,
                "combined_confidence": 0.0
            }
            
            # Combine topic preferences
            implicit_topics = implicit_preferences.get("topic_preferences", {})
            explicit_topics = explicit_preferences.get("explicit_topics", [])
            
            for topic, pref in implicit_topics.items():
                combined["topic_preferences"][topic] = {
                    "implicit_score": pref["preference_score"],
                    "explicit_preference": topic in explicit_topics,
                    "combined_score": pref["preference_score"] * self.implicit_weight + 
                                    (1.0 if topic in explicit_topics else 0.0) * self.explicit_weight
                }
            
            # Combine difficulty preferences
            implicit_difficulty = implicit_preferences.get("difficulty_preferences", {})
            explicit_difficulty = explicit_preferences.get("explicit_difficulty", 3)
            
            combined["difficulty_preferences"] = {
                "implicit_preferences": implicit_difficulty,
                "explicit_preference": explicit_difficulty,
                "optimal_range": implicit_difficulty.get("optimal_difficulty_range", [3])
            }
            
            # Combine format preferences
            implicit_formats = implicit_preferences.get("format_preferences", {})
            explicit_formats = explicit_preferences.get("explicit_formats", [])
            
            for format_type, pref in implicit_formats.items():
                combined["format_preferences"][format_type] = {
                    "implicit_score": pref["preference_score"],
                    "explicit_preference": format_type in explicit_formats,
                    "combined_score": pref["preference_score"] * self.implicit_weight + 
                                    (1.0 if format_type in explicit_formats else 0.0) * self.explicit_weight
                }
            
            # Add learning patterns
            combined["learning_patterns"] = implicit_preferences.get("learning_patterns", {})
            
            # Calculate overall confidence
            confidence = implicit_preferences.get("confidence", 0.0)
            combined["combined_confidence"] = confidence
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining preferences: {e}")
            return {}
    
    async def _get_recent_interactions(
        self,
        user_id: str,
        subject: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get recent user interactions"""
        
        try:
            # This would query user_interactions table
            # For now, return empty list as placeholder
            return []
            
        except Exception as e:
            logger.error(f"Error getting recent interactions: {e}")
            return []
    
    def _calculate_preference_confidence(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate confidence in learned preferences"""
        
        try:
            if not interactions:
                return 0.0
            
            # Confidence based on number of interactions
            interaction_count = len(interactions)
            if interaction_count < self.min_interactions:
                return interaction_count / self.min_interactions
            
            # Confidence based on consistency
            accuracies = [1 if i.get("is_correct", False) else 0 for i in interactions]
            consistency = 1 - statistics.stdev(accuracies) if len(accuracies) > 1 else 0
            
            # Combined confidence
            confidence = min(1.0, interaction_count / 50) * 0.7 + consistency * 0.3
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating preference confidence: {e}")
            return 0.0
    
    async def _get_user_preferences(self, user_id: str, subject: str) -> Dict[str, Any]:
        """Get user preferences"""
        
        try:
            # This would query user_preferences table
            # For now, return default preferences
            return {
                "topic_preferences": {},
                "difficulty_preferences": {"optimal_difficulty_range": [3]},
                "format_preferences": {},
                "learning_patterns": {},
                "combined_confidence": 0.5
            }
            
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return {}
    
    async def _get_learning_progress(self, user_id: str, subject: str) -> Dict[str, Any]:
        """Get user's learning progress"""
        
        try:
            # This would calculate from user_interactions table
            # For now, return default progress
            return {
                "current_level": 3,
                "total_questions_answered": 0,
                "correct_answers": 0,
                "accuracy_rate": 0.0,
                "topics_covered": [],
                "weak_areas": [],
                "strong_areas": []
            }
            
        except Exception as e:
            logger.error(f"Error getting learning progress: {e}")
            return {"current_level": 1}
    
    async def get_personalization_statistics(self) -> Dict[str, Any]:
        """Get personalization service statistics"""
        
        return {
            "learning_rate": self.learning_rate,
            "decay_factor": self.decay_factor,
            "min_confidence": self.min_confidence,
            "preference_weights": {
                "implicit": self.implicit_weight,
                "explicit": self.explicit_weight,
                "recent": self.recent_weight,
                "historical": self.historical_weight
            },
            "learning_settings": {
                "min_interactions": self.min_interactions,
                "max_history_days": self.max_history_days,
                "adaptation_threshold": self.adaptation_threshold
            },
            "recommendation_settings": {
                "max_recommendations": self.max_recommendations,
                "diversity_factor": self.diversity_factor,
                "serendipity_factor": self.serendipity_factor
            }
        }


# Global instance
personalization_service = PersonalizationService()
