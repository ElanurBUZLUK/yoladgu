"""
Math Question Selector with IRT + Multi-Skill Elo Integration
Optimal question selection for maximizing learning gains
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import structlog

from .irt import IRTModel, ItemParams, StudentAbility, irt_model
from .multiskill_elo import MultiSkillElo, multiskill_elo

logger = structlog.get_logger()

@dataclass
class QuestionCandidate:
    """Candidate question for selection"""
    item_id: str
    item_params: ItemParams
    required_skills: List[str]
    skill_weights: Dict[str, float]
    elo_rating: float
    difficulty: float
    expected_probability: float
    learning_gain: float
    metadata: Dict[str, Any]

class MathQuestionSelector:
    """Intelligent question selector using IRT + Multi-Skill Elo"""
    
    def __init__(self, 
                 target_prob_low: float = 0.55,
                 target_prob_high: float = 0.8,
                 learning_gain_weight: float = 1.0,
                 difficulty_balance_weight: float = 0.3,
                 skill_diversity_weight: float = 0.2):
        """
        Initialize question selector
        
        Args:
            target_prob_low: Lower bound for target success probability
            target_prob_high: Upper bound for target success probability
            learning_gain_weight: Weight for learning gain in selection
            difficulty_balance_weight: Weight for difficulty balance
            skill_diversity_weight: Weight for skill diversity
        """
        self.target_prob_low = target_prob_low
        self.target_prob_high = target_prob_high
        self.learning_gain_weight = learning_gain_weight
        self.difficulty_balance_weight = difficulty_balance_weight
        self.skill_diversity_weight = skill_diversity_weight
        
        self.irt_model = irt_model
        self.elo_system = multiskill_elo
        
        logger.info("Math question selector initialized", 
                   target_prob=(target_prob_low, target_prob_high))
    
    def calculate_expected_probability(self, user_id: str, item_id: str, 
                                     item_params: ItemParams, 
                                     required_skills: List[str]) -> float:
        """
        Calculate expected probability of correct answer using both IRT and Elo
        
        Args:
            user_id: User ID
            item_id: Item ID
            item_params: IRT item parameters
            required_skills: List of required skills
            
        Returns:
            Expected probability of correct answer
        """
        # Get IRT probability
        if user_id in self.irt_model.students:
            student = self.irt_model.students[user_id]
            
            # Multi-skill IRT calculation
            if item_params.skill_weights and student.skill_thetas:
                # Use skill-specific thetas
                weighted_theta = sum(
                    student.skill_thetas.get(skill, student.theta) * weight
                    for skill, weight in item_params.skill_weights.items()
                )
                irt_prob = self.irt_model.p_correct(weighted_theta, item_params)
            else:
                # Use overall theta
                irt_prob = self.irt_model.p_correct(student.theta, item_params)
        else:
            # Default probability if no IRT data
            irt_prob = 0.5
        
        # Get Elo probability
        elo_prob = self.elo_system.calculate_expected_score(user_id, item_id)
        
        # Weighted combination (can be tuned based on data availability)
        irt_weight = 0.7 if user_id in self.irt_model.students else 0.0
        elo_weight = 1.0 - irt_weight
        
        combined_prob = irt_weight * irt_prob + elo_weight * elo_prob
        
        return combined_prob
    
    def calculate_learning_gain(self, user_id: str, item_params: ItemParams, 
                              required_skills: List[str], skill_weights: Dict[str, float]) -> float:
        """
        Calculate expected learning gain from attempting this question
        
        Args:
            user_id: User ID
            item_params: IRT item parameters
            required_skills: List of required skills
            skill_weights: Skill importance weights
            
        Returns:
            Expected learning gain
        """
        # Get current user abilities
        if user_id in self.irt_model.students:
            student = self.irt_model.students[user_id]
            current_theta = student.theta
        else:
            current_theta = 0.0
        
        # Calculate current probability
        current_prob = self.irt_model.p_correct(current_theta, item_params)
        
        # Learning gain is proportional to uncertainty (p * (1-p))
        # This represents the information gain from the question
        uncertainty = current_prob * (1 - current_prob)
        
        # Weight by skill importance
        skill_importance = sum(skill_weights.values()) / len(skill_weights) if skill_weights else 1.0
        
        # Weight by skill diversity (more skills = more learning potential)
        skill_diversity = len(required_skills) / 5.0  # Normalize to max 5 skills
        
        learning_gain = uncertainty * skill_importance * (1 + skill_diversity * 0.5)
        
        return learning_gain
    
    def calculate_skill_diversity_score(self, user_id: str, required_skills: List[str]) -> float:
        """
        Calculate skill diversity score for balanced learning
        
        Args:
            user_id: User ID
            required_skills: List of required skills
            
        Returns:
            Diversity score (higher = more diverse)
        """
        if user_id not in self.elo_system.user_skills:
            return 1.0  # Maximum diversity for new users
        
        user_skills = self.elo_system.user_skills[user_id]
        
        # Calculate skill mastery variance
        skill_masteries = []
        for skill in required_skills:
            if skill in user_skills:
                mastery = self.elo_system.get_skill_mastery(user_id, skill)
                skill_masteries.append(mastery)
            else:
                skill_masteries.append(0.5)  # Default for unknown skills
        
        if len(skill_masteries) <= 1:
            return 1.0
        
        # Higher variance = more diverse skill levels = better for learning
        variance = np.var(skill_masteries)
        diversity_score = min(1.0, variance * 4)  # Scale to 0-1
        
        return diversity_score
    
    def filter_questions_by_difficulty(self, questions: List[Dict[str, Any]], 
                                     user_id: str) -> List[Dict[str, Any]]:
        """
        Filter questions by target difficulty range
        
        Args:
            questions: List of question data
            user_id: User ID
            
        Returns:
            Filtered questions within target difficulty range
        """
        filtered_questions = []
        
        for question in questions:
            item_id = question["id"]
            item_params = ItemParams.from_dict(question.get("irt_params", {}))
            required_skills = question.get("required_skills", [])
            
            # Calculate expected probability
            expected_prob = self.calculate_expected_probability(
                user_id, item_id, item_params, required_skills
            )
            
            # Check if within target range
            if self.target_prob_low <= expected_prob <= self.target_prob_high:
                filtered_questions.append(question)
        
        # If no questions in target range, expand range gradually
        if not filtered_questions:
            logger.warning("No questions in target difficulty range, expanding range")
            
            for question in questions:
                item_id = question["id"]
                item_params = ItemParams.from_dict(question.get("irt_params", {}))
                required_skills = question.get("required_skills", [])
                
                expected_prob = self.calculate_expected_probability(
                    user_id, item_id, item_params, required_skills
                )
                
                # Expanded range
                if 0.3 <= expected_prob <= 0.9:
                    filtered_questions.append(question)
        
        return filtered_questions
    
    def select_next_question(self, user_id: str, available_questions: List[Dict[str, Any]], 
                           recent_questions: Optional[List[str]] = None,
                           avoid_recent: bool = True) -> Optional[Dict[str, Any]]:
        """
        Select the next optimal question for a user
        
        Args:
            user_id: User ID
            available_questions: List of available questions
            recent_questions: List of recently attempted question IDs
            avoid_recent: Whether to avoid recently attempted questions
            
        Returns:
            Selected question data or None if no suitable question found
        """
        if not available_questions:
            logger.warning("No available questions for selection")
            return None
        
        # Filter out recent questions if requested
        if avoid_recent and recent_questions:
            available_questions = [
                q for q in available_questions 
                if q["id"] not in recent_questions
            ]
        
        if not available_questions:
            logger.warning("No questions available after filtering recent questions")
            return None
        
        # Filter by difficulty range
        filtered_questions = self.filter_questions_by_difficulty(available_questions, user_id)
        
        if not filtered_questions:
            logger.warning("No questions in suitable difficulty range")
            return None
        
        # Score all candidate questions
        candidates = []
        
        for question in filtered_questions:
            item_id = question["id"]
            item_params = ItemParams.from_dict(question.get("irt_params", {}))
            required_skills = question.get("required_skills", [])
            skill_weights = question.get("skill_weights", {})
            
            # Get Elo rating
            elo_rating = self.elo_system.get_item_rating(item_id)
            difficulty = self.elo_system.get_item_difficulty(item_id)
            
            # Calculate metrics
            expected_prob = self.calculate_expected_probability(
                user_id, item_id, item_params, required_skills
            )
            
            learning_gain = self.calculate_learning_gain(
                user_id, item_params, required_skills, skill_weights
            )
            
            skill_diversity = self.calculate_skill_diversity_score(user_id, required_skills)
            
            # Create candidate
            candidate = QuestionCandidate(
                item_id=item_id,
                item_params=item_params,
                required_skills=required_skills,
                skill_weights=skill_weights,
                elo_rating=elo_rating,
                difficulty=difficulty,
                expected_probability=expected_prob,
                learning_gain=learning_gain,
                metadata=question
            )
            
            candidates.append(candidate)
        
        # Score and rank candidates
        scored_candidates = []
        
        for candidate in candidates:
            # Base score from learning gain
            score = candidate.learning_gain * self.learning_gain_weight
            
            # Add skill diversity bonus
            score += skill_diversity * self.skill_diversity_weight
            
            # Add difficulty balance (prefer questions that balance user's skill profile)
            if user_id in self.elo_system.user_skills:
                user_skills = self.elo_system.user_skills[user_id]
                
                # Calculate skill balance score
                skill_ratings = []
                for skill in candidate.required_skills:
                    if skill in user_skills:
                        skill_ratings.append(user_skills[skill].rating)
                    else:
                        skill_ratings.append(self.elo_system.initial_rating)
                
                if skill_ratings:
                    skill_balance = 1.0 - (np.std(skill_ratings) / np.mean(skill_ratings))
                    score += skill_balance * self.difficulty_balance_weight
            
            scored_candidates.append((candidate, score))
        
        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        if scored_candidates:
            best_candidate = scored_candidates[0][0]
            
            logger.info("Selected question", 
                       user_id=user_id, 
                       item_id=best_candidate.item_id,
                       expected_prob=best_candidate.expected_probability,
                       learning_gain=best_candidate.learning_gain,
                       score=scored_candidates[0][1])
            
            return best_candidate.metadata
        
        return None
    
    def get_adaptive_difficulty_range(self, user_id: str) -> Tuple[float, float]:
        """
        Get adaptive difficulty range based on user performance
        
        Args:
            user_id: User ID
            
        Returns:
            Tuple of (min_probability, max_probability)
        """
        # Base range
        min_prob = self.target_prob_low
        max_prob = self.target_prob_high
        
        # Adjust based on user performance history
        if user_id in self.irt_model.students:
            student = self.irt_model.students[user_id]
            
            # Get recent performance (simplified)
            recent_responses = [r for r in self.irt_model.responses if r["user_id"] == user_id]
            
            if len(recent_responses) >= 5:
                recent_correct = sum(1 for r in recent_responses[-10:] if r["correct"])
                recent_accuracy = recent_correct / min(10, len(recent_responses))
                
                # Adjust range based on performance
                if recent_accuracy > 0.8:
                    # High performer - increase difficulty
                    min_prob -= 0.05
                    max_prob -= 0.05
                elif recent_accuracy < 0.5:
                    # Low performer - decrease difficulty
                    min_prob += 0.05
                    max_prob += 0.05
        
        # Ensure valid range
        min_prob = max(0.3, min_prob)
        max_prob = min(0.9, max_prob)
        
        return (min_prob, max_prob)
    
    def get_user_recommendation_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive recommendation summary for a user
        
        Args:
            user_id: User ID
            
        Returns:
            Recommendation summary
        """
        # Get difficulty range
        difficulty_range = self.get_adaptive_difficulty_range(user_id)
        
        # Get user stats
        elo_stats = self.elo_system.get_user_stats(user_id)
        
        # Get IRT stats
        irt_stats = {}
        if user_id in self.irt_model.students:
            student = self.irt_model.students[user_id]
            irt_stats = {
                "theta": student.theta,
                "skill_thetas": student.skill_thetas
            }
        
        return {
            "user_id": user_id,
            "difficulty_range": difficulty_range,
            "elo_stats": elo_stats,
            "irt_stats": irt_stats,
            "recommendation_strategy": {
                "target_probability_range": (self.target_prob_low, self.target_prob_high),
                "learning_gain_weight": self.learning_gain_weight,
                "skill_diversity_weight": self.skill_diversity_weight
            }
        }

# Global question selector instance
math_selector = MathQuestionSelector()
