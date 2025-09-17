"""
Multi-Skill Elo Rating System for Math Recommendation
Adaptive learning with skill-specific ability tracking
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import structlog
import json
from datetime import datetime

logger = structlog.get_logger()

@dataclass
class SkillElo:
    """Elo rating for a specific skill"""
    skill_name: str
    rating: float = 1200.0  # Initial rating
    volatility: float = 0.06  # Rating volatility
    games_played: int = 0
    last_updated: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "rating": self.rating,
            "volatility": self.volatility,
            "games_played": self.games_played,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SkillElo':
        return cls(
            skill_name=data["skill_name"],
            rating=data["rating"],
            volatility=data["volatility"],
            games_played=data["games_played"],
            last_updated=data.get("last_updated")
        )

@dataclass
class ItemElo:
    """Elo rating for an item (question)"""
    item_id: str
    rating: float = 1200.0  # Initial rating
    volatility: float = 0.06
    attempts: int = 0
    correct_attempts: int = 0
    last_updated: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "rating": self.rating,
            "volatility": self.volatility,
            "attempts": self.attempts,
            "correct_attempts": self.correct_attempts,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ItemElo':
        return cls(
            item_id=data["item_id"],
            rating=data["rating"],
            volatility=data["volatility"],
            attempts=data["attempts"],
            correct_attempts=data["correct_attempts"],
            last_updated=data.get("last_updated")
        )

class MultiSkillElo:
    """Multi-skill Elo rating system"""
    
    def __init__(self, 
                 initial_rating: float = 1200.0,
                 k_factor_user: float = 16.0,
                 k_factor_item: float = 16.0,
                 skill_decay_factor: float = 0.95,
                 min_games_for_stable: int = 30):
        """
        Initialize multi-skill Elo system
        
        Args:
            initial_rating: Initial rating for new players/items
            k_factor_user: K-factor for user rating updates
            k_factor_item: K-factor for item rating updates
            skill_decay_factor: Factor for rating decay over time
            min_games_for_stable: Minimum games before considering rating stable
        """
        self.initial_rating = initial_rating
        self.k_factor_user = k_factor_user
        self.k_factor_item = k_factor_item
        self.skill_decay_factor = skill_decay_factor
        self.min_games_for_stable = min_games_for_stable
        
        # Storage
        self.user_skills: Dict[str, Dict[str, SkillElo]] = {}  # user_id -> skill_name -> SkillElo
        self.item_ratings: Dict[str, ItemElo] = {}  # item_id -> ItemElo
        self.skill_definitions: Dict[str, List[str]] = {}  # item_id -> list of required skills
        
        logger.info("Multi-skill Elo system initialized", 
                   initial_rating=initial_rating, k_user=k_factor_user, k_item=k_factor_item)
    
    def add_user_skill(self, user_id: str, skill_name: str, initial_rating: Optional[float] = None):
        """Add a skill for a user"""
        if user_id not in self.user_skills:
            self.user_skills[user_id] = {}
        
        rating = initial_rating or self.initial_rating
        self.user_skills[user_id][skill_name] = SkillElo(
            skill_name=skill_name,
            rating=rating,
            last_updated=datetime.now().isoformat()
        )
        
        logger.debug("Added user skill", user_id=user_id, skill=skill_name, rating=rating)
    
    def add_item(self, item_id: str, required_skills: List[str], initial_rating: Optional[float] = None):
        """Add an item with required skills"""
        rating = initial_rating or self.initial_rating
        self.item_ratings[item_id] = ItemElo(
            item_id=item_id,
            rating=rating,
            last_updated=datetime.now().isoformat()
        )
        self.skill_definitions[item_id] = required_skills
        
        logger.debug("Added item", item_id=item_id, skills=required_skills, rating=rating)
    
    def get_user_skill_rating(self, user_id: str, skill_name: str) -> float:
        """Get user's rating for a specific skill"""
        if user_id not in self.user_skills or skill_name not in self.user_skills[user_id]:
            # Return default rating if skill doesn't exist
            return self.initial_rating
        return self.user_skills[user_id][skill_name].rating
    
    def get_item_rating(self, item_id: str) -> float:
        """Get item's rating"""
        if item_id not in self.item_ratings:
            return self.initial_rating
        return self.item_ratings[item_id].rating
    
    def calculate_expected_score(self, user_id: str, item_id: str) -> float:
        """
        Calculate expected score for user-item interaction
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Expected score (probability of correct answer)
        """
        if item_id not in self.skill_definitions:
            return 0.5  # Default probability
        
        required_skills = self.skill_definitions[item_id]
        item_rating = self.get_item_rating(item_id)
        
        # Calculate weighted user ability
        user_ability = 0.0
        total_weight = 0.0
        
        for skill in required_skills:
            skill_rating = self.get_user_skill_rating(user_id, skill)
            # Weight by skill importance (could be made more sophisticated)
            weight = 1.0 / len(required_skills)  # Equal weights for now
            user_ability += skill_rating * weight
            total_weight += weight
        
        if total_weight > 0:
            user_ability /= total_weight
        else:
            user_ability = self.initial_rating
        
        # Expected score using logistic function
        rating_diff = user_ability - item_rating
        expected_score = 1.0 / (1.0 + np.exp(-rating_diff / 400.0))
        
        return expected_score
    
    def update_ratings(self, user_id: str, item_id: str, correct: bool, 
                      response_time: Optional[float] = None, 
                      skill_weights: Optional[Dict[str, float]] = None) -> Tuple[float, float]:
        """
        Update ratings after a response
        
        Args:
            user_id: User ID
            item_id: Item ID
            correct: Whether the answer was correct
            response_time: Time taken to answer (optional)
            skill_weights: Custom weights for skills (optional)
            
        Returns:
            Tuple of (expected_score, actual_score)
        """
        # Get current ratings
        item_rating = self.get_item_rating(item_id)
        expected_score = self.calculate_expected_score(user_id, item_id)
        actual_score = 1.0 if correct else 0.0
        
        # Calculate rating change
        score_diff = actual_score - expected_score
        
        # Update item rating
        if item_id in self.item_ratings:
            item_elo = self.item_ratings[item_id]
            
            # Adjust K-factor based on item volatility
            k_item = self.k_factor_item * (2 - item_elo.volatility)
            
            item_elo.rating += k_item * score_diff
            item_elo.attempts += 1
            if correct:
                item_elo.correct_attempts += 1
            item_elo.last_updated = datetime.now().isoformat()
            
            # Update volatility (simplified)
            item_elo.volatility = max(0.04, item_elo.volatility * 0.99)
        
        # Update user skill ratings
        if item_id in self.skill_definitions:
            required_skills = self.skill_definitions[item_id]
            
            # Use custom weights or equal weights
            if skill_weights:
                weights = {skill: skill_weights.get(skill, 1.0) for skill in required_skills}
            else:
                weights = {skill: 1.0 for skill in required_skills}
            
            total_weight = sum(weights.values())
            
            for skill in required_skills:
                # Ensure user has this skill
                if user_id not in self.user_skills:
                    self.user_skills[user_id] = {}
                if skill not in self.user_skills[user_id]:
                    self.add_user_skill(user_id, skill)
                
                skill_elo = self.user_skills[user_id][skill]
                
                # Adjust K-factor based on skill volatility and games played
                k_user = self.k_factor_user * (2 - skill_elo.volatility)
                if skill_elo.games_played < self.min_games_for_stable:
                    k_user *= 1.5  # Higher K-factor for new players
                
                # Weight the update by skill importance
                weight_factor = weights[skill] / total_weight
                
                skill_elo.rating += k_user * score_diff * weight_factor
                skill_elo.games_played += 1
                skill_elo.last_updated = datetime.now().isoformat()
                
                # Update volatility
                skill_elo.volatility = max(0.04, skill_elo.volatility * 0.99)
        
        logger.debug("Updated ratings", user_id=user_id, item_id=item_id, 
                    correct=correct, expected=expected_score, actual=actual_score)
        
        return expected_score, actual_score
    
    def get_user_skill_vector(self, user_id: str, skill_list: List[str]) -> np.ndarray:
        """Get user's skill ratings as a vector"""
        vector = []
        for skill in skill_list:
            rating = self.get_user_skill_rating(user_id, skill)
            vector.append(rating)
        return np.array(vector)
    
    def get_item_difficulty(self, item_id: str) -> float:
        """Get item difficulty (inverted rating for easier interpretation)"""
        rating = self.get_item_rating(item_id)
        # Convert rating to difficulty (higher rating = easier item)
        # Scale to 0-1 range where 0 is very hard, 1 is very easy
        difficulty = 1.0 / (1.0 + np.exp(-(rating - 1200) / 200.0))
        return difficulty
    
    def get_skill_mastery(self, user_id: str, skill_name: str) -> float:
        """Get skill mastery level (0-1 scale)"""
        rating = self.get_user_skill_rating(user_id, skill_name)
        # Convert rating to mastery level
        mastery = 1.0 / (1.0 + np.exp(-(rating - 1200) / 200.0))
        return mastery
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user statistics"""
        if user_id not in self.user_skills:
            return {"user_id": user_id, "skills": {}, "overall_rating": self.initial_rating}
        
        skills = {}
        total_rating = 0.0
        skill_count = 0
        
        for skill_name, skill_elo in self.user_skills[user_id].items():
            skills[skill_name] = {
                "rating": skill_elo.rating,
                "mastery": self.get_skill_mastery(user_id, skill_name),
                "games_played": skill_elo.games_played,
                "volatility": skill_elo.volatility
            }
            total_rating += skill_elo.rating
            skill_count += 1
        
        overall_rating = total_rating / skill_count if skill_count > 0 else self.initial_rating
        
        return {
            "user_id": user_id,
            "skills": skills,
            "overall_rating": overall_rating,
            "total_skills": skill_count
        }
    
    def get_item_stats(self, item_id: str) -> Dict[str, Any]:
        """Get comprehensive item statistics"""
        if item_id not in self.item_ratings:
            return {"item_id": item_id, "rating": self.initial_rating, "difficulty": 0.5}
        
        item_elo = self.item_ratings[item_id]
        
        return {
            "item_id": item_id,
            "rating": item_elo.rating,
            "difficulty": self.get_item_difficulty(item_id),
            "attempts": item_elo.attempts,
            "correct_attempts": item_elo.correct_attempts,
            "success_rate": item_elo.correct_attempts / max(item_elo.attempts, 1),
            "volatility": item_elo.volatility,
            "required_skills": self.skill_definitions.get(item_id, [])
        }
    
    def decay_ratings(self, days_since_last_activity: float = 30.0):
        """Apply rating decay for inactive users"""
        decay_factor = self.skill_decay_factor ** (days_since_last_activity / 30.0)
        
        for user_id, skills in self.user_skills.items():
            for skill_name, skill_elo in skills.items():
                # Decay towards initial rating
                skill_elo.rating = self.initial_rating + (skill_elo.rating - self.initial_rating) * decay_factor
        
        logger.info("Applied rating decay", factor=decay_factor)
    
    def save_system(self, filepath: str):
        """Save the entire Elo system to file"""
        system_data = {
            "config": {
                "initial_rating": self.initial_rating,
                "k_factor_user": self.k_factor_user,
                "k_factor_item": self.k_factor_item,
                "skill_decay_factor": self.skill_decay_factor,
                "min_games_for_stable": self.min_games_for_stable
            },
            "user_skills": {
                user_id: {
                    skill_name: skill.to_dict() 
                    for skill_name, skill in skills.items()
                }
                for user_id, skills in self.user_skills.items()
            },
            "item_ratings": {
                item_id: item.to_dict() 
                for item_id, item in self.item_ratings.items()
            },
            "skill_definitions": self.skill_definitions
        }
        
        with open(filepath, 'w') as f:
            json.dump(system_data, f, indent=2)
        
        logger.info("Elo system saved", filepath=filepath)
    
    def load_system(self, filepath: str):
        """Load the entire Elo system from file"""
        with open(filepath, 'r') as f:
            system_data = json.load(f)
        
        # Load config
        config = system_data["config"]
        self.initial_rating = config["initial_rating"]
        self.k_factor_user = config["k_factor_user"]
        self.k_factor_item = config["k_factor_item"]
        self.skill_decay_factor = config["skill_decay_factor"]
        self.min_games_for_stable = config["min_games_for_stable"]
        
        # Load user skills
        self.user_skills = {}
        for user_id, skills_data in system_data["user_skills"].items():
            self.user_skills[user_id] = {}
            for skill_name, skill_data in skills_data.items():
                self.user_skills[user_id][skill_name] = SkillElo.from_dict(skill_data)
        
        # Load item ratings
        self.item_ratings = {}
        for item_id, item_data in system_data["item_ratings"].items():
            self.item_ratings[item_id] = ItemElo.from_dict(item_data)
        
        # Load skill definitions
        self.skill_definitions = system_data["skill_definitions"]
        
        logger.info("Elo system loaded", filepath=filepath,
                   users=len(self.user_skills), items=len(self.item_ratings))

# Global multi-skill Elo instance
multiskill_elo = MultiSkillElo()
