"""
Item Response Theory (IRT) Implementation for Math Recommendation System
Supports 1PL, 2PL, and 3PL models with MLE parameter estimation
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import structlog
from scipy.optimize import minimize
from scipy.stats import norm
import json

logger = structlog.get_logger()

@dataclass
class ItemParams:
    """Item parameters for IRT models"""
    item_id: str
    a: float = 1.0   # discrimination parameter
    b: float = 0.0   # difficulty parameter  
    c: float = 0.0   # guessing parameter (3PL only)
    skill_weights: Optional[Dict[str, float]] = None  # Multi-skill weights
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "item_id": self.item_id,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "skill_weights": self.skill_weights or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ItemParams':
        """Create from dictionary"""
        return cls(
            item_id=data["item_id"],
            a=data["a"],
            b=data["b"],
            c=data["c"],
            skill_weights=data.get("skill_weights", {})
        )

@dataclass
class StudentAbility:
    """Student ability parameters"""
    user_id: str
    theta: float = 0.0  # Overall ability
    skill_thetas: Dict[str, float] = None  # Per-skill abilities
    
    def __post_init__(self):
        if self.skill_thetas is None:
            self.skill_thetas = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "user_id": self.user_id,
            "theta": self.theta,
            "skill_thetas": self.skill_thetas
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StudentAbility':
        """Create from dictionary"""
        return cls(
            user_id=data["user_id"],
            theta=data["theta"],
            skill_thetas=data.get("skill_thetas", {})
        )

class IRTModel:
    """Item Response Theory model implementation"""
    
    def __init__(self, model_type: str = "2PL"):
        """
        Initialize IRT model
        
        Args:
            model_type: "1PL", "2PL", or "3PL"
        """
        self.model_type = model_type.upper()
        self.items: Dict[str, ItemParams] = {}
        self.students: Dict[str, StudentAbility] = {}
        self.responses: List[Dict[str, Any]] = []
        
        logger.info("IRT model initialized", model_type=self.model_type)
    
    def add_item(self, item: ItemParams):
        """Add item to the model"""
        self.items[item.item_id] = item
        logger.debug("Added item", item_id=item.item_id, params=item.to_dict())
    
    def add_student(self, student: StudentAbility):
        """Add student to the model"""
        self.students[student.user_id] = student
        logger.debug("Added student", user_id=student.user_id, theta=student.theta)
    
    def add_response(self, user_id: str, item_id: str, correct: bool, 
                    response_time: Optional[float] = None, skills: Optional[List[str]] = None):
        """Add response data"""
        response = {
            "user_id": user_id,
            "item_id": item_id,
            "correct": correct,
            "response_time": response_time,
            "skills": skills or [],
            "timestamp": np.datetime64('now')
        }
        self.responses.append(response)
        logger.debug("Added response", user_id=user_id, item_id=item_id, correct=correct)
    
    def p_correct(self, theta: float, item: ItemParams) -> float:
        """
        Calculate probability of correct response
        
        Args:
            theta: Student ability
            item: Item parameters
            
        Returns:
            Probability of correct response
        """
        if self.model_type == "1PL":
            # 1PL: P(θ) = 1 / (1 + exp(-(θ - b)))
            z = theta - item.b
            return 1.0 / (1.0 + np.exp(-z))
        
        elif self.model_type == "2PL":
            # 2PL: P(θ) = 1 / (1 + exp(-a(θ - b)))
            z = item.a * (theta - item.b)
            return 1.0 / (1.0 + np.exp(-z))
        
        elif self.model_type == "3PL":
            # 3PL: P(θ) = c + (1-c) / (1 + exp(-a(θ - b)))
            z = item.a * (theta - item.b)
            return item.c + (1 - item.c) / (1.0 + np.exp(-z))
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def p_correct_multi_skill(self, skill_thetas: Dict[str, float], item: ItemParams) -> float:
        """
        Calculate probability for multi-skill items
        
        Args:
            skill_thetas: Student abilities per skill
            item: Item parameters with skill weights
            
        Returns:
            Probability of correct response
        """
        if not item.skill_weights:
            # Fallback to overall ability
            return self.p_correct(skill_thetas.get("overall", 0.0), item)
        
        # Weighted ability score
        weighted_ability = sum(
            skill_thetas.get(skill, 0.0) * weight 
            for skill, weight in item.skill_weights.items()
        )
        
        return self.p_correct(weighted_ability, item)
    
    def estimate_ability(self, user_id: str, responses: List[Dict[str, Any]]) -> float:
        """
        Estimate student ability using MLE
        
        Args:
            user_id: Student ID
            responses: List of response data
            
        Returns:
            Estimated ability (theta)
        """
        def neg_log_likelihood(theta):
            """Negative log-likelihood for MLE"""
            ll = 0.0
            for resp in responses:
                if resp["item_id"] not in self.items:
                    continue
                
                item = self.items[resp["item_id"]]
                p = self.p_correct(theta, item)
                
                # Avoid log(0)
                p = max(min(p, 0.999), 0.001)
                
                if resp["correct"]:
                    ll += np.log(p)
                else:
                    ll += np.log(1 - p)
            
            return -ll
        
        # Initial guess
        initial_theta = 0.0
        
        # Optimize using scipy
        result = minimize(neg_log_likelihood, initial_theta, method='BFGS')
        
        if result.success:
            return float(result.x[0])
        else:
            logger.warning("Ability estimation failed", user_id=user_id)
            return 0.0
    
    def estimate_item_params(self, item_id: str, responses: List[Dict[str, Any]]) -> ItemParams:
        """
        Estimate item parameters using MLE
        
        Args:
            item_id: Item ID
            responses: List of response data for this item
            
        Returns:
            Estimated item parameters
        """
        def neg_log_likelihood(params):
            """Negative log-likelihood for item parameters"""
            if self.model_type == "1PL":
                b = params[0]
                a = 1.0
                c = 0.0
            elif self.model_type == "2PL":
                a, b = params
                c = 0.0
            elif self.model_type == "3PL":
                a, b, c = params
            
            # Ensure valid parameters
            a = max(0.1, a)
            c = max(0.0, min(0.5, c))
            
            ll = 0.0
            for resp in responses:
                theta = resp.get("theta", 0.0)
                
                if self.model_type == "1PL":
                    p = 1.0 / (1.0 + np.exp(-(theta - b)))
                elif self.model_type == "2PL":
                    z = a * (theta - b)
                    p = 1.0 / (1.0 + np.exp(-z))
                else:  # 3PL
                    z = a * (theta - b)
                    p = c + (1 - c) / (1.0 + np.exp(-z))
                
                # Avoid log(0)
                p = max(min(p, 0.999), 0.001)
                
                if resp["correct"]:
                    ll += np.log(p)
                else:
                    ll += np.log(1 - p)
            
            return -ll
        
        # Initial parameters
        if self.model_type == "1PL":
            initial_params = [0.0]
            bounds = [(-4, 4)]
        elif self.model_type == "2PL":
            initial_params = [1.0, 0.0]
            bounds = [(0.1, 3.0), (-4, 4)]
        else:  # 3PL
            initial_params = [1.0, 0.0, 0.0]
            bounds = [(0.1, 3.0), (-4, 4), (0.0, 0.5)]
        
        # Optimize
        result = minimize(neg_log_likelihood, initial_params, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            if self.model_type == "1PL":
                return ItemParams(item_id=item_id, a=1.0, b=result.x[0], c=0.0)
            elif self.model_type == "2PL":
                return ItemParams(item_id=item_id, a=result.x[0], b=result.x[1], c=0.0)
            else:  # 3PL
                return ItemParams(item_id=item_id, a=result.x[0], b=result.x[1], c=result.x[2])
        else:
            logger.warning("Item parameter estimation failed", item_id=item_id)
            # Return default parameters
            return ItemParams(item_id=item_id)
    
    def calibrate_model(self, max_iterations: int = 10, tolerance: float = 0.01):
        """
        Calibrate the entire IRT model using joint maximum likelihood estimation
        
        Args:
            max_iterations: Maximum number of calibration iterations
            tolerance: Convergence tolerance
        """
        logger.info("Starting IRT model calibration", iterations=max_iterations)
        
        for iteration in range(max_iterations):
            old_params = {item_id: item.to_dict() for item_id, item in self.items.items()}
            
            # Estimate abilities given current item parameters
            for user_id, student in self.students.items():
                user_responses = [r for r in self.responses if r["user_id"] == user_id]
                if user_responses:
                    new_theta = self.estimate_ability(user_id, user_responses)
                    student.theta = new_theta
            
            # Estimate item parameters given current abilities
            for item_id, item in self.items.items():
                item_responses = [r for r in self.responses if r["item_id"] == item_id]
                if item_responses:
                    # Add current theta estimates to responses
                    for resp in item_responses:
                        resp["theta"] = self.students[resp["user_id"]].theta
                    
                    new_item = self.estimate_item_params(item_id, item_responses)
                    self.items[item_id] = new_item
            
            # Check convergence
            new_params = {item_id: item.to_dict() for item_id, item in self.items.items()}
            
            max_change = 0.0
            for item_id in self.items:
                for param in ["a", "b", "c"]:
                    change = abs(new_params[item_id][param] - old_params[item_id][param])
                    max_change = max(max_change, change)
            
            logger.info("Calibration iteration", iteration=iteration+1, max_change=max_change)
            
            if max_change < tolerance:
                logger.info("IRT model calibration converged", iterations=iteration+1)
                break
        
        logger.info("IRT model calibration completed")
    
    def get_optimal_difficulty_range(self, theta: float, target_prob: float = 0.7) -> Tuple[float, float]:
        """
        Get optimal difficulty range for a student
        
        Args:
            theta: Student ability
            target_prob: Target probability of success
            
        Returns:
            Tuple of (min_difficulty, max_difficulty)
        """
        if self.model_type == "1PL":
            # For 1PL: b = theta - log(p/(1-p))
            logit = np.log(target_prob / (1 - target_prob))
            optimal_b = theta - logit
        else:
            # For 2PL/3PL: b = theta - log(p/(1-p)) / a
            # Use average discrimination
            avg_a = np.mean([item.a for item in self.items.values()]) if self.items else 1.0
            logit = np.log(target_prob / (1 - target_prob))
            optimal_b = theta - logit / avg_a
        
        # Range around optimal difficulty
        range_width = 0.5
        return (optimal_b - range_width, optimal_b + range_width)
    
    def save_model(self, filepath: str):
        """Save model to file"""
        model_data = {
            "model_type": self.model_type,
            "items": {item_id: item.to_dict() for item_id, item in self.items.items()},
            "students": {user_id: student.to_dict() for user_id, student in self.students.items()},
            "responses": self.responses
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2, default=str)
        
        logger.info("IRT model saved", filepath=filepath)
    
    def load_model(self, filepath: str):
        """Load model from file"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.model_type = model_data["model_type"]
        
        # Load items
        self.items = {}
        for item_id, item_data in model_data["items"].items():
            self.items[item_id] = ItemParams.from_dict(item_data)
        
        # Load students
        self.students = {}
        for user_id, student_data in model_data["students"].items():
            self.students[user_id] = StudentAbility.from_dict(student_data)
        
        # Load responses
        self.responses = model_data["responses"]
        
        logger.info("IRT model loaded", filepath=filepath, 
                   items=len(self.items), students=len(self.students), responses=len(self.responses))

# Global IRT model instance
irt_model = IRTModel(model_type="2PL")
