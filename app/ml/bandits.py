import numpy as np
import pickle
import json
import structlog
from typing import List, Dict, Tuple, Optional
from scipy.linalg import solve
import time

logger = structlog.get_logger()

class EnhancedLinUCBBandit:
    """Gelişmiş LinUCB (Contextual Bandit) algoritması"""
    
    def __init__(self, redis_client, alpha: float = 1.0, feature_dim: int = 15, 
                 regularization: float = 1.0):
        self.redis = redis_client
        self.alpha = alpha  # Exploration parameter
        self.feature_dim = feature_dim
        self.regularization = regularization
        
        # Redis keys
        self.A_key = "bandit:A_matrices"
        self.b_key = "bandit:b_vectors"
        self.stats_key = "bandit:statistics"
        
        # Model state
        self.A = {}  # Covariance matrices for each arm/question
        self.b = {}  # Reward vectors for each arm/question
        self.stats = {
            'total_selections': 0,
            'total_rewards': 0,
            'avg_reward': 0.0,
            'exploration_ratio': 0.0
        }
        
        # Load from Redis
        self._load_state()

    def select_question(self, user_features: Dict, candidate_questions: List[Dict], 
                       context: Optional[Dict] = None) -> Tuple[int, Dict]:
        """
        En iyi soruyu seç (exploration vs exploitation)
        Returns: (question_id, selection_info)
        """
        if not candidate_questions:
            raise ValueError("No candidate questions provided")
        
        ucb_values = {}
        selection_details = {}
        
        try:
            for question in candidate_questions:
                q_id = question['id']
                x = self._build_context_vector(user_features, question, context)
                
                # Initialize matrices if not exists
                if q_id not in self.A:
                    self._initialize_arm(q_id)
                
                # Calculate UCB value
                ucb_value, confidence, expected_reward = self._calculate_ucb(q_id, x)
                ucb_values[q_id] = ucb_value
                
                selection_details[q_id] = {
                    'expected_reward': expected_reward,
                    'confidence': confidence,
                    'ucb_value': ucb_value,
                    'context_norm': np.linalg.norm(x)
                }
            
            # Select question with highest UCB value
            selected_id = max(ucb_values, key=ucb_values.get)
            
            # Update statistics
            self.stats['total_selections'] += 1
            exploration_bonus = selection_details[selected_id]['confidence']
            expected_reward = selection_details[selected_id]['expected_reward']
            is_exploration = exploration_bonus > expected_reward
            
            if is_exploration:
                self.stats['exploration_count'] = self.stats.get('exploration_count', 0) + 1
            
            self.stats['exploration_ratio'] = (
                self.stats.get('exploration_count', 0) / self.stats['total_selections']
            )
            
            logger.info("bandit_question_selected",
                       question_id=selected_id,
                       ucb_value=ucb_values[selected_id],
                       is_exploration=is_exploration,
                       total_selections=self.stats['total_selections'])
            
            return selected_id, selection_details[selected_id]
            
        except Exception as e:
            logger.error("bandit_selection_error", error=str(e))
            # Fallback: random selection
            return candidate_questions[0]['id'], {"error": "fallback_selection"}

    def update(self, question_id: int, user_features: Dict, question_features: Dict, 
              reward: float, context: Optional[Dict] = None):
        """Bandit modelini cevaba göre güncelle"""
        try:
            x = self._build_context_vector(user_features, question_features, context)
            
            # Initialize if needed
            if question_id not in self.A:
                self._initialize_arm(question_id)
            
            # Update matrices
            self.A[question_id] += np.outer(x, x)
            self.b[question_id] += reward * x
            
            # Update statistics
            self.stats['total_rewards'] += reward
            if self.stats['total_selections'] > 0:
                self.stats['avg_reward'] = self.stats['total_rewards'] / self.stats['total_selections']
            
            # Save state periodically
            if self.stats['total_selections'] % 10 == 0:
                self._save_state()
            
            logger.info("bandit_updated",
                       question_id=question_id,
                       reward=reward,
                       avg_reward=self.stats['avg_reward'])
            
        except Exception as e:
            logger.error("bandit_update_error", error=str(e))

    def _calculate_ucb(self, question_id: int, context_vector: np.ndarray) -> Tuple[float, float, float]:
        """UCB değerini hesapla"""
        try:
            A_inv = np.linalg.inv(self.A[question_id])
            theta = A_inv.dot(self.b[question_id])
            
            # Expected reward
            expected_reward = theta.dot(context_vector)
            
            # Confidence interval
            confidence = self.alpha * np.sqrt(context_vector.T.dot(A_inv).dot(context_vector))
            
            # UCB value
            ucb_value = expected_reward + confidence
            
            return float(ucb_value), float(confidence), float(expected_reward)
            
        except np.linalg.LinAlgError:
            # Matrix inversion failed, use regularized version
            A_reg = self.A[question_id] + self.regularization * np.identity(self.feature_dim)
            theta = solve(A_reg, self.b[question_id])
            expected_reward = theta.dot(context_vector)
            confidence = self.alpha * np.sqrt(context_vector.T.dot(np.linalg.inv(A_reg)).dot(context_vector))
            return float(expected_reward + confidence), float(confidence), float(expected_reward)

    def _build_context_vector(self, user_features: Dict, question_features: Dict,
                            context: Optional[Dict] = None) -> np.ndarray:
        """Context vektörü oluştur - daha zengin feature set"""
        features = []
        
        # User features (normalized)
        features.extend([
            user_features.get("accuracy_rate_overall", 0.5),
            user_features.get("accuracy_rate_last_10", 0.5),
            min(user_features.get("avg_response_time", 5000) / 10000, 1.0),
            user_features.get("topic_mastery_math", 0.5),
            min(user_features.get("consecutive_correct", 0) / 10, 1.0),
            min(user_features.get("session_question_count", 0) / 20, 1.0),
        ])
        
        # Question features (normalized)
        features.extend([
            question_features.get("difficulty_level", 3) / 5,
            question_features.get("avg_success_rate", 0.5),
            min(question_features.get("avg_response_time", 5000) / 10000, 1.0),
            min(question_features.get("attempt_count", 0) / 100, 1.0),
        ])
        
        # Interaction features
        user_skill = user_features.get("topic_mastery_math", 0.5)
        question_difficulty = question_features.get("difficulty_level", 3) / 5
        features.extend([
            user_skill * question_difficulty,  # Skill-difficulty interaction
            abs(user_skill - question_difficulty),  # Skill-difficulty gap
            user_features.get("accuracy_rate_overall", 0.5) * question_features.get("avg_success_rate", 0.5),
        ])
        
        # Context features (if provided)
        if context:
            features.extend([
                context.get("time_of_day", 12) / 24,  # Time of day effect
                context.get("session_progress", 0.5),  # How far in session
            ])
        else:
            features.extend([0.5, 0.5])  # Default context
        
        # Pad or truncate to feature_dim
        if len(features) < self.feature_dim:
            features.extend([0.0] * (self.feature_dim - len(features)))
        elif len(features) > self.feature_dim:
            features = features[:self.feature_dim]
        
        return np.array(features)

    def _initialize_arm(self, question_id: int):
        """Yeni bir arm (soru) için matrisleri başlat"""
        self.A[question_id] = np.identity(self.feature_dim) * self.regularization
        self.b[question_id] = np.zeros(self.feature_dim)

    def _save_state(self):
        """Bandit state'ini Redis'e kaydet"""
        try:
            # Save A matrices
            A_serialized = {str(k): v.tolist() for k, v in self.A.items()}
            self.redis.setex(self.A_key, 86400, json.dumps(A_serialized))
            
            # Save b vectors
            b_serialized = {str(k): v.tolist() for k, v in self.b.items()}
            self.redis.setex(self.b_key, 86400, json.dumps(b_serialized))
            
            # Save statistics
            self.redis.setex(self.stats_key, 86400, json.dumps(self.stats))
            
            logger.info("bandit_state_saved", arms_count=len(self.A))
            
        except Exception as e:
            logger.error("bandit_save_error", error=str(e))

    def _load_state(self):
        """Bandit state'ini Redis'ten yükle"""
        try:
            # Load A matrices
            A_data = self.redis.get(self.A_key)
            if A_data:
                A_serialized = json.loads(A_data)
                self.A = {int(k): np.array(v) for k, v in A_serialized.items()}
            
            # Load b vectors
            b_data = self.redis.get(self.b_key)
            if b_data:
                b_serialized = json.loads(b_data)
                self.b = {int(k): np.array(v) for k, v in b_serialized.items()}
            
            # Load statistics
            stats_data = self.redis.get(self.stats_key)
            if stats_data:
                self.stats.update(json.loads(stats_data))
            
            logger.info("bandit_state_loaded", arms_count=len(self.A))
            
        except Exception as e:
            logger.info("bandit_load_failed", error=str(e))

    def get_statistics(self) -> Dict:
        """Bandit istatistiklerini al"""
        stats = self.stats.copy()
        stats['arms_count'] = len(self.A)
        stats['alpha'] = self.alpha
        stats['feature_dim'] = self.feature_dim
        return stats

    def reset(self):
        """Bandit'i sıfırla"""
        try:
            self.A.clear()
            self.b.clear()
            self.stats = {
                'total_selections': 0,
                'total_rewards': 0,
                'avg_reward': 0.0,
                'exploration_ratio': 0.0
            }
            
            # Clear Redis
            self.redis.delete(self.A_key, self.b_key, self.stats_key)
            logger.info("bandit_reset")
            
        except Exception as e:
            logger.error("bandit_reset_error", error=str(e))

# Backward compatibility
LinUCBBandit = EnhancedLinUCBBandit 