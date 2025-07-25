import numpy as np
from typing import List, Dict

class LinUCBBandit:
    def __init__(self, alpha: float = 1.0, feature_dim: int = 10):
        self.alpha = alpha
        self.feature_dim = feature_dim
        self.A = {}  # Her soru için A matrisi
        self.b = {}  # Her soru için b vektörü

    def select_question(self, user_features: Dict, candidate_questions: List[Dict]) -> int:
        ucb_values = {}
        for question in candidate_questions:
            q_id = question['id']
            x = self._build_context_vector(user_features, question)
            if q_id not in self.A:
                self.A[q_id] = np.identity(self.feature_dim)
                self.b[q_id] = np.zeros(self.feature_dim)
            A_inv = np.linalg.inv(self.A[q_id])
            theta = A_inv.dot(self.b[q_id])
            confidence = self.alpha * np.sqrt(x.T.dot(A_inv).dot(x))
            ucb_values[q_id] = theta.dot(x) + confidence
        return max(ucb_values, key=ucb_values.get)

    def update(self, question_id: int, user_features: Dict, question_features: Dict, reward: float):
        x = self._build_context_vector(user_features, question_features)
        if question_id not in self.A:
            self.A[question_id] = np.identity(self.feature_dim)
            self.b[question_id] = np.zeros(self.feature_dim)
        self.A[question_id] += np.outer(x, x)
        self.b[question_id] += reward * x

    def _build_context_vector(self, user_features: Dict, question_features: Dict) -> np.ndarray:
        # Basit birleştirme ve normalize örneği
        features = [
            user_features.get("accuracy_rate_overall", 0.5),
            user_features.get("topic_mastery_math", 0.5),
            user_features.get("avg_response_time", 5000) / 10000,
            question_features.get("difficulty", 1) / 5,
            question_features.get("avg_success_rate", 0.5),
        ]
        if len(features) < self.feature_dim:
            features.extend([0.0] * (self.feature_dim - len(features)))
        return np.array(features[:self.feature_dim]) 