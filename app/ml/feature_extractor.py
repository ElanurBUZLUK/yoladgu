import json
from typing import Dict
from sqlalchemy.orm import Session

class UserFeatureExtractor:
    def __init__(self, db: Session, redis_client):
        self.db = db
        self.redis = redis_client
    
    def extract_features(self, user_id: int) -> Dict[str, float]:
        """Real-time feature extraction"""
        cached = self.redis.get(f"user:{user_id}:features")
        if cached:
            return json.loads(cached)
        
        features = {
            "accuracy_rate_overall": self._get_accuracy_rate(user_id),
            "accuracy_rate_last_10": self._get_recent_accuracy(user_id, 10),
            "avg_response_time": self._get_avg_response_time(user_id),
            "topic_mastery_math": self._get_topic_mastery(user_id, "math"),
            "consecutive_correct": self._get_consecutive_correct(user_id),
            "session_question_count": self._get_session_count(user_id),
            "time_since_last_question": self._get_time_since_last(user_id),
            "difficulty_trend": self._get_difficulty_trend(user_id)
        }
        self.redis.setex(f"user:{user_id}:features", 60, json.dumps(features))
        return features

    def _get_accuracy_rate(self, user_id: int) -> float:
        # Kullanıcının tüm çözümlerindeki doğruluk oranı
        from app.db.models import Solution
        total = self.db.query(Solution).filter(Solution.user_id == user_id).count()
        if total == 0:
            return 0.0
        correct = self.db.query(Solution).filter(
            Solution.user_id == user_id,
            Solution.is_correct == True
        ).count()
        return correct / total

    def _get_recent_accuracy(self, user_id: int, n: int) -> float:
        # Kullanıcının son n çözümündeki doğruluk oranı
        from app.db.models import Solution
        recent_solutions = (
            self.db.query(Solution)
            .filter(Solution.user_id == user_id)
            .order_by(Solution.solved_at.desc())
            .limit(n)
            .all()
        )
        if not recent_solutions:
            return 0.0
        correct = sum(1 for s in recent_solutions if s.is_correct)
        return correct / len(recent_solutions)

    def _get_avg_response_time(self, user_id: int) -> float:
        # TODO: SQL query for average response time
        return 4200.0

    def _get_topic_mastery(self, user_id: int, topic: str) -> float:
        # TODO: SQL query for topic-specific success rate
        return 0.6

    def _get_consecutive_correct(self, user_id: int) -> int:
        # TODO: SQL query for consecutive correct answers
        return 3

    def _get_session_count(self, user_id: int) -> int:
        # TODO: SQL query for session question count
        return 12

    def _get_time_since_last(self, user_id: int) -> float:
        # TODO: SQL query for time since last question
        return 1800.0

    def _get_difficulty_trend(self, user_id: int) -> float:
        # TODO: SQL query for difficulty trend
        return 0.1 