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
            "difficulty_trend": self._get_difficulty_trend(user_id),
        }
        self.redis.setex(f"user:{user_id}:features", 60, json.dumps(features))
        return features

    def _get_accuracy_rate(self, user_id: int) -> float:
        # Kullanıcının tüm çözümlerindeki doğruluk oranı
        from app.db.models import Solution

        total = self.db.query(Solution).filter(Solution.user_id == user_id).count()
        if total == 0:
            return 0.0
        correct = (
            self.db.query(Solution)
            .filter(Solution.user_id == user_id, Solution.is_correct == True)
            .count()
        )
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
        # Kullanıcının ortalama cevap süresi
        from app.db.models import StudentResponse
        from sqlalchemy import func

        result = (
            self.db.query(func.avg(StudentResponse.response_time))
            .filter(StudentResponse.student_id == user_id)
            .filter(StudentResponse.response_time.isnot(None))
            .scalar()
        )
        return float(result) if result else 0.0

    def _get_topic_mastery(self, user_id: int, topic: str) -> float:
        # Belirli bir konudaki başarı oranı
        from app.db.models import Question, StudentResponse, Subject

        result = (
            self.db.query(StudentResponse)
            .join(Question, StudentResponse.question_id == Question.id)
            .join(Subject, Question.subject_id == Subject.id)
            .filter(StudentResponse.student_id == user_id)
            .filter(Subject.name.ilike(f"%{topic}%"))
            .all()
        )

        if not result:
            return 0.0

        correct = sum(1 for r in result if r.is_correct)
        return correct / len(result)

    def _get_consecutive_correct(self, user_id: int) -> int:
        # Ardışık doğru cevap sayısı
        from app.db.models import StudentResponse

        responses = (
            self.db.query(StudentResponse)
            .filter(StudentResponse.student_id == user_id)
            .order_by(StudentResponse.created_at.desc())
            .all()
        )

        consecutive = 0
        for response in responses:
            if response.is_correct:
                consecutive += 1
            else:
                break

        return consecutive

    def _get_session_count(self, user_id: int) -> int:
        # Mevcut oturumdaki soru sayısı (son 30 dakika)
        from datetime import datetime, timedelta

        from app.db.models import StudentResponse

        thirty_minutes_ago = datetime.utcnow() - timedelta(minutes=30)

        count = (
            self.db.query(StudentResponse)
            .filter(StudentResponse.student_id == user_id)
            .filter(StudentResponse.created_at >= thirty_minutes_ago)
            .count()
        )

        return count

    def _get_time_since_last(self, user_id: int) -> float:
        # Son sorudan bu yana geçen süre (saniye)
        from datetime import datetime

        from app.db.models import StudentResponse

        last_response = (
            self.db.query(StudentResponse)
            .filter(StudentResponse.student_id == user_id)
            .order_by(StudentResponse.created_at.desc())
            .first()
        )

        if not last_response:
            return 0.0

        time_diff = datetime.utcnow() - last_response.created_at
        return time_diff.total_seconds()

    def _get_difficulty_trend(self, user_id: int) -> float:
        # Zorluk seviyesi trendi (son 10 soruda)
        from app.db.models import Question, StudentResponse

        recent_responses = (
            self.db.query(StudentResponse, Question.difficulty_level)
            .join(Question, StudentResponse.question_id == Question.id)
            .filter(StudentResponse.student_id == user_id)
            .order_by(StudentResponse.created_at.desc())
            .limit(10)
            .all()
        )

        if len(recent_responses) < 2:
            return 0.0

        # Son 5 soru ile önceki 5 sorunun ortalama zorluk farkı
        recent_avg = sum(r[1] for r in recent_responses[:5]) / 5
        older_avg = sum(r[1] for r in recent_responses[5:]) / 5

        return recent_avg - older_avg
