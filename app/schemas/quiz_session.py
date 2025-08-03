from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class QuizSessionBase(BaseModel):
    total_questions: int
    correct_answers: int
    wrong_answers: int
    accuracy_percentage: float
    total_time_seconds: int


class QuizSessionCreate(QuizSessionBase):
    session_data: Optional[dict] = None

    def get_started_at(self) -> Optional[datetime]:
        if self.session_data and "started_at" in self.session_data:
            return datetime.fromisoformat(
                self.session_data["started_at"].replace("Z", "+00:00")
            )
        return None

    def get_completed_at(self) -> Optional[datetime]:
        if self.session_data and "completed_at" in self.session_data:
            return datetime.fromisoformat(
                self.session_data["completed_at"].replace("Z", "+00:00")
            )
        return None


class QuizSession(QuizSessionBase):
    model_config = {"from_attributes": True}

    id: int
    student_id: int
    started_at: datetime
    completed_at: datetime
    created_at: datetime
    updated_at: Optional[datetime] = None


class QuizSessionList(BaseModel):
    sessions: list[QuizSession]
    total_count: int
