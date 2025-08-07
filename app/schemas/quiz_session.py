from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel


class QuizSessionBase(BaseModel):
    total_questions: int
    correct_answers: int
    wrong_answers: int
    accuracy_percentage: float
    total_time_seconds: int


class QuizSessionCreate(QuizSessionBase):
    session_data: Optional[dict] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def get_started_at(self) -> Optional[datetime]:
        if self.started_at:
            return self.started_at
        if self.session_data and "started_at" in self.session_data:
            return datetime.fromisoformat(
                self.session_data["started_at"].replace("Z", "+00:00")
            )
        return None

    def get_completed_at(self) -> Optional[datetime]:
        if self.completed_at:
            return self.completed_at
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
    sessions: List[QuizSession]
    total_count: int


# Student Response şemaları
class StudentResponseBase(BaseModel):
    question_id: int
    user_id: int
    topic_id: Optional[int] = None
    answer_text: Optional[str] = None
    is_correct: bool
    time_taken_seconds: Optional[int] = None
    confidence_level: Optional[float] = None  # 0-1 arası


class StudentResponseCreate(StudentResponseBase):
    pass


class StudentResponse(StudentResponseBase):
    model_config = {"from_attributes": True}

    id: int
    quiz_session_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None


class StudentResponseList(BaseModel):
    responses: List[StudentResponse]
    total_count: int
