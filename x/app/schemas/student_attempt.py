from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from uuid import UUID


class StudentAttemptBase(BaseModel):
    user_id: UUID
    question_id: UUID
    student_answer: Optional[str] = None
    is_correct: bool
    time_spent: Optional[int] = None  # in seconds
    error_category: Optional[str] = None
    grammar_errors: Optional[Dict[str, Any]] = None
    vocabulary_errors: Optional[Dict[str, Any]] = None


class StudentAttemptCreate(StudentAttemptBase):
    pass


class StudentAttemptUpdate(BaseModel):
    student_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    time_spent: Optional[int] = None
    error_category: Optional[str] = None
    grammar_errors: Optional[Dict[str, Any]] = None
    vocabulary_errors: Optional[Dict[str, Any]] = None


class StudentAttemptInDB(StudentAttemptBase):
    id: UUID
    attempt_date: str  # Using str for datetime for simplicity in Pydantic

    class Config:
        from_attributes = True
