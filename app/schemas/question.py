from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from app.db.models import UserRole

class QuestionBase(BaseModel):
    content: str = Field(..., description="Question content")
    question_type: str = Field(..., description="Type of question (multiple_choice, true_false, open_ended)")
    difficulty_level: int = Field(..., ge=1, le=5, description="Difficulty level from 1 to 5")
    subject_id: int = Field(..., description="ID of the subject this question belongs to")
    options: Optional[Dict[str, Any]] = Field(None, description="Options for multiple choice questions")
    correct_answer: str = Field(..., description="Correct answer to the question")
    explanation: Optional[str] = Field(None, description="Explanation of the correct answer")
    tags: Optional[Dict[str, Any]] = Field(None, description="Additional tags for categorization")

class QuestionCreate(QuestionBase):
    skill_ids: Optional[Dict[int, float]] = Field(None, description="Skill IDs and their weights for this question")

class QuestionUpdate(BaseModel):
    content: Optional[str] = None
    question_type: Optional[str] = None
    difficulty_level: Optional[int] = Field(None, ge=1, le=5)
    subject_id: Optional[int] = None
    options: Optional[Dict[str, Any]] = None
    correct_answer: Optional[str] = None
    explanation: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class QuestionResponse(QuestionBase):
    id: int
    created_by: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class QuestionRecommendation(BaseModel):
    question_id: int
    score: float
    river_score: float
    difficulty_match: float
    skill_match: float
    question: QuestionResponse

    class Config:
        from_attributes = True

class SubjectBase(BaseModel):
    name: str
    description: Optional[str] = None

class SubjectCreate(SubjectBase):
    pass

class SubjectResponse(SubjectBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class SkillBase(BaseModel):
    name: str
    description: Optional[str] = None
    subject_id: int
    difficulty_level: int = Field(..., ge=1, le=5)

class SkillCreate(SkillBase):
    pass

class SkillResponse(SkillBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class StudentResponseBase(BaseModel):
    answer: str
    confidence_level: Optional[int] = Field(None, ge=1, le=5)
    feedback: Optional[str] = None

class StudentResponseCreate(StudentResponseBase):
    question_id: int

class StudentResponseResponse(StudentResponseBase):
    id: int
    student_id: int
    question_id: int
    is_correct: bool
    response_time: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True

class AnswerSubmission(BaseModel):
    answer: str
    confidence_level: Optional[int] = Field(None, ge=1, le=5)
    feedback: Optional[str] = None 