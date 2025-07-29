from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from app.db.models import UserRole

class QuestionBase(BaseModel):
    content: str = Field(..., description="Question content")
    question_type: str = Field(default="multiple_choice", description="Type of question (multiple_choice, true_false, open_ended)")
    difficulty_level: int = Field(..., ge=1, le=5, description="Difficulty level from 1 to 5")
    subject_id: int = Field(..., description="ID of the subject this question belongs to")
    topic_id: Optional[int] = Field(None, description="ID of the topic this question belongs to")
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
    topic_id: Optional[int] = None
    options: Optional[Dict[str, Any]] = None
    correct_answer: Optional[str] = None
    explanation: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class QuestionResponse(QuestionBase):
    id: int
    created_by: int
    is_active: bool
    embedding: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class QuestionRecommendation(BaseModel):
    question_id: int
    ensemble_score: float
    river_score: float
    embedding_similarity: float
    skill_mastery: float
    difficulty_match: float
    neo4j_similarity: float
    adjusted_difficulty: int
    original_difficulty: int
    question: QuestionResponse

    class Config:
        from_attributes = True

class QuestionSimilarity(BaseModel):
    question_id: int
    similarity_score: float
    shared_skills: List[str]
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
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class TopicBase(BaseModel):
    name: str
    description: Optional[str] = None
    subject_id: int

class TopicCreate(TopicBase):
    pass

class TopicResponse(TopicBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

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
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class SkillCentrality(BaseModel):
    skill_id: int
    skill_name: str
    centrality_score: float
    question_count: int
    student_mastery_avg: float

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
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class AnswerSubmission(BaseModel):
    answer: str
    confidence_level: Optional[int] = Field(None, ge=1, le=5)
    feedback: Optional[str] = None

class StudentSkillMastery(BaseModel):
    skill_id: int
    skill_name: str
    mastery_level: float
    questions_answered: int
    correct_answers: int
    average_response_time: float

    class Config:
        from_attributes = True

class LearningPath(BaseModel):
    skill_id: int
    skill_name: str
    current_level: float
    target_level: float
    recommended_questions: List[int]
    estimated_time: int  # minutes

    class Config:
        from_attributes = True

class RecommendationRequest(BaseModel):
    n_recommendations: int = Field(default=5, ge=1, le=20)
    subject_id: Optional[int] = None
    topic_id: Optional[int] = None
    difficulty_range: Optional[tuple[int, int]] = None
    include_explanations: bool = Field(default=False)
    use_ensemble: bool = Field(default=True)

class RecommendationResponse(BaseModel):
    recommendations: List[QuestionRecommendation]
    student_level: float
    total_questions_answered: int
    accuracy_rate: float
    average_response_time: float
    generated_at: datetime 