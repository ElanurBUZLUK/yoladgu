from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from app.models.question import Subject, QuestionType, SourceType


class QuestionBase(BaseModel):
    subject: Subject
    content: str = Field(..., min_length=10, max_length=2000)
    question_type: QuestionType
    difficulty_level: int = Field(..., ge=1, le=5)
    topic_category: str = Field(..., min_length=2, max_length=100)
    correct_answer: Optional[str] = None
    options: Optional[List[str]] = None
    source_type: SourceType = SourceType.MANUAL
    pdf_source_path: Optional[str] = None
    question_metadata: Optional[Dict[str, Any]] = None


class QuestionCreate(QuestionBase):
    ...


class QuestionUpdate(BaseModel):
    content: Optional[str] = Field(None, min_length=10, max_length=2000)
    question_type: Optional[QuestionType] = None
    difficulty_level: Optional[int] = Field(None, ge=1, le=5)
    topic_category: Optional[str] = Field(None, min_length=2, max_length=100)
    correct_answer: Optional[str] = None
    options: Optional[List[str]] = None
    question_metadata: Optional[Dict[str, Any]] = None


class QuestionResponse(QuestionBase):
    id: str
    original_difficulty: int
    created_at: str
    
    class Config:
        from_attributes = True


class QuestionRecommendationRequest(BaseModel):
    subject: Subject
    user_level: Optional[int] = Field(None, ge=1, le=5)
    preferred_difficulty: Optional[int] = Field(None, ge=1, le=5)
    topic_categories: Optional[List[str]] = None
    question_types: Optional[List[QuestionType]] = None
    exclude_recent: bool = True
    limit: int = Field(10, ge=1, le=50)
    learning_style: Optional[str] = None
    error_patterns: Optional[List[str]] = None


class QuestionRecommendationResponse(BaseModel):
    questions: List[QuestionResponse]
    recommendation_reason: str
    difficulty_adjustment: Optional[str] = None
    total_available: int
    user_level: int
    next_recommendations: List[str]


class QuestionSearchQuery(BaseModel):
    query: Optional[str] = None
    subject: Optional[Subject] = None
    difficulty_level: Optional[int] = Field(None, ge=1, le=5)
    question_type: Optional[QuestionType] = None
    topic_category: Optional[str] = None
    source_type: Optional[SourceType] = None
    min_difficulty: Optional[int] = Field(None, ge=1, le=5)
    max_difficulty: Optional[int] = Field(None, ge=1, le=5)


class QuestionStats(BaseModel):
    total_questions: int
    by_subject: Dict[str, int]
    by_difficulty: Dict[str, int]
    by_type: Dict[str, int]
    by_source: Dict[str, int]
    average_difficulty: float
    most_common_topics: List[Dict[str, Any]]


class QuestionDifficultyAdjustment(BaseModel):
    question_id: str
    old_difficulty: int
    new_difficulty: int
    reason: str
    adjusted_by: str
    adjustment_date: str


class QuestionPool(BaseModel):
    subject: Subject
    difficulty_level: int
    topic_category: str
    questions: List[QuestionResponse]
    pool_size: int
    last_updated: str


class QuestionMetadata(BaseModel):
    estimated_time: Optional[int] = None
    complexity_score: Optional[float] = None
    prerequisite_topics: Optional[List[str]] = None
    learning_objectives: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    author: Optional[str] = None
    review_status: Optional[str] = None
    usage_count: Optional[int] = None
    success_rate: Optional[float] = None


class BulkQuestionOperation(BaseModel):
    question_ids: List[str]
    operation: str
    parameters: Optional[Dict[str, Any]] = None


class QuestionValidationResult(BaseModel):
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    quality_score: Optional[float] = None


class QuestionContent(BaseModel):
    id: str
    content: str
    question_type: str
    difficulty_level: int
    topic_category: str
    options: Optional[List[str]]

class GetQuestionsByLevelResponse(BaseModel):
    level: int
    count: int
    questions: List[QuestionContent]
    user_current_level: int


class GetQuestionPoolResponse(BaseModel):
    pool_size: int
    subject: str
    difficulty_level: Optional[int]
    total_questions: int


class GetMathTopicsResponse(BaseModel):
    total_topics: int
    all_topics: List[str]
    topics_by_level: Dict[str, List[str]]


class GetDifficultyDistributionResponse(BaseModel):
    total_questions: int
    user_level: int
    distribution: Dict[str, int]


class GetRandomQuestionResponse(BaseModel):
    question: QuestionContent
    difficulty_level: int
