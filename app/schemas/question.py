"""
Question DTOs
Soru veri transfer objeleri
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class QuestionType(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    FILL_BLANK = "fill_blank"
    SHORT_ANSWER = "short_answer"
    ESSAY = "essay"
    MATCHING = "matching"
    ORDERING = "ordering"

class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"

class QuestionStatus(str, Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    REVIEW = "review"

class QuestionCreate(BaseModel):
    title: str = Field(..., min_length=10, max_length=500)
    content: str = Field(..., min_length=20)
    question_type: QuestionType
    difficulty: DifficultyLevel
    subject: str = Field(..., min_length=2, max_length=100)
    topic: str = Field(..., min_length=2, max_length=100)
    subtopic: Optional[str] = None
    options: Optional[List[str]] = []
    correct_answer: str
    explanation: Optional[str] = None
    tags: List[str] = []
    image_url: Optional[str] = None
    audio_url: Optional[str] = None
    video_url: Optional[str] = None
    points: int = Field(1, ge=1, le=100)
    time_limit: Optional[int] = Field(None, ge=30, le=3600)  # seconds
    hints: List[str] = []
    metadata: Dict[str, Any] = {}

class QuestionUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=10, max_length=500)
    content: Optional[str] = Field(None, min_length=20)
    question_type: Optional[QuestionType] = None
    difficulty: Optional[DifficultyLevel] = None
    subject: Optional[str] = Field(None, min_length=2, max_length=100)
    topic: Optional[str] = Field(None, min_length=2, max_length=100)
    subtopic: Optional[str] = None
    options: Optional[List[str]] = None
    correct_answer: Optional[str] = None
    explanation: Optional[str] = None
    tags: Optional[List[str]] = None
    image_url: Optional[str] = None
    audio_url: Optional[str] = None
    video_url: Optional[str] = None
    points: Optional[int] = Field(None, ge=1, le=100)
    time_limit: Optional[int] = Field(None, ge=30, le=3600)
    hints: Optional[List[str]] = None
    status: Optional[QuestionStatus] = None
    metadata: Optional[Dict[str, Any]] = None

class QuestionResponse(BaseModel):
    """Frontend'e gönderilen soru"""
    id: int
    content: str
    options: List[str]
    correct_answer: str
    difficulty_level: int
    subject_id: int
    subject: Optional[str] = None
    topic: Optional[str] = None
    hint: Optional[str] = None
    explanation: Optional[str] = None
    question_type: str
    tags: Optional[List[str]] = None
    created_by: int
    is_active: bool

class QuestionAnswer(BaseModel):
    question_id: int
    user_id: int
    answer: str
    is_correct: bool
    time_taken: float  # seconds
    points_earned: int
    submitted_at: datetime

class QuestionStats(BaseModel):
    question_id: int
    total_attempts: int
    correct_attempts: int
    incorrect_attempts: int
    average_score: float
    average_time: float
    difficulty_rating: float
    popularity_score: float
    success_rate: float

class QuestionSearch(BaseModel):
    query: Optional[str] = None
    subject: Optional[str] = None
    topic: Optional[str] = None
    question_type: Optional[QuestionType] = None
    difficulty: Optional[DifficultyLevel] = None
    tags: Optional[List[str]] = None
    status: Optional[QuestionStatus] = None
    created_by: Optional[int] = None
    min_points: Optional[int] = None
    max_points: Optional[int] = None
    page: int = 1
    per_page: int = 20
    sort_by: str = "created_at"
    sort_order: str = "desc"

class QuestionList(BaseModel):
    questions: List[QuestionResponse]
    total: int
    page: int
    per_page: int
    total_pages: int

class QuestionBatch(BaseModel):
    questions: List[QuestionCreate]
    batch_name: str
    description: Optional[str] = None
    tags: List[str] = []

class QuestionImport(BaseModel):
    file_url: str
    file_type: str  # csv, excel, json
    subject: str
    topic: str
    tags: List[str] = []
    overwrite_existing: bool = False

class QuestionExport(BaseModel):
    question_ids: List[int]
    format: str = "json"  # json, csv, excel
    include_stats: bool = False
    include_answers: bool = False

class QuestionFeedback(BaseModel):
    question_id: int
    user_id: int
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None
    difficulty_rating: Optional[int] = Field(None, ge=1, le=5)
    clarity_rating: Optional[int] = Field(None, ge=1, le=5)
    relevance_rating: Optional[int] = Field(None, ge=1, le=5)
    submitted_at: datetime

class QuestionAnalytics(BaseModel):
    question_id: int
    daily_attempts: Dict[str, int]
    weekly_attempts: Dict[str, int]
    monthly_attempts: Dict[str, int]
    success_rate_by_difficulty: Dict[str, float]
    average_time_by_difficulty: Dict[str, float]
    popular_wrong_answers: List[Dict[str, Any]]
    user_performance_distribution: Dict[str, int]


class AnswerSubmission(BaseModel):
    answer: str = Field(..., description="Öğrencinin cevabı")
    response_time: Optional[float] = Field(None, description="Cevap süresi (saniye)")
    confidence_level: Optional[int] = Field(None, ge=1, le=5, description="Güven seviyesi (1-5)")
    feedback: Optional[str] = Field(None, description="Öğrenci geri bildirimi")


class AnswerResponse(BaseModel):
    question_id: int
    is_correct: bool
    correct_answer: str
    explanation: Optional[str] = None
    response_time: float
    response_id: int
    points_earned: int
    current_streak: Optional[int] = None
    message: str

class AnswerSubmit(BaseModel):
    """Frontend'ten gelen cevap"""
    answer: str = Field(..., description="Öğrencinin cevabı")
    response_time: Optional[float] = Field(None, description="Cevap süresi (saniye)")
    confidence_level: Optional[int] = Field(None, ge=1, le=5, description="Güven seviyesi (1-5)")

class SubmitAnswerResponse(BaseModel):
    """Cevap sonucu"""
    is_correct: bool
    correct_answer: str
    explanation: Optional[str] = None
    points_earned: int = 0
    current_streak: Optional[int] = None
    message: str
