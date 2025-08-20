from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class MasteryLevel(str, Enum):
    LEARNING = "learning"
    REVIEWING = "reviewing"
    MASTERED = "mastered"


class ReviewQuality(int, Enum):
    """SM-2 Quality levels (0-5)"""
    COMPLETE_FAILURE = 0
    INCORRECT_EASY = 1
    INCORRECT_HARD = 2
    CORRECT_HARD = 3
    CORRECT_NORMAL = 4
    CORRECT_EASY = 5


class SpacedRepetitionBase(BaseModel):
    user_id: str
    question_id: str
    ease_factor: float = Field(default=2.5, ge=1.3, le=4.0)
    review_count: int = Field(default=0, ge=0)
    next_review_at: datetime
    last_reviewed: Optional[datetime] = None


class SpacedRepetitionCreate(BaseModel):
    question_id: str
    quality: int = Field(ge=0, le=5, description="Quality score (0-5)")
    response_time: Optional[int] = Field(None, ge=0, description="Response time in seconds")


class SpacedRepetitionUpdate(BaseModel):
    quality: int = Field(ge=0, le=5, description="Quality score (0-5)")
    response_time: Optional[int] = Field(None, ge=0, description="Response time in seconds")


class SpacedRepetitionResponse(SpacedRepetitionBase):
    class Config:
        from_attributes = True


class ReviewScheduleResult(BaseModel):
    user_id: str
    question_id: str
    quality: int
    ease_factor: float
    interval_days: int
    review_count: int
    next_review_at: str
    last_reviewed: str


class DueReview(BaseModel):
    question_id: str
    question_content: str
    question_type: str
    subject: str
    difficulty_level: int
    topic_category: str
    review_count: int
    ease_factor: float
    next_review_at: str
    overdue_hours: float
    priority: int = Field(ge=1, le=5, description="Priority level (1-5)")
    last_reviewed: Optional[str] = None


class ReviewStatistics(BaseModel):
    total_reviews: int
    average_ease_factor: float
    due_today: int
    overdue: int
    upcoming_7_days: int
    retention_rate: float = Field(description="Percentage of questions with good retention")
    average_interval: float = Field(description="Average interval between reviews in days")
    analysis_period_days: int


class ReviewCalendarDay(BaseModel):
    question_id: str
    question_content: str
    subject: str
    difficulty_level: int
    review_count: int
    ease_factor: float
    scheduled_time: str


class ReviewCalendar(BaseModel):
    calendar: Dict[str, List[ReviewCalendarDay]]
    total_days: int
    total_reviews: int


class LearningProgress(BaseModel):
    total_questions: int
    mastery_levels: Dict[str, int] = Field(
        description="Count of questions in each mastery level"
    )
    average_ease_factor: float
    progress_percentage: float = Field(
        ge=0, le=100, description="Overall learning progress percentage"
    )
    subject: str


class BulkReviewRequest(BaseModel):
    question_results: List[Dict[str, Any]] = Field(
        description="List of question results with question_id, quality, and optional response_time"
    )


class BulkReviewResponse(BaseModel):
    total_processed: int
    scheduled_count: int
    failed_count: int
    results: List[Dict[str, Any]]


class ReviewProgressReset(BaseModel):
    success: bool
    message: str
    question_id: str
    new_ease_factor: Optional[float] = None
    new_review_count: Optional[int] = None
    next_review_at: Optional[str] = None


class SpacedRepetitionSettings(BaseModel):
    """User settings for spaced repetition"""
    max_daily_reviews: int = Field(default=50, ge=1, le=200)
    review_buffer_hours: int = Field(default=2, ge=0, le=24)
    enable_notifications: bool = Field(default=True)
    preferred_review_time: Optional[str] = Field(None, description="Preferred time for reviews (HH:MM)")
    weekend_reviews: bool = Field(default=True)


class ReviewSession(BaseModel):
    """A review session containing multiple questions"""
    session_id: str
    user_id: str
    subject: Optional[str] = None
    questions: List[DueReview]
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_questions: int
    completed_questions: int = 0
    session_score: Optional[float] = None


class ReviewSessionResult(BaseModel):
    """Result of a completed review session"""
    session_id: str
    total_questions: int
    correct_answers: int
    accuracy_rate: float
    average_response_time: Optional[float] = None
    questions_promoted: int = Field(description="Questions that moved to longer intervals")
    questions_demoted: int = Field(description="Questions that moved to shorter intervals")
    session_duration_minutes: int
    next_session_recommended_at: Optional[datetime] = None


class ReviewReminder(BaseModel):
    """Review reminder notification"""
    user_id: str
    due_count: int
    overdue_count: int
    subjects: List[str]
    priority_questions: List[str] = Field(description="High priority question IDs")
    reminder_type: str = Field(description="Type of reminder: daily, urgent, weekly")
    scheduled_for: datetime


class ReviewAnalytics(BaseModel):
    """Analytics for spaced repetition performance"""
    user_id: str
    period_days: int
    total_reviews_completed: int
    average_daily_reviews: float
    retention_trend: str = Field(description="improving, stable, declining")
    difficult_topics: List[str] = Field(description="Topics with low ease factors")
    mastery_progression: Dict[str, int] = Field(description="Questions moved between mastery levels")
    optimal_review_time: Optional[str] = Field(description="Best performing time of day")
    consistency_score: float = Field(ge=0, le=100, description="How consistently user does reviews")


class ReviewRecommendation(BaseModel):
    """Personalized review recommendations"""
    user_id: str
    recommended_daily_reviews: int
    focus_subjects: List[str]
    suggested_session_duration: int = Field(description="Recommended session duration in minutes")
    difficulty_adjustment: str = Field(description="easier, maintain, harder")
    break_recommendation: Optional[str] = Field(description="Suggested break if user is overloaded")
    motivation_message: str


# Request/Response models for API endpoints
class GetDueReviewsRequest(BaseModel):
    subject: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=100)


class ScheduleReviewRequest(BaseModel):
    question_id: str
    quality: int = Field(ge=0, le=5)
    response_time: Optional[int] = Field(None, ge=0)


class ReviewStatisticsRequest(BaseModel):
    days: int = Field(default=30, ge=1, le=365)


class ReviewCalendarRequest(BaseModel):
    days_ahead: int = Field(default=30, ge=1, le=90)


class LearningProgressRequest(BaseModel):
    subject: Optional[str] = None