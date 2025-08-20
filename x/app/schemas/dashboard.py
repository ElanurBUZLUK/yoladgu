from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from app.models.question import Subject


class SubjectChoice(BaseModel):
    subject: Subject
    selected_at: str


class DashboardStats(BaseModel):
    total_questions_answered: int
    correct_answers: int
    accuracy_percentage: float
    current_level: int
    questions_today: int
    current_streak: int
    best_streak: int
    time_spent_today: int  # in seconds
    time_spent_total: int  # in seconds


class SubjectProgress(BaseModel):
    subject: Subject
    current_level: int
    progress_percentage: float  # Progress within current level
    total_questions: int
    correct_answers: int
    accuracy_rate: float
    last_activity: Optional[str]
    next_review_count: int  # Spaced repetition queue
    weak_areas: List[str]
    strong_areas: List[str]


class DashboardData(BaseModel):
    user_info: Dict[str, Any]
    math_progress: Optional[SubjectProgress]
    english_progress: Optional[SubjectProgress]
    overall_stats: DashboardStats
    recent_activity: List[Dict[str, Any]]
    recommendations: List[str]
    achievements: List[Dict[str, Any]]


class SubjectSelectionResponse(BaseModel):
    available_subjects: List[Subject]
    user_levels: Dict[str, int]
    recommendations: List[str]
    last_selected: Optional[Subject]


class LearningStyleAdaptation(BaseModel):
    learning_style: str
    adaptations: Dict[str, Any]
    recommended_features: List[str]
    ui_preferences: Dict[str, Any]


class PerformanceSummary(BaseModel):
    subject: Subject
    period: str  # "today", "week", "month", "all_time"
    questions_attempted: int
    correct_answers: int
    accuracy_rate: float
    average_time_per_question: float
    improvement_rate: float  # Compared to previous period
    difficulty_distribution: Dict[str, int]
    topic_performance: Dict[str, float]


class WeeklyProgress(BaseModel):
    week_start: str
    week_end: str
    daily_stats: List[Dict[str, Any]]
    weekly_goals: Dict[str, Any]
    achievements_unlocked: List[str]
    total_time_spent: int
    questions_answered: int
    accuracy_trend: List[float]


class RecommendationItem(BaseModel):
    type: str  # "question", "topic", "level_adjustment", "break"
    title: str
    description: str
    priority: int  # 1-5, 5 being highest
    subject: Optional[Subject]
    estimated_time: Optional[int]  # in minutes
    difficulty_level: Optional[int]


class UserPreferences(BaseModel):
    preferred_question_types: List[str]
    difficulty_preference: str  # "adaptive", "challenging", "comfortable"
    session_length_preference: int  # in minutes
    reminder_settings: Dict[str, bool]
    ui_theme: str  # "light", "dark", "auto"
    accessibility_options: Dict[str, bool]