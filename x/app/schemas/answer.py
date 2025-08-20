from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from app.models.question import Subject, QuestionType


class AnswerSubmission(BaseModel):
    question_id: str
    student_answer: str
    time_spent: Optional[int] = Field(None, description="Time spent in seconds")
    attempt_metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional attempt data")


class AnswerEvaluation(BaseModel):
    is_correct: bool
    score: float = Field(..., ge=0, le=100, description="Score out of 100")
    max_score: float = Field(100, description="Maximum possible score")
    feedback: str
    detailed_feedback: Optional[str] = None
    explanation: Optional[str] = None
    error_analysis: Dict[str, List[str]] = Field(default={})
    error_category: Optional[str] = None
    recommendations: List[str] = Field(default=[])
    next_difficulty: Optional[int] = Field(None, ge=1, le=5)
    accuracy_rate: Optional[float] = None
    improvement_suggestions: List[str] = Field(default=[])


class AnswerEvaluationRequest(BaseModel):
    question_id: str
    student_answer: str
    question_content: Optional[str] = None
    correct_answer: Optional[str] = None
    question_type: Optional[QuestionType] = None
    subject: Optional[Subject] = None
    difficulty_level: Optional[int] = Field(None, ge=1, le=5)
    time_spent: Optional[int] = None
    use_llm: bool = Field(True, description="Use LLM for evaluation")


class StudentAttemptResponse(BaseModel):
    id: str
    question_id: str
    student_answer: str
    is_correct: bool
    time_spent: Optional[int]
    attempt_date: str
    error_category: Optional[str]
    score: Optional[float] = None
    feedback: Optional[str] = None
    question_content: Optional[str] = None
    question_type: Optional[str] = None
    subject: Optional[str] = None
    difficulty_level: Optional[int] = None
    
    class Config:
        from_attributes = True


class ErrorAnalysisResult(BaseModel):
    error_type: str
    description: str
    frequency: int
    last_occurrence: str
    subject: Subject
    topic_category: Optional[str] = None
    difficulty_level: Optional[int] = None
    practice_recommendations: List[str] = Field(default=[])
    related_topics: List[str] = Field(default=[])


class PerformanceMetrics(BaseModel):
    total_attempts: int
    correct_attempts: int
    accuracy_rate: float
    average_score: float
    average_time_per_question: float
    subject_performance: Dict[str, Dict[str, Any]] = Field(default={})
    difficulty_performance: Dict[str, Dict[str, Any]] = Field(default={})
    recent_accuracy: Optional[float] = None
    improvement_trend: Optional[str] = None
    current_streak: int = 0
    best_streak: int = 0


class DetailedErrorAnalysis(BaseModel):
    math_errors: Dict[str, Any] = Field(default={})
    grammar_errors: List[Dict[str, Any]] = Field(default=[])
    vocabulary_errors: List[Dict[str, Any]] = Field(default=[])
    conceptual_errors: List[str] = Field(default=[])
    procedural_errors: List[str] = Field(default=[])
    error_frequency: Dict[str, int] = Field(default={})
    most_common_errors: List[str] = Field(default=[])


class LevelAdjustmentRecommendation(BaseModel):
    current_level: int
    recommended_level: int
    reason: str
    confidence: float = Field(..., ge=0, le=1)
    supporting_evidence: List[str] = Field(default=[])
    accuracy_threshold: float
    consistency_threshold: float
    time_threshold: Optional[float] = None


class FeedbackGeneration(BaseModel):
    positive_feedback: List[str] = Field(default=[])
    constructive_feedback: List[str] = Field(default=[])
    encouragement: List[str] = Field(default=[])
    specific_improvements: List[str] = Field(default=[])
    learning_style_adaptations: List[str] = Field(default=[])
    motivational_elements: List[str] = Field(default=[])


class AnswerSubmissionResponse(BaseModel):
    success: bool
    evaluation: AnswerEvaluation
    attempt: StudentAttemptResponse
    spaced_repetition: Optional[Dict]
    message: str


class QuestionDetails(BaseModel):
    id: str
    content: str
    correct_answer: str
    options: Optional[Dict[str, Any]]
    topic_category: Optional[str]
    source_type: str

class MathErrorDetailSchema(BaseModel):
    operation: str
    math_concept: str
    error_step: str

class ErrorDetails(BaseModel):
    grammar_errors: Optional[List[str]]
    vocabulary_errors: Optional[List[str]]
    math_error_details: List[MathErrorDetailSchema]

class AttemptDetailsResponse(BaseModel):
    attempt: StudentAttemptResponse
    question_details: QuestionDetails
    error_details: ErrorDetails


class FeedbackResponse(BaseModel):
    question_id: str
    attempt_id: str
    is_correct: bool
    personalized_feedback: Dict
    generated_at: str


class OverallSummary(BaseModel):
    total_attempts: int
    accuracy_rate: float
    current_streak: int
    best_streak: int

class SubjectSummary(BaseModel):
    attempts: int
    accuracy: float
    current_level: int

class RecentError(BaseModel):
    error_type: str
    frequency: int
    subject: str

class StatisticsSummaryResponse(BaseModel):
    overall: OverallSummary
    by_subject: Dict[str, SubjectSummary]
    recent_errors: List[RecentError]
    improvement_areas: List[str]


class DifficultyPerformance(BaseModel):
    attempts: int
    accuracy: float

class CommonError(BaseModel):
    error_type: str
    frequency: int

class ClassPerformanceAnalyticsResponse(BaseModel):
    total_attempts: int
    class_accuracy: float
    active_students: int
    difficulty_performance: Dict[str, DifficultyPerformance]
    common_errors: List[CommonError]
    analysis_period: str
    subject_filter: str


class SimilarStudentsResponse(BaseModel):
    similar_students: List[Dict]
    total_found: int
    subject: str
    user_id: str


class InterventionRecommendationsResponse(BaseModel):
    interventions: List[Dict]
    total_recommendations: int
    subject: str
    generated_at: str


class TrackedErrorPattern(BaseModel):
    id: str
    error_type: str
    error_count: int
    subject: str
    last_occurrence: str

class TrackErrorPatternResponse(BaseModel):
    success: bool
    error_pattern: TrackedErrorPattern
    message: str


class ClassErrorAnalyticsResponse(BaseModel):
    total_error_types: int
    affected_students: int
    most_common_errors: List[Dict]
    analysis_period: str
    subject_filter: str


class UserErrorPatternsAdminResponse(BaseModel):
    user_id: str
    error_patterns: List[Dict]
    total_patterns: int
    subject_filter: str


class ErrorOverview(BaseModel):
    total_error_types: int
    affected_students: int
    analysis_period: str

class SubjectErrorSummary(BaseModel):
    error_types: int
    affected_students: int
    top_errors: List[Dict]

class CriticalError(BaseModel):
    error_type: str
    frequency: int
    subject: str
    student_count: int

class ErrorAnalyticsOverviewResponse(BaseModel):
    overview: ErrorOverview
    by_subject: Dict[str, SubjectErrorSummary]
    critical_errors: List[CriticalError]
    recommendations: List[str]


class BatchEvaluateLevelAdjustmentsResponse(BaseModel):
    recommendations: List[Dict]
    total_users_evaluated: int
    subject: str
    min_attempts_threshold: int
    evaluation_date: str


class AdminApplyLevelAdjustmentResponse(BaseModel):
    success: bool
    adjustment: Dict
    target_user_id: str
    applied_by: str
    message: str


class UserLevelHistoryAdminResponse(BaseModel):
    level_history: List[Dict]
    total_adjustments: int
    user_id: str
    subject_filter: str
    requested_by: str


class LevelAnalyticsOverviewResponse(BaseModel):
    level_distribution: Dict[str, Dict]
    pending_recommendations: Dict[str, Dict]
    recommendations_summary: Dict[str, List[Dict]]
    generated_at: str


class ReviewCalendarResponse(BaseModel):
    calendar: Dict[str, List[Dict]]
    total_days: int
    total_reviews: int
    days_ahead: int


class ResetQuestionProgressResponse(BaseModel):
    success: bool
    message: str
    question_id: str
    user_id: str


class ProcessAnswerForReviewResponse(BaseModel):
    question_id: str
    next_review_date: str
    interval: int
    ease_factor: float
    repetitions: int
    message: str


class UserReviewStatisticsAdminResponse(BaseModel):
    user_id: str
    statistics: Dict
    requested_by: str


class UserLearningProgressAdminResponse(BaseModel):
    user_id: str
    progress: Dict
    requested_by: str


class ResetUserQuestionProgressAdminResponse(BaseModel):
    user_id: str
    reset_result: Dict
    applied_by: str


class PerformanceSummary(BaseModel):
    accuracy_rate: float
    current_level: int
    total_attempts: int
    current_streak: int
    progress_percentage: float
    overdue_reviews: int
    due_today: int

class RecommendationItem(BaseModel):
    type: str
    priority: str
    title: str
    description: str
    action: str
    confidence: float

class PerformanceBasedRecommendationsResponse(BaseModel):
    subject: str
    user_id: str
    performance_summary: PerformanceSummary
    recommendations: List[RecommendationItem]
    total_recommendations: int
    generated_at: str


class StudyPlanActivity(BaseModel):
    type: str
    title: str
    duration_minutes: int
    priority: str
    questions: Optional[int] = None

class StudyPlanDay(BaseModel):
    date: str
    day_name: str
    total_time_minutes: int
    activities: List[StudyPlanActivity]
    scheduled_reviews: int
    focus_area: str

class StudyPlanRecommendationsResponse(BaseModel):
    subject: str
    user_id: str
    study_plan: List[StudyPlanDay]
    total_days: int
    average_daily_time: float
    total_scheduled_reviews: int
    generated_at: str


class RecentPerformance(BaseModel):
    attempts_analyzed: int
    accuracy: float
    average_difficulty: float
    average_time_seconds: float
    current_streak: int

class AdaptiveRecommendationItem(BaseModel):
    type: str
    title: str
    description: str
    immediate_action: str
    confidence: float

class AdaptiveRecommendationsResponse(BaseModel):
    subject: str
    user_id: Optional[str] = None
    recent_performance: RecentPerformance
    recommendations: List[AdaptiveRecommendationItem]
    next_suggested_action: str
    generated_at: str
    message: Optional[str] = None


class AnswersHealthResponse(BaseModel):
    status: str
    module: str
