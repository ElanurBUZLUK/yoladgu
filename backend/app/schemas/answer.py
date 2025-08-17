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
    
    # Error analysis
    error_analysis: Dict[str, List[str]] = Field(default={})
    error_category: Optional[str] = None
    
    # Recommendations
    recommendations: List[str] = Field(default=[])
    next_difficulty: Optional[int] = Field(None, ge=1, le=5)
    
    # Performance metrics
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
    
    # Question details
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
    
    # Improvement suggestions
    practice_recommendations: List[str] = Field(default=[])
    related_topics: List[str] = Field(default=[])


class PerformanceMetrics(BaseModel):
    total_attempts: int
    correct_attempts: int
    accuracy_rate: float
    average_score: float
    average_time_per_question: float
    
    # Subject-specific metrics
    subject_performance: Dict[str, Dict[str, Any]] = Field(default={})
    
    # Difficulty progression
    difficulty_performance: Dict[str, Dict[str, Any]] = Field(default={})
    
    # Recent performance
    recent_accuracy: Optional[float] = None
    improvement_trend: Optional[str] = None  # "improving", "declining", "stable"
    
    # Streaks
    current_streak: int = 0
    best_streak: int = 0


class DetailedErrorAnalysis(BaseModel):
    # Math-specific errors
    math_errors: Dict[str, Any] = Field(default={})
    
    # English-specific errors
    grammar_errors: List[Dict[str, Any]] = Field(default=[])
    vocabulary_errors: List[Dict[str, Any]] = Field(default=[])
    
    # Common error patterns
    conceptual_errors: List[str] = Field(default=[])
    procedural_errors: List[str] = Field(default=[])
    
    # Error frequency analysis
    error_frequency: Dict[str, int] = Field(default={})
    most_common_errors: List[str] = Field(default=[])


class LevelAdjustmentRecommendation(BaseModel):
    current_level: int
    recommended_level: int
    reason: str
    confidence: float = Field(..., ge=0, le=1)
    supporting_evidence: List[str] = Field(default=[])
    
    # Performance thresholds
    accuracy_threshold: float
    consistency_threshold: float
    time_threshold: Optional[float] = None


class FeedbackGeneration(BaseModel):
    positive_feedback: List[str] = Field(default=[])
    constructive_feedback: List[str] = Field(default=[])
    encouragement: List[str] = Field(default=[])
    specific_improvements: List[str] = Field(default=[])
    
    # Personalized elements
    learning_style_adaptations: List[str] = Field(default=[])
    motivational_elements: List[str] = Field(default=[])


class BatchAnswerEvaluation(BaseModel):
    evaluations: List[AnswerEvaluation]
    summary: Dict[str, Any]
    batch_metrics: PerformanceMetrics
    recommendations: List[str] = Field(default=[])


class AnswerValidation(BaseModel):
    is_valid_format: bool
    validation_errors: List[str] = Field(default=[])
    normalized_answer: Optional[str] = None
    confidence_score: Optional[float] = Field(None, ge=0, le=1)