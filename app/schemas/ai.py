from typing import List

from pydantic import BaseModel


class HintRequest(BaseModel):
    question: str


class HintResponse(BaseModel):
    hint: str


class ExplanationRequest(BaseModel):
    question: str
    answer: str


class ExplanationResponse(BaseModel):
    explanation: str


class DifficultyRequest(BaseModel):
    question: str


class DifficultyResponse(BaseModel):
    difficulty: str


class AdaptiveHintRequest(BaseModel):
    question_id: int
    student_id: int


class AdaptiveHintResponse(BaseModel):
    hint: str


class ContextualExplanationRequest(BaseModel):
    question_id: int
    student_id: int
    student_answer: str


class ContextualExplanationResponse(BaseModel):
    explanation: str


class BatchEnrichRequest(BaseModel):
    subject: str
    topic: str


class BatchEnrichResponse(BaseModel):
    enriched_count: int


class LLMStatusResponse(BaseModel):
    openai_configured: bool
    huggingface_configured: bool
    status: str


class IngestWebsiteRequest(BaseModel):
    url: str
    subject: str
    topic: str


class IngestCSVRequest(BaseModel):
    file_path: str
    subject: str


class IngestResponse(BaseModel):
    saved_count: int


# Eski schema'lar (geriye uyumluluk için)
class FeedbackRequest(BaseModel):
    is_correct: bool
    question_topic: str
    student_level: int


class FeedbackResponse(BaseModel):
    feedback: str


class StudyRecommendationRequest(BaseModel):
    weak_topics: List[str]
    strong_topics: List[str]


class StudyRecommendationResponse(BaseModel):
    recommendation: str
