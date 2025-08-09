from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class StartQuizRequest(BaseModel):
    topic_id: Optional[int] = Field(default=None, description="Optional topic/skill identifier")


class StartQuizResponse(BaseModel):
    session_id: str
    question: Dict[str, Any]


class AnswerRequest(BaseModel):
    session_id: str
    question_id: int
    selected_option: int
    time_ms: int = 0
    hint_used: bool = False


class NextRequest(BaseModel):
    session_id: str


class FinishRequest(BaseModel):
    session_id: str


