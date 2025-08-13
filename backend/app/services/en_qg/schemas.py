from __future__ import annotations

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class GeneratedOption(BaseModel):
    text: str


class GeneratedQuestion(BaseModel):
    id: str
    type: str  # "vocab_gap" | "grammar_fix" | "mcq"
    stem: str
    options: List[GeneratedOption]
    correct_index: int
    rationale: Optional[str] = None
    meta: Dict[str, Any] = {}


class SubmitIn(BaseModel):
    question_id: str
    student_id: int
    student_answer: str
    gold_answer: str
    meta: Dict[str, Any] = {}


class NextParams(BaseModel):
    student_id: int
    mode: str = "mixed"
    cefr: Optional[str] = None
    k_ctx: int = 5


