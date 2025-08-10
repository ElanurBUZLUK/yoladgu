from pydantic import BaseModel
from typing import Optional, Dict, List

class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None
    role: Optional[str] = "student"

class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class LoginRequest(BaseModel):
    email: str
    password: str

class RefreshRequest(BaseModel):
    refresh_token: str

class VectorQuery(BaseModel):
    text: str
    k: int = 5

class RawSearch(BaseModel):
    embedding: List[float]
    k: int = 5

class BanditRequest(BaseModel):
    question_id: int
    user_features: Dict[str, float] = {}
    question_features: Dict[str, float] = {}

class OnlineRequest(BaseModel):
    student_id: int
    user_features: Dict[str, float] = {}
    question_features: Dict[str, float] = {}

class EnsembleRequest(BaseModel):
    student_id: int
    question_id: int
    user_features: Dict[str, float] = {}
    question_features: Dict[str, float] = {}
    weights: Optional[Dict[str, float]] = None

class AssignmentCreate(BaseModel):
    title: str
    description: Optional[str] = None
    due_date: Optional[str] = None
    topic: Optional[str] = None

class SubmissionCreate(BaseModel):
    assignment_id: int
    content: str
