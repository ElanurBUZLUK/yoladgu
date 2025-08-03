from datetime import datetime
from typing import Optional

from app.db.models import UserRole
from pydantic import BaseModel, ConfigDict, EmailStr


class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: str
    grade: Optional[str] = None
    role: UserRole = UserRole.STUDENT
    is_active: bool = True


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    grade: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class ProgressUpdate(BaseModel):
    level: Optional[float] = None
    total_questions_answered: Optional[int] = None
    total_correct_answers: Optional[int] = None
    average_response_time: Optional[float] = None


class User(UserBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
