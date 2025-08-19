from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from app.models.user import UserRole, LearningStyle


class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    role: UserRole = UserRole.STUDENT
    learning_style: LearningStyle = LearningStyle.MIXED


class UserCreate(UserBase):
    password: str = Field(..., min_length=6, max_length=100)

    class Config:
        json_schema_extra = {
            "example": {
                "username": "newuser",
                "email": "newuser@example.com",
                "role": "student",
                "learning_style": "auditory",
                "password": "securepass"
            }
        }


class UserLogin(BaseModel):
    username: str
    password: str

    class Config:
        json_schema_extra = {
            "example": {
                "username": "testuser",
                "password": "testpass"
            }
        }


class UserResponse(UserBase):
    id: str
    current_math_level: int
    current_english_level: int
    is_active: str
    created_at: str
    updated_at: str
    
    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    learning_style: Optional[LearningStyle] = None
    current_math_level: Optional[int] = Field(None, ge=1, le=5)
    current_english_level: Optional[int] = Field(None, ge=1, le=5)


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 3600
            }
        }


class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[str] = None


class PasswordReset(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(..., min_length=6, max_length=100)


class ChangePassword(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=6, max_length=100)


class UserStats(BaseModel):
    total_attempts: int
    correct_attempts: int
    accuracy_rate: float
    average_time_spent: float
    current_streak: int
    last_activity: Optional[str]
    subjects_progress: dict


class BulkUserOperation(BaseModel):
    user_ids: List[str]
    operation: str  # activate, deactivate, delete
    
    
class UserSearchQuery(BaseModel):
    query: Optional[str] = None
    role: Optional[UserRole] = None
    learning_style: Optional[LearningStyle] = None
    min_level: Optional[int] = Field(None, ge=1, le=5)
    max_level: Optional[int] = Field(None, ge=1, le=5)
    is_active: Optional[bool] = None