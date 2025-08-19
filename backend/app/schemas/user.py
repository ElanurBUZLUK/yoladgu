from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.models.user import UserRole, LearningStyle


class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_]+$")
    email: EmailStr
    role: UserRole = UserRole.STUDENT
    current_math_level: int = Field(1, ge=1, le=5)
    current_english_level: int = Field(1, ge=1, le=5)
    learning_style: LearningStyle = LearningStyle.MIXED


class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=128)
    confirm_password: str = Field(..., min_length=8, max_length=128)

    class Config:
        json_schema_extra = {
            "example": {
                "username": "testuser",
                "email": "test@example.com",
                "role": "student",
                "current_math_level": 3,
                "current_english_level": 2,
                "learning_style": "visual",
                "password": "securepassword123",
                "confirm_password": "securepassword123"
            }
        }


class UserUpdate(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_]+$")
    email: Optional[EmailStr] = None
    current_math_level: Optional[int] = Field(None, ge=1, le=5)
    current_english_level: Optional[int] = Field(None, ge=1, le=5)
    learning_style: Optional[LearningStyle] = None


class UserResponse(UserBase):
    id: str
    is_active: str
    created_at: str
    updated_at: str
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "a1b2c3d4e5f6g7h8i9j0k1l2",
                "username": "testuser",
                "email": "test@example.com",
                "role": "student",
                "current_math_level": 3,
                "current_english_level": 2,
                "learning_style": "visual",
                "is_active": True,
                "created_at": "2023-01-01T12:00:00Z",
                "updated_at": "2023-01-01T12:00:00Z"
            }
        }


class UserLogin(BaseModel):
    email: EmailStr
    password: str

    class Config:
        json_schema_extra = {
            "example": {
                "email": "test@example.com",
                "password": "securepassword123"
            }
        }


class UserLoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class UserRefreshToken(BaseModel):
    refresh_token: str


class UserPasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)
    confirm_new_password: str = Field(..., min_length=8, max_length=128)


class UserPasswordReset(BaseModel):
    email: EmailStr


class UserPasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8, max_length=128)
    confirm_new_password: str = Field(..., min_length=8, max_length=128)


class UserLevelUpdate(BaseModel):
    math_level: Optional[int] = Field(None, ge=1, le=5)
    english_level: Optional[int] = Field(None, ge=1, le=5)
    reason: Optional[str] = None


class UserSearchQuery(BaseModel):
    query: Optional[str] = None
    role: Optional[UserRole] = None
    learning_style: Optional[LearningStyle] = None
    min_math_level: Optional[int] = Field(None, ge=1, le=5)
    max_math_level: Optional[int] = Field(None, ge=1, le=5)
    min_english_level: Optional[int] = Field(None, ge=1, le=5)
    max_english_level: Optional[int] = Field(None, ge=1, le=5)
    is_active: Optional[bool] = None


class UserStats(BaseModel):
    total_attempts: int
    correct_attempts: int
    accuracy_rate: float
    avg_difficulty: float
    avg_time_spent: float
    math_level: int
    english_level: int
    learning_style: str
    last_activity: Optional[str] = None


class UserSummary(BaseModel):
    user_id: str
    username: str
    email: str
    role: str
    is_active: str
    stats: UserStats
    created_at: str


class UserBulkOperation(BaseModel):
    user_ids: List[str]
    operation: str  # "activate", "deactivate", "change_role", "update_levels"
    parameters: Optional[Dict[str, Any]] = None


class UserCountByRole(BaseModel):
    role: str
    count: int
    active_count: int
    inactive_count: int


class UserStatsSummary(BaseModel):
    total_users: int
    active_users: int
    users_by_role: List[UserCountByRole]
    users_by_learning_style: Dict[str, int]
    average_math_level: float
    average_english_level: float


class UserValidationResult(BaseModel):
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]


class UserProfile(BaseModel):
    user: UserResponse
    stats: UserStats
    recent_activity: List[Dict[str, Any]]
    achievements: List[Dict[str, Any]]
    preferences: Dict[str, Any]


class UserActivity(BaseModel):
    activity_type: str  # "login", "question_attempt", "level_up", "achievement"
    timestamp: str
    details: Dict[str, Any]
    subject: Optional[str] = None
    difficulty_level: Optional[int] = None


class UserAchievement(BaseModel):
    achievement_id: str
    name: str
    description: str
    icon: str
    earned_at: str
    progress: Optional[float] = None
    max_progress: Optional[int] = None


class UserPreferences(BaseModel):
    notification_settings: Dict[str, bool]
    privacy_settings: Dict[str, bool]
    learning_preferences: Dict[str, Any]
    ui_preferences: Dict[str, Any]


class UserNotification(BaseModel):
    notification_id: str
    type: str  # "info", "warning", "success", "error"
    title: str
    message: str
    created_at: str
    read: bool
    action_url: Optional[str] = None
