"""
User DTOs
Kullanıcı veri transfer objeleri
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"
    MODERATOR = "moderator"

class UserStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"

class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    first_name: str = Field(..., min_length=2, max_length=50)
    last_name: str = Field(..., min_length=2, max_length=50)
    role: UserRole = UserRole.STUDENT
    grade_level: Optional[int] = Field(None, ge=1, le=12)
    school: Optional[str] = None
    interests: Optional[List[str]] = []
    learning_goals: Optional[List[str]] = []

class UserUpdate(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    first_name: Optional[str] = Field(None, min_length=2, max_length=50)
    last_name: Optional[str] = Field(None, min_length=2, max_length=50)
    grade_level: Optional[int] = Field(None, ge=1, le=12)
    school: Optional[str] = None
    interests: Optional[List[str]] = []
    learning_goals: Optional[List[str]] = []
    bio: Optional[str] = None
    avatar_url: Optional[str] = None

class UserProfile(BaseModel):
    id: int
    email: EmailStr
    username: str
    first_name: str
    last_name: str
    role: UserRole
    status: UserStatus
    grade_level: Optional[int] = None
    school: Optional[str] = None
    interests: List[str] = []
    learning_goals: List[str] = []
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    total_points: int = 0
    achievements_count: int = 0
    rank: Optional[int] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str
    remember_me: bool = False

class UserResponse(BaseModel):
    user: UserProfile
    token: str
    refresh_token: str
    expires_in: int

class UserList(BaseModel):
    users: List[UserProfile]
    total: int
    page: int
    per_page: int
    total_pages: int

class UserStats(BaseModel):
    total_users: int
    active_users: int
    new_users_today: int
    new_users_this_week: int
    users_by_role: Dict[str, int]
    users_by_grade: Dict[str, int]

class UserActivity(BaseModel):
    user_id: int
    activity_type: str
    description: str
    metadata: Dict[str, Any] = {}
    timestamp: datetime

class UserPreferences(BaseModel):
    user_id: int
    notification_email: bool = True
    notification_push: bool = True
    notification_sms: bool = False
    privacy_profile_public: bool = True
    privacy_show_progress: bool = True
    privacy_show_achievements: bool = True
    language: str = "tr"
    timezone: str = "Europe/Istanbul"
    theme: str = "light"

class UserSearch(BaseModel):
    query: Optional[str] = None
    role: Optional[UserRole] = None
    status: Optional[UserStatus] = None
    grade_level: Optional[int] = None
    school: Optional[str] = None
    interests: Optional[List[str]] = None
    page: int = 1
    per_page: int = 20
    sort_by: str = "created_at"
    sort_order: str = "desc"
