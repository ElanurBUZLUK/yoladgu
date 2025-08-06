"""
Token Schemas
"""

from pydantic import BaseModel
from typing import Optional


class Token(BaseModel):
    """Token schema"""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Token data schema"""
    username: Optional[str] = None


class LoginRequest(BaseModel):
    """Login request schema"""
    username: str
    password: str


class RegisterRequest(BaseModel):
    """Register request schema"""
    username: str
    email: str
    password: str
    full_name: str


class TokenResponse(BaseModel):
    """Token response schema"""
    access_token: str
    token_type: str
    user_id: int
    username: str
    email: str
    role: str


__all__ = [
    "Token",
    "TokenData",
    "LoginRequest", 
    "RegisterRequest",
    "TokenResponse"
]
