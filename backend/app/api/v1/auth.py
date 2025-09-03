"""
Authentication endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer

from app.models.request import LoginRequest, RegisterRequest
from app.models.response import TokenResponse, UserResponse

router = APIRouter()
security = HTTPBearer()


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """User login endpoint."""
    # TODO: Implement authentication logic
    # - Validate credentials against database
    # - Generate JWT tokens
    # - Return access and refresh tokens
    
    raise HTTPException(
        status_code=501,
        detail="Authentication not implemented yet"
    )


@router.post("/register", response_model=UserResponse)
async def register(request: RegisterRequest):
    """User registration endpoint."""
    # TODO: Implement user registration
    # - Validate input data
    # - Check if user already exists
    # - Create new user in database
    # - Return user information
    
    raise HTTPException(
        status_code=501,
        detail="User registration not implemented yet"
    )


@router.post("/refresh")
async def refresh_token():
    """Refresh access token."""
    # TODO: Implement token refresh
    # - Validate refresh token
    # - Generate new access token
    # - Return new tokens
    
    raise HTTPException(
        status_code=501,
        detail="Token refresh not implemented yet"
    )