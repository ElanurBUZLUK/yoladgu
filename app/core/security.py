from datetime import datetime, timedelta
from typing import Optional, Union

from app.core.config import settings
from app.db.database import get_db
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


def create_access_token(
    data: dict, expires_delta: Union[timedelta, None] = None
) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def decode_access_token(token: str) -> Optional[dict]:
    """
    JWT token'ı decode eder ve payload'ı döndürür.
    Token geçersiz veya expired ise None döndürür.
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        return payload
    except JWTError:
        return None


def is_token_expired(token: str) -> bool:
    """
    Token'ın expire olup olmadığını kontrol eder.
    Token geçersiz ise True döndürür.
    """
    payload = decode_access_token(token)
    if payload is None:
        return True

    exp = payload.get("exp")
    if exp is None:
        return True

    return datetime.utcnow().timestamp() > exp


def get_token_username(token: str) -> Optional[str]:
    """
    Token'dan username'i çıkarır.
    """
    payload = decode_access_token(token)
    if payload is None:
        return None

    return payload.get("sub")


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    """
    Get current authenticated user from JWT token
    Queries real user from database instead of returning fake user
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        email: str = payload.get("sub")
        user_id: int = payload.get("user_id")

        if email is None:
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    # Query real user from database
    from app.crud.user import get_user, get_user_by_email

    # Try to get user by ID first (more efficient), fallback to email
    user = None
    if user_id:
        user = get_user(db, user_id=user_id)

    if not user and email:
        user = get_user_by_email(db, email=email)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user"
        )

    return user
