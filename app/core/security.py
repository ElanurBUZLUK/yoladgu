from datetime import datetime, timedelta
from typing import Optional, Union

from app.core.config import settings
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

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


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
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
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # Burada gerçek bir database bağlantısı olmalı
    # Şimdilik mock bir kullanıcı döndürüyoruz
    from app.db.models import User, UserRole

    user = User(
        id=1,
        first_name="Test",
        last_name="User",
        email=email,
        password_hash="",
        role=UserRole.STUDENT,
    )
    return user
