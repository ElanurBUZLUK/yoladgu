from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
from app.core.config import settings

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(p: str) -> str:
    return pwd_ctx.hash(p)

def verify_password(p: str, hashed: str) -> bool:
    return pwd_ctx.verify(p, hashed)

def create_access_token(sub: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MIN)
    to_encode = {"sub": sub, "exp": expire, "type": "access"}
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALG)

def create_refresh_token(sub: str) -> str:
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode = {"sub": sub, "exp": expire, "type": "refresh"}
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALG)

def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALG])
