import os, time
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from jose import jwt, JWTError
from redis import Redis

router = APIRouter(prefix="/auth", tags=["auth"])

ALGO = "HS256"
JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
ACCESS_TTL_MIN = int(os.getenv("ACCESS_TTL_MIN", 30))
REFRESH_TTL_DAYS = int(os.getenv("REFRESH_TTL_DAYS", 10))

redis = Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"))

class LoginIn(BaseModel):
    user_id: str

class TokenOut(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

def _mint_access(sub: str) -> str:
    exp = int(time.time()) + ACCESS_TTL_MIN * 60
    return jwt.encode({"sub": sub, "exp": exp, "typ": "access"}, JWT_SECRET, algorithm=ALGO)

def _mint_refresh(sub: str, jti: str) -> str:
    exp = int(time.time()) + REFRESH_TTL_DAYS * 86400
    return jwt.encode({"sub": sub, "exp": exp, "typ": "refresh", "jti": jti}, JWT_SECRET, algorithm=ALGO)

@router.post("/login", response_model=TokenOut)
def login(body: LoginIn):
    jti = f"rf_{int(time.time()*1000)}_{body.user_id}"
    refresh = _mint_refresh(body.user_id, jti)
    access = _mint_access(body.user_id)
    redis.setex(f"rfw:{jti}", REFRESH_TTL_DAYS*86400, "1")
    return TokenOut(access_token=access, refresh_token=refresh)

class RefreshIn(BaseModel):
    refresh_token: str

@router.post("/refresh", response_model=TokenOut)
def refresh_token(body: RefreshIn):
    try:
        payload = jwt.decode(body.refresh_token, JWT_SECRET, algorithms=[ALGO])
        if payload.get("typ") != "refresh":
            raise JWTError("not refresh token")
        jti = payload.get("jti")
        if not jti or not redis.get(f"rfw:{jti}"):
            raise JWTError("refresh not whitelisted or revoked")
        sub = payload["sub"]
        redis.delete(f"rfw:{jti}")
        redis.setex(f"rblk:{jti}", REFRESH_TTL_DAYS*86400, "1")
        new_jti = f"rf_{int(time.time()*1000)}_{sub}"
        new_refresh = _mint_refresh(sub, new_jti)
        redis.setex(f"rfw:{new_jti}", REFRESH_TTL_DAYS*86400, "1")
        new_access = _mint_access(sub)
        return TokenOut(access_token=new_access, refresh_token=new_refresh)
    except JWTError as e:
        raise HTTPException(401, detail=f"Invalid refresh token: {str(e)}")

class LogoutIn(BaseModel):
    refresh_token: str

@router.post("/logout")
def logout(body: LogoutIn):
    try:
        payload = jwt.decode(body.refresh_token, JWT_SECRET, algorithms=[ALGO])
        if payload.get("typ") != "refresh":
            raise JWTError("not refresh token")
        jti = payload.get("jti")
        if jti:
            redis.delete(f"rfw:{jti}")
            redis.setex(f"rblk:{jti}", REFRESH_TTL_DAYS*86400, "1")
        return {"status": "ok"}
    except JWTError:
        return {"status": "ok"}