from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from datetime import datetime, timedelta
from app.core.db import get_db
from app.models import User, RefreshToken
from app.schemas import RegisterRequest, LoginRequest, TokenPair, RefreshRequest
from app.utils.auth import hash_password, verify_password, create_access_token, create_refresh_token, decode_token

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=TokenPair)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(400, "Email already registered")
    u = User(email=body.email, password_hash=hash_password(body.password), full_name=body.full_name, role=body.role or "student")
    db.add(u)
    await db.flush()
    # tokens
    access = create_access_token(str(u.id))
    refresh = create_refresh_token(str(u.id))
    rt = RefreshToken(user_id=u.id, token=refresh, expires_at=datetime.utcnow() + timedelta(days=7))
    db.add(rt)
    await db.commit()
    return TokenPair(access_token=access, refresh_token=refresh)

@router.post("/login", response_model=TokenPair)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(User).where(User.email == body.email))
    u = res.scalar_one_or_none()
    if not u or not verify_password(body.password, u.password_hash):
        raise HTTPException(401, "Invalid credentials")
    access = create_access_token(str(u.id))
    refresh = create_refresh_token(str(u.id))
    rt = RefreshToken(user_id=u.id, token=refresh, expires_at=datetime.utcnow() + timedelta(days=7))
    db.add(rt)
    await db.commit()
    return TokenPair(access_token=access, refresh_token=refresh)

@router.post("/refresh", response_model=TokenPair)
async def refresh(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    try:
        payload = decode_token(body.refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(401, "Invalid token type")
        uid = int(payload["sub"])
    except Exception:
        raise HTTPException(401, "Invalid refresh token")
    # token exists?
    res = await db.execute(select(RefreshToken).where(RefreshToken.token == body.refresh_token))
    t = res.scalar_one_or_none()
    if not t:
        raise HTTPException(401, "Refresh token not found")
    # issue new
    access = create_access_token(str(uid))
    refresh = create_refresh_token(str(uid))
    # rotate
    await db.execute(delete(RefreshToken).where(RefreshToken.id == t.id))
    db.add(RefreshToken(user_id=uid, token=refresh, expires_at=datetime.utcnow() + timedelta(days=7)))
    await db.commit()
    return TokenPair(access_token=access, refresh_token=refresh)
