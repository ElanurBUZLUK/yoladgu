from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, update, insert
from datetime import datetime, timedelta
from app.core.db import get_db
from app.models import User, RefreshToken
from app.schemas import RegisterRequest, LoginRequest, TokenPair, RefreshRequest
from app.utils.auth import hash_password, verify_password, create_access_token, create_refresh_token, decode_token
from app.core.rate_limit import SlidingWindowRateLimiter
import re

router = APIRouter(prefix="/auth", tags=["auth"])
rl_login = SlidingWindowRateLimiter(max_calls=10, window_seconds=60)
rl_register = SlidingWindowRateLimiter(max_calls=5, window_seconds=60)
rl_forgot = SlidingWindowRateLimiter(max_calls=5, window_seconds=60)

@router.post("/register", response_model=TokenPair)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    if not rl_register.allow(f"register:{body.email}"):
        raise HTTPException(429, "rate limited")
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
    if not rl_login.allow(f"login:{body.email}"):
        raise HTTPException(429, "rate limited")
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
    # token exists and not expired?
    res = await db.execute(select(RefreshToken).where(RefreshToken.token == body.refresh_token))
    t = res.scalar_one_or_none()
    if not t or t.expires_at < datetime.utcnow():
        raise HTTPException(401, "Refresh token invalid or expired")
    # issue new and rotate (prevent reuse): delete old, insert new
    access = create_access_token(str(uid))
    refresh = create_refresh_token(str(uid))
    await db.execute(delete(RefreshToken).where(RefreshToken.id == t.id))
    await db.execute(insert(RefreshToken).values(user_id=uid, token=refresh, expires_at=datetime.utcnow() + timedelta(days=7)))
    await db.commit()
    return TokenPair(access_token=access, refresh_token=refresh)


@router.post("/forgot")
async def forgot(email: str, db: AsyncSession = Depends(get_db)):
    if not rl_forgot.allow(f"forgot:{email}"):
        raise HTTPException(429, "rate limited")
    # basit doğrulama
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email or ""):
        raise HTTPException(400, "invalid email")
    # not: burada gerçek e-posta gönderimi yerine log/placeholder dönüyoruz
    res = await db.execute(select(User).where(User.email == email))
    u = res.scalar_one_or_none()
    if not u:
        # kullanıcı yoksa bile başarılı dön (enumaration önleme)
        return {"ok": True}
    # create reset token
    import secrets
    token = secrets.token_urlsafe(32)
    from sqlalchemy import insert as _insert
    await db.execute(
        _insert(__import__("app.models", fromlist=["password_resets"]).password_resets).values(  # type: ignore
            user_id=u.id,
            token=token,
            expires_at=datetime.utcnow() + timedelta(hours=2),
        )
    )
    await db.commit()
    # Here, send email with token link e.g., https://app/reset?token=...
    return {"ok": True}


@router.post("/reset")
async def reset_password(token: str, new_password: str, db: AsyncSession = Depends(get_db)):
    # validate token
    from sqlalchemy import text as _text
    q = await db.execute(_text("SELECT id, user_id, expires_at, used FROM password_resets WHERE token = :t"), {"t": token})
    row = q.first()
    if not row:
        raise HTTPException(400, "invalid token")
    rid, uid, exp, used = int(row[0]), int(row[1]), row[2], bool(row[3])
    if used or exp < datetime.utcnow():
        raise HTTPException(400, "expired or used token")
    # update password and mark token used
    await db.execute(update(User).where(User.id == uid).set({User.password_hash: hash_password(new_password)}))
    await db.execute(_text("UPDATE password_resets SET used = true WHERE id = :rid"), {"rid": rid})
    await db.commit()
    return {"ok": True}
