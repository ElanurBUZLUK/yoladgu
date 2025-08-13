from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from app.services.en_qg.service import EnQGService
from app.services.en_qg.schemas import GeneratedQuestion, SubmitIn, NextParams
from app.core.config import settings
from app.core.rate_limit import SlidingWindowRateLimiter


router = APIRouter(prefix="/en", tags=["english-qg"])

_lim_next = SlidingWindowRateLimiter(max_calls=5, window_seconds=60)


@router.post("/submit")
def en_submit(body: SubmitIn):
    svc = EnQGService()
    return svc.submit(body)


@router.get("/next", response_model=GeneratedQuestion)
def en_next(request: Request, student_id: int, mode: str | None = Query(None), cefr: str | None = None, k_ctx: int = 5):
    if not getattr(settings, "EN_QG_ENABLED", False):
        raise HTTPException(400, "EN_QG is disabled")
    ip = request.client.host if request and request.client else "unknown"
    if not _lim_next.allow(f"next:{ip}"):
        raise HTTPException(429, "Too Many Requests")
    eff_mode = mode or getattr(settings, "EN_QG_MODE_DEFAULT", "mixed")
    svc = EnQGService()
    return svc.next(NextParams(student_id=student_id, mode=eff_mode, cefr=cefr, k_ctx=k_ctx))


