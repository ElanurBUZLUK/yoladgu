from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.core.db import get_db
from app.core.deps import require_roles
from app.models import Event
from app.services.adaptive import repo as adaptive_repo
from app.core.config import settings
from datetime import datetime

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/metrics/overview")
async def metrics_overview(db: AsyncSession = Depends(get_db), user=Depends(require_roles("admin"))):
    # basit metrikler: event sayıları ve son 24 saat p95 latency placeholder (gerçek ölçüm yok)
    q_total = await db.execute(select(func.count()).select_from(Event))
    total_events = int(q_total.scalar() or 0)
    return {
        "total_events": total_events,
        "latency_p95_ms": None,
        "recall_at_k": None,
        "model_version": None,
    }


@router.post("/rebucket")
async def rebucket(db: AsyncSession = Depends(get_db), user=Depends(require_roles("admin"))):
    from sqlalchemy import select as _select
    from app.models import Question
    qs = (await db.execute(_select(Question))).scalars().all()
    if not qs:
        return {"updated": 0}
    vals = sorted([q.difficulty_rating for q in qs])
    def pct(p: float) -> float:
        if not vals:
            return 0.0
        k = max(0, min(len(vals) - 1, int(round((p/100.0) * (len(vals)-1)))))
        return vals[k]
    pcts = [float(x) for x in settings.REBUCKET_PCT.split(",")]
    p1, p2, p3 = pct(pcts[0]), pct(pcts[1]), pct(pcts[2])
    now = datetime.utcnow()
    updated = 0
    for q in qs:
        old = q.difficulty_level
        if q.difficulty_rating < p1:
            lvl = 1
        elif q.difficulty_rating < p2:
            lvl = 2
        elif q.difficulty_rating < p3:
            lvl = 3
        else:
            lvl = 4
        if lvl != old:
            q.difficulty_level = lvl
            q.last_recalibrated_at = now
            await adaptive_repo.update_question(db, q)
            updated += 1
    return {"updated": updated}
