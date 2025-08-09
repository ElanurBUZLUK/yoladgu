from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.core.db import get_db
from app.core.deps import require_roles
from app.models import Event

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


