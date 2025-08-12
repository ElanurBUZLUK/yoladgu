from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.core.db import get_db
from app.core.deps import require_roles, get_index_manager
from app.models import Event
from app.core.config import settings
from app.services.adaptive import repo as adaptive_repo
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import subprocess
from app.core.deps import get_index_manager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import subprocess

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/metrics/overview")
async def metrics_overview(db: AsyncSession = Depends(get_db), user=Depends(require_roles("admin"))):
    # basit metrikler: event sayıları ve son 24 saat p95 latency placeholder (gerçek ölçüm yok)
    q_total = await db.execute(select(func.count()).select_from(Event))
    total_events = int(q_total.scalar() or 0)
    # aktif index ve embed bilgisi
    try:
        from app.core.deps import get_index_manager
        mgr = get_index_manager()
        index_stats = mgr.stats()
    except Exception:
        index_stats = {}
    embed_info = {
        "embed_dim": settings.EMBED_DIM,
        "embedding_provider": getattr(settings, "EMBEDDING_PROVIDER", None),
        "embedding_model": getattr(settings, "EMBEDDING_MODEL", None),
    }
    return {
        "total_events": total_events,
        "latency_p95_ms": None,
        "recall_at_k": None,
        "model_version": None,
        "index": index_stats,
        "embedding": embed_info,
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


@router.get("/metrics/events/by_type")
async def events_by_type(db: AsyncSession = Depends(get_db), user=Depends(require_roles("admin"))):
    # tüm zamanlar için event_type bazında sayım
    rows = (await db.execute(select(Event.event_type, func.count()).group_by(Event.event_type))).all()
    return {"counts": {et: int(c) for et, c in rows}}


@router.get("/metrics/events/by_type_24h")
async def events_by_type_24h(db: AsyncSession = Depends(get_db), user=Depends(require_roles("admin"))):
    # son 24 saat için event_type bazında sayım (Event.created_at UTC varsayımıyla)
    since = datetime.utcnow() - timedelta(hours=24)
    rows = (
        await db.execute(
            select(Event.event_type, func.count())
            .where(Event.created_at >= since)
            .group_by(Event.event_type)
        )
    ).all()
    return {"since": since.isoformat() + "Z", "counts": {et: int(c) for et, c in rows}}


# --- Minimal Scheduler-based batch pipelines (MVP) ---
_scheduler: AsyncIOScheduler | None = None


def _ensure_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler()
        _scheduler.start()
    return _scheduler


def _run_cmd(cmd: list[str]) -> tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return 0, out.decode("utf-8", errors="ignore")
    except subprocess.CalledProcessError as e:
        return e.returncode, e.output.decode("utf-8", errors="ignore")


@router.post("/jobs/cf_retrain")
async def jobs_cf_retrain(user=Depends(require_roles("admin"))):
    # Ensure export first
    _run_cmd(["python", "tools/export_interactions.py", "--out", "examples/interactions.jsonl", "--lookback_days", "180"])
    code, out = _run_cmd(["python", "tools/train_cf.py", "--data", "examples/interactions.jsonl"])
    return {"ok": code == 0, "code": code, "output": out}


@router.post("/jobs/index_build_swap")
async def jobs_index_build_swap(user=Depends(require_roles("admin"))):
    code, out = _run_cmd(["python", "tools/batch_indexer.py"])
    swap_res = {}
    swapped = False
    if code == 0:
        mgr = get_index_manager()
        swap_res = mgr.swap()
        swapped = bool(swap_res.get("swapped"))
    return {"ok": (code == 0 and swapped), "build": {"code": code, "output": out}, "swap": swap_res}


@router.post("/jobs/schedule")
async def jobs_schedule(cron_cf: str = "0 2 * * *", cron_index: str = "0 3 * * *", user=Depends(require_roles("admin"))):
    sched = _ensure_scheduler()
    for job in list(sched.get_jobs()):
        sched.remove_job(job.id)
    def _cf_job():
        _run_cmd(["python", "tools/export_interactions.py", "--out", "examples/interactions.jsonl", "--lookback_days", "180"])
        _run_cmd(["python", "tools/train_cf.py", "--data", "examples/interactions.jsonl"])
    sched.add_job(_cf_job, CronTrigger.from_crontab(cron_cf), id="cf_retrain")
    def _index_job():
        _run_cmd(["python", "tools/batch_indexer.py"])
        try:
            mgr = get_index_manager()
            mgr.swap()
        except Exception:
            pass
    sched.add_job(_index_job, CronTrigger.from_crontab(cron_index), id="index_build_swap")
    return {"ok": True, "scheduled": {"cf_retrain": cron_cf, "index_build_swap": cron_index}}


# --- Ensemble Weights Admin ---
from app.services.ensemble_weights import get_ensemble_weights_service


@router.get("/ensemble/variants")
async def ensemble_list(db: AsyncSession = Depends(get_db), user=Depends(require_roles("admin"))):
    svc = get_ensemble_weights_service()
    items = await svc.list_variants(db)
    return {"variants": [{"variant": v.name, "weights": v.weights, "is_active": v.is_active} for v in items]}


@router.put("/ensemble/variants/{variant}")
async def ensemble_upsert(variant: str, weights: dict, is_active: bool | None = None, db: AsyncSession = Depends(get_db), user=Depends(require_roles("admin"))):
    svc = get_ensemble_weights_service()
    await svc.upsert_variant(db, variant, weights, is_active)
    return {"ok": True}


@router.post("/ensemble/variants/{variant}/activate")
async def ensemble_activate(variant: str, db: AsyncSession = Depends(get_db), user=Depends(require_roles("admin"))):
    svc = get_ensemble_weights_service()
    await svc.set_active(db, variant)
    return {"ok": True, "active": variant}


# --- Policy Variants Admin ---
from sqlalchemy import select, update, insert
from app.models import PolicyVariant, User
from app.schemas import PolicyVariantUpsert, PolicyAssignRequest


@router.get("/policy/variants")
async def policy_list(db: AsyncSession = Depends(get_db), user=Depends(require_roles("admin"))):
    rows = (await db.execute(select(PolicyVariant))).scalars().all()
    return {
        "variants": [
            {
                "id": v.id,
                "name": v.name,
                "description": v.description,
                "parameters": v.parameters,
                "assignment_rule": v.assignment_rule,
                "is_active": v.is_active,
            }
            for v in rows
        ]
    }


@router.put("/policy/variants/{name}")
async def policy_upsert(name: str, body: PolicyVariantUpsert, db: AsyncSession = Depends(get_db), user=Depends(require_roles("admin"))):
    existing = (await db.execute(select(PolicyVariant).where(PolicyVariant.name == name))).scalar_one_or_none()
    if existing is None:
        await db.execute(
            insert(PolicyVariant).values(
                name=name,
                description=body.description,
                parameters=body.parameters,
                assignment_rule=body.assignment_rule,
                is_active=bool(body.is_active),
            )
        )
    else:
        await db.execute(
            update(PolicyVariant)
            .where(PolicyVariant.name == name)
            .values(
                description=body.description,
                parameters=body.parameters,
                assignment_rule=body.assignment_rule,
                is_active=bool(body.is_active),
            )
        )
    await db.commit()
    return {"ok": True}


@router.post("/policy/variants/{name}/activate")
async def policy_activate(name: str, db: AsyncSession = Depends(get_db), user=Depends(require_roles("admin"))):
    await db.execute(update(PolicyVariant).values(is_active=False))
    await db.execute(update(PolicyVariant).where(PolicyVariant.name == name).values(is_active=True))
    await db.commit()
    return {"ok": True, "active": name}


@router.post("/policy/assign")
async def policy_assign(body: PolicyAssignRequest, db: AsyncSession = Depends(get_db), user=Depends(require_roles("admin"))):
    # Assign cohort by variant name
    await db.execute(
        update(User)
        .where(User.id.in_(body.user_ids))
        .values(experiment_cohort=body.variant_name)
    )
    await db.commit()
    return {"ok": True, "assigned": len(body.user_ids), "cohort": body.variant_name}
