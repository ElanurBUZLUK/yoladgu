from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert
from app.schemas import BanditRequest, OnlineRequest, EnsembleRequest
from app.core.deps import get_linucb_service, get_ftrl_service, require_roles
from app.core.dependencies import get_cf_model
from app.services.bandit.linucb import LinUCBService
from app.services.online.ftrl import FTRLService
from app.core.db import get_db
from app.models import Event, User

router = APIRouter(prefix="/recommend", tags=["recommend"])

@router.post("/bandit")
def bandit_score(body: BanditRequest, svc: LinUCBService = Depends(get_linucb_service)):
    score = svc.predict(body.user_features, body.question_features, body.question_id)
    return {"score": score}

@router.post("/bandit/update")
def bandit_update(body: BanditRequest, reward: float, svc: LinUCBService = Depends(get_linucb_service)):
    svc.update(body.user_features, body.question_features, body.question_id, reward)
    return {"ok": True}

@router.post("/online")
def online_score(body: OnlineRequest, svc: FTRLService = Depends(get_ftrl_service)):
    score = svc.predict(body.student_id, body.user_features, body.question_features)
    return {"score": score}

@router.post("/online/update")
def online_update(body: OnlineRequest, label: int, svc: FTRLService = Depends(get_ftrl_service)):
    svc.update(label, body.user_features, body.question_features)
    return {"ok": True}

@router.post("/ensemble")
def ensemble(body: EnsembleRequest,
             lin: LinUCBService = Depends(get_linucb_service),
             ftrl: FTRLService = Depends(get_ftrl_service),
             db: AsyncSession = Depends(get_db),
             user: User = Depends(require_roles("student"))):
    b = lin.predict(body.user_features, body.question_features, body.question_id)
    o = ftrl.predict(body.student_id, body.user_features, body.question_features)
    try:
        cf = get_cf_model().score(body.student_id, body.question_id)
    except Exception:
        cf = 0.0
    # extend with CF/retrieval if available later; currently weight by config
    from app.core.config import settings
    w_cf = float(settings.W_CF)
    w_b = float(settings.W_BANDIT)
    w_o = float(settings.W_ONLINE)
    w_r = float(getattr(settings, "W_RETR", 0.0))
    if body.weights:
        w_b = float(body.weights.get("bandit", w_b))
        w_o = float(body.weights.get("online", w_o))
        if "cf" in body.weights: w_cf = float(body.weights["cf"])
        if "retr" in body.weights: w_r = float(body.weights["retr"])
    final = w_cf*cf + w_b*b + w_o*o + w_r*0.0
    # exposure event
    await db.execute(insert(Event).values(user_id=user.id, event_type="question_exposure", payload=str({"question_id": body.question_id, "scores": {"bandit": b, "online": o}, "final": final})))
    await db.commit()
    return {"final_score": final, "individual_scores": {"cf": cf, "bandit": b, "online": o, "retr": 0.0}, "weights_used": {"cf": w_cf, "bandit": w_b, "online": w_o, "retr": w_r}}
