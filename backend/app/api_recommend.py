from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert
from app.schemas import BanditRequest, OnlineRequest, EnsembleRequest
from app.core.deps import get_linucb_service, get_ftrl_service, require_roles, get_peer_service
from app.core.dependencies import get_cf_model
from app.services.bandit.linucb import LinUCBService
from app.services.online.ftrl import FTRLService
from app.services.peer_hardness import PeerHardnessService
from app.core.db import get_db
from app.models import Event, User
from app.services.ensemble_weights import get_ensemble_weights_service
import json
import time
from app.utils.metrics import rec_latency, rec_scores

router = APIRouter(prefix="/recommend", tags=["recommend"])

@router.post("/bandit")
def bandit_score(body: BanditRequest, svc: LinUCBService = Depends(get_linucb_service)):
    score = svc.predict(body.user_features, body.question_features, body.question_id)
    return {"score": score}

@router.post("/bandit/update")
async def bandit_update(
    body: BanditRequest,
    reward: float,
    svc: LinUCBService = Depends(get_linucb_service),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles("student")),
):
    svc.update(body.user_features, body.question_features, body.question_id, reward)
    # event log
    await db.execute(
        insert(Event).values(
            user_id=user.id,
            event_type="bandit_update",
            payload=json.dumps(
                {
                    "question_id": body.question_id,
                    "reward": reward,
                    "user_features": body.user_features,
                    "question_features": body.question_features,
                }
            ),
        )
    )
    await db.commit()
    return {"ok": True}

@router.post("/online")
def online_score(body: OnlineRequest, svc: FTRLService = Depends(get_ftrl_service)):
    score = svc.predict(body.student_id, body.user_features, body.question_features)
    return {"score": score}

@router.post("/online/update")
async def online_update(
    body: OnlineRequest,
    label: int,
    svc: FTRLService = Depends(get_ftrl_service),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles("student")),
):
    svc.update(label, body.user_features, body.question_features)
    # event log
    await db.execute(
        insert(Event).values(
            user_id=user.id,
            event_type="online_update",
            payload=json.dumps(
                {
                    "student_id": body.student_id,
                    "label": label,
                    "user_features": body.user_features,
                    "question_features": body.question_features,
                }
            ),
        )
    )
    await db.commit()
    return {"ok": True}

@router.post("/ensemble")
async def ensemble(body: EnsembleRequest,
                   lin: LinUCBService = Depends(get_linucb_service),
                   ftrl: FTRLService = Depends(get_ftrl_service),
                   peer: PeerHardnessService = Depends(get_peer_service),
                   db: AsyncSession = Depends(get_db),
                   user: User = Depends(require_roles("student"))):
    t0 = time.time()
    b = lin.predict(body.user_features, body.question_features, body.question_id)
    o = ftrl.predict(body.student_id, body.user_features, body.question_features)
    try:
        cf = get_cf_model().score(body.student_id, body.question_id)
    except Exception:
        cf = 0.0
    # optional served model score
    s_served = 0.0
    try:
        from app.core.config import settings
        if getattr(settings, "SERVING_PROVIDER", "none") == "ts":
            from app.services.online.ts_client import TorchServeClient
            ts = TorchServeClient()
            s_served = float(ts.predict({
                "student_id": body.student_id,
                "question_id": body.question_id,
                "user_features": body.user_features,
                "question_features": body.question_features,
            }))
    except Exception:
        s_served = 0.0
    # Determine weights: request override > user-assigned variant > defaults
    svc_w = get_ensemble_weights_service()
    variant, weights = await svc_w.get_effective_weights(db, user.id, override=body.weights)
    w_cf = float(weights.get("cf", 0.0))
    w_b = float(weights.get("bandit", 0.0))
    w_o = float(weights.get("online", 0.0))
    w_r = float(weights.get("retr", 0.0))
    w_p = float(weights.get("peer", 0.0))
    w_s = float(weights.get("served", 0.0))

    # optional peer score
    try:
        p_peer = float(peer.peer_score_for(body.student_id, body.question_id)) if w_p > 0.0 else 0.0
    except Exception:
        p_peer = 0.0

    final = w_cf*cf + w_b*b + w_o*o + w_r*0.0 + w_p*p_peer + w_s*s_served
    # exposure event
    await db.execute(insert(Event).values(user_id=user.id, event_type="question_exposure", payload=json.dumps({"question_id": body.question_id, "scores": {"bandit": b, "online": o, "peer": p_peer, "cf": cf, "served": s_served}, "final": final, "variant": variant, "weights": weights})))
    await db.commit()
    # metrics
    try:
        rec_scores.labels("bandit").observe(float(b))
        rec_scores.labels("online").observe(float(o))
        rec_scores.labels("cf").observe(float(cf))
        rec_scores.labels("peer").observe(float(p_peer))
        rec_scores.labels("served").observe(float(s_served))
        rec_latency.observe(time.time() - t0)
    except Exception:
        pass
    return {"final_score": final, "individual_scores": {"cf": cf, "bandit": b, "online": o, "retr": 0.0, "peer": p_peer, "served": s_served}, "weights_used": {"cf": w_cf, "bandit": w_b, "online": w_o, "retr": w_r, "peer": w_p, "served": w_s}, "variant": variant}
