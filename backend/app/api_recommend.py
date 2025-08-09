from fastapi import APIRouter, Depends
from app.schemas import BanditRequest, OnlineRequest, EnsembleRequest
from app.core.deps import get_linucb_service, get_ftrl_service
from app.services.bandit.linucb import LinUCBService
from app.services.online.ftrl import FTRLService

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
             ftrl: FTRLService = Depends(get_ftrl_service)):
    b = lin.predict(body.user_features, body.question_features, body.question_id)
    o = ftrl.predict(body.student_id, body.user_features, body.question_features)
    w_b, w_o = 0.5, 0.5
    if body.weights:
        w_b = float(body.weights.get("bandit", w_b))
        w_o = float(body.weights.get("online", w_o))
    final = w_b*b + w_o*o
    return {"final_score": final, "individual_scores": {"bandit": b, "online": o}, "weights_used": {"bandit": w_b, "online": w_o}}
