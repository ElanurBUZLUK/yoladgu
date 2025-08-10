from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict
from app.core.deps import get_linucb_service, get_ftrl_service, require_roles
from app.services.bandit.linucb import LinUCBService
from app.services.online.ftrl import FTRLService


router = APIRouter(prefix="/ml", tags=["ml"])


class BanditPredict(BaseModel):
    student_id: int
    question_id: int
    user_features: Dict[str, float] = {}
    question_features: Dict[str, float] = {}


class BanditUpdate(BaseModel):
    question_id: int
    user_features: Dict[str, float] = {}
    question_features: Dict[str, float] = {}
    reward: float


class OnlinePredict(BaseModel):
    student_id: int
    user_features: Dict[str, float] = {}
    question_features: Dict[str, float] = {}


class OnlineUpdate(BaseModel):
    student_id: int
    user_features: Dict[str, float] = {}
    question_features: Dict[str, float] = {}
    label: int  # 0/1


@router.post("/linucb/predict")
def linucb_predict(
    body: BanditPredict,
    svc: LinUCBService = Depends(get_linucb_service),
    _user=Depends(require_roles("student")),
):
    score = svc.predict(body.user_features, body.question_features, body.question_id)
    return {"score": score}


@router.post("/linucb/update")
def linucb_update(
    body: BanditUpdate,
    svc: LinUCBService = Depends(get_linucb_service),
    _user=Depends(require_roles("student")),
):
    svc.update(body.user_features, body.question_features, body.question_id, body.reward)
    return {"ok": True}


@router.post("/ftrl/predict")
def ftrl_predict(
    body: OnlinePredict,
    svc: FTRLService = Depends(get_ftrl_service),
    _user=Depends(require_roles("student")),
):
    score = svc.predict(body.student_id, body.user_features, body.question_features)
    return {"score": score}


@router.post("/ftrl/update")
def ftrl_update(
    body: OnlineUpdate,
    svc: FTRLService = Depends(get_ftrl_service),
    _user=Depends(require_roles("student")),
):
    svc.update(body.label, body.user_features, body.question_features)
    return {"ok": True}


