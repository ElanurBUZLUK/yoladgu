from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert
from pydantic import BaseModel
from typing import Dict
from app.core.deps import get_linucb_service, get_ftrl_service, require_roles
from app.services.bandit.linucb import LinUCBService
from app.services.online.ftrl import FTRLService
from app.core.db import get_db
from app.models import Event, User
import json


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
async def linucb_update(
    body: BanditUpdate,
    svc: LinUCBService = Depends(get_linucb_service),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles("student")),
):
    svc.update(body.user_features, body.question_features, body.question_id, body.reward)
    await db.execute(
        insert(Event).values(
            user_id=user.id,
            event_type="linucb_update",
            payload=json.dumps(
                {
                    "question_id": body.question_id,
                    "reward": body.reward,
                    "user_features": body.user_features,
                    "question_features": body.question_features,
                }
            ),
        )
    )
    await db.commit()
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
async def ftrl_update(
    body: OnlineUpdate,
    svc: FTRLService = Depends(get_ftrl_service),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles("student")),
):
    svc.update(body.label, body.user_features, body.question_features)
    await db.execute(
        insert(Event).values(
            user_id=user.id,
            event_type="online_update",
            payload=json.dumps(
                {
                    "label": body.label,
                    "user_features": body.user_features,
                    "question_features": body.question_features,
                }
            ),
        )
    )
    await db.commit()
    return {"ok": True}


