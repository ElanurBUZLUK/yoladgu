from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert
from app.core.deps import require_roles, get_linucb_service, get_ftrl_service
from app.services.bandit.linucb import LinUCBService
from app.services.online.ftrl import FTRLService
from app.core.db import get_db
from app.models import Event, User
from app.schemas import EnsembleRequest

router = APIRouter(prefix="/questions", tags=["questions"])


@router.post("/next")
async def questions_next(
    body: EnsembleRequest,
    lin: LinUCBService = Depends(get_linucb_service),
    ftrl: FTRLService = Depends(get_ftrl_service),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles("student")),
):
    b = lin.predict(body.user_features, body.question_features, body.question_id)
    o = ftrl.predict(body.student_id, body.user_features, body.question_features)
    final = 0.5*b + 0.5*o
    # log exposure
    await db.execute(insert(Event).values(user_id=user.id, event_type="question_exposure", payload=str({"question_id": body.question_id, "score": final})))
    await db.commit()
    return {"question_id": body.question_id, "score": final}


@router.post("/submit")
async def questions_submit(
    question_id: int,
    correct: bool,
    lin: LinUCBService = Depends(get_linucb_service),
    ftrl: FTRLService = Depends(get_ftrl_service),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_roles("student")),
):
    reward = 1.0 if correct else 0.0
    # basit güncelleme: sabit uf/qf boş
    lin.update({}, {}, question_id, reward)
    ftrl.update(int(reward), {}, {})
    await db.execute(insert(Event).values(user_id=user.id, event_type="question_outcome", payload=str({"question_id": question_id, "correct": correct})))
    await db.commit()
    return {"ok": True}


