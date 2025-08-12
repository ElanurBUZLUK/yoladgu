from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert
import json
from app.core.db import get_db
from app.core.config import settings
from app.schemas_adaptive import (
    ServeRequest,
    ServeResponse,
    ServeResponseItem,
    SubmitRequest,
    SubmitResponse,
)
from app.services.adaptive.rating_service import RatingService
from app.services.adaptive.selector_service import SelectorService
from app.services.adaptive import repo
from app.models import Event


router = APIRouter(prefix="/adaptive", tags=["adaptive"])
_RS = RatingService(settings)
_SS = SelectorService(settings)


@router.post("/serve-question", response_model=ServeResponse)
async def serve_question(payload: ServeRequest, db: AsyncSession = Depends(get_db)):
    student = await repo.get_student(db, payload.student_id)
    if not student:
        raise HTTPException(404, "student not found")
    chosen = await _SS.serve(db, student, k=payload.k)
    items = [ServeResponseItem(question_id=q.id, difficulty_level=q.difficulty_level, t_ref_ms=q.t_ref_ms) for q in chosen]
    return ServeResponse(items=items)


@router.post("/submit-answer", response_model=SubmitResponse)
async def submit_answer(payload: SubmitRequest, db: AsyncSession = Depends(get_db)):
    student = await repo.get_student(db, payload.student_id)
    question = await repo.get_question(db, payload.question_id)
    if not student or not question:
        raise HTTPException(404, "not found")
    new_rs, new_rq = await _RS.update_after_attempt(db, student, question, payload.is_correct, payload.time_ms)
    # event log
    await db.execute(
        insert(Event).values(
            user_id=student.id,
            event_type="adaptive_submit",
            payload=json.dumps(
                {
                    "student_id": payload.student_id,
                    "question_id": payload.question_id,
                    "is_correct": payload.is_correct,
                    "time_ms": payload.time_ms,
                    "new_skill": new_rs,
                    "new_question_diff": new_rq,
                }
            ),
        )
    )
    await db.commit()
    return SubmitResponse(new_skill=new_rs, new_question_diff=new_rq)


