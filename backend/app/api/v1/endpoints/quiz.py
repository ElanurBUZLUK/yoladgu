from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Dict, Any
import time
from app.schemas_quiz import StartQuizRequest, StartQuizResponse, AnswerRequest, NextRequest, FinishRequest
from app.services.quiz.session_service import QuizSessionService
from app.services.content.questions_service import QuestionsService
from app.services.policy.next_question_policy import pick_next_question
from app.core.config import settings
from app.core.deps import require_roles
from app.core.db import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.dependencies import get_event_logger, get_current_user

router = APIRouter(prefix="/quiz", tags=["quiz"])


def _get_user_id(req: Request) -> int:
    # In production, use JWT; for now not parsing the token payload here
    return 0


@router.post("/start", response_model=StartQuizResponse)
async def start_quiz(body: StartQuizRequest, request: Request, user=Depends(require_roles("student")), db: AsyncSession = Depends(get_db)):
    user_id = user.id
    topic_id = body.topic_id
    sess = QuizSessionService(settings.REDIS_URL)
    qsvc = QuestionsService(settings.REDIS_URL)

    session_id = sess.create(user_id=user_id, topic_id=topic_id)
    qid = await pick_next_question(user=user, db=db, topic_id=topic_id, asked_ids=[])
    if qid is None:
        raise HTTPException(404, "No candidate question found")
    q = qsvc.get(qid)
    if not q:
        raise HTTPException(404, "Question missing")
    sess.set_current(session_id, qid)

    evt = get_event_logger()
    evt.log_exposure(user_id=user_id, variant="ensemble", payload={"question_id": qid, "session_id": session_id})

    return {"session_id": session_id, "question": q}


@router.post("/answer")
def submit_answer(body: AnswerRequest, request: Request, user=Depends(require_roles("student"))):
    user_id = user.id
    sess = QuizSessionService(settings.REDIS_URL)
    qsvc = QuestionsService(settings.REDIS_URL)

    s = sess.get(body.session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    if s.get("current_qid") != body.question_id:
        pass

    q = qsvc.get(body.question_id)
    if not q:
        raise HTTPException(404, "Question not found")

    correct_index = int(q.get("correct_index", -1))
    is_correct = int(body.selected_option) == correct_index

    sess.add_answer(body.session_id, {
        "qid": body.question_id,
        "selected_option": body.selected_option,
        "correct": bool(is_correct),
        "time_ms": body.time_ms,
        "hint_used": bool(body.hint_used),
        "ts": int(time.time()*1000),
    })

    from app.core.dependencies import get_linucb_service, get_ftrl_service, get_feature_store
    linucb = get_linucb_service()
    ftrl = get_ftrl_service()
    fs = get_feature_store()
    uf = fs.get_user_features(user_id) if fs else {}
    qf = fs.get_question_features(body.question_id) if fs else {}
    try:
        linucb.update(uf, qf, body.question_id, float(1.0 if is_correct else 0.0))
    except Exception:
        pass
    try:
        ftrl.update(user_id, uf, qf, float(1.0 if is_correct else 0.0))
    except Exception:
        pass

    evt = get_event_logger()
    evt.log_outcome(user_id=user_id, variant="ensemble", payload={
        "session_id": body.session_id,
        "question_id": body.question_id,
        "is_correct": bool(is_correct),
        "response_time_ms": int(body.time_ms),
        "hint_used": bool(body.hint_used)
    })

    return {"ok": True, "correct": bool(is_correct)}


@router.post("/next")
async def next_question(body: NextRequest, request: Request, user=Depends(require_roles("student")), db: AsyncSession = Depends(get_db)):
    user_id = user.id
    sess = QuizSessionService(settings.REDIS_URL)
    qsvc = QuestionsService(settings.REDIS_URL)
    s = sess.get(body.session_id)
    if not s:
        raise HTTPException(404, "Session not found")

    asked = s.get("asked_ids", [])
    topic_id = s.get("topic_id")
    qid = await pick_next_question(user=user, db=db, topic_id=topic_id, asked_ids=asked)
    if qid is None:
        raise HTTPException(404, "No more candidates")
    q = qsvc.get(qid)
    if not q:
        raise HTTPException(404, "Question missing")
    sess.set_current(body.session_id, qid)

    evt = get_event_logger()
    evt.log_exposure(user_id=user_id, variant="ensemble", payload={"question_id": qid, "session_id": body.session_id})

    return {"question": q}


@router.post("/finish")
def finish_quiz(body: FinishRequest, request: Request, user=Depends(require_roles("student"))):
    sess = QuizSessionService(settings.REDIS_URL)
    s = sess.finish(body.session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    hist = s.get("history", [])
    n = len(hist)
    correct = sum(1 for h in hist if h.get("correct"))
    acc = (correct / n) if n else 0.0
    return {
        "total": n,
        "correct": correct,
        "wrong": n - correct,
        "accuracy": acc,
        "duration_ms": (s.get("end_ts", 0) - s.get("start_ts", 0)),
        "asked_ids": s.get("asked_ids", []),
    }


