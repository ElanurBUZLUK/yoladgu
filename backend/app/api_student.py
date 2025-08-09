from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.db import get_db
from app.models import TeacherStudentRequest, Assignment, Submission
from app.schemas import SubmissionCreate

router = APIRouter(prefix="/student", tags=["student"])

@router.post("/request-teacher")
async def request_teacher(student_id: int, teacher_id: int, db: AsyncSession = Depends(get_db)):
    r = TeacherStudentRequest(teacher_id=teacher_id, student_id=student_id, status="pending")
    db.add(r); await db.commit()
    return {"ok": True, "request_id": r.id}

@router.get("/assignments")
async def list_assignments(student_id: int, db: AsyncSession = Depends(get_db)):
    # basit: tüm assignments (gerçekte sadece öğretmen-öğrenci eşleşmiş olan)
    q = await db.execute(select(Assignment))
    rows = q.scalars().all()
    return [{"id": r.id, "title": r.title, "topic": r.topic} for r in rows]

@router.post("/submit")
async def submit(body: SubmissionCreate, student_id: int, db: AsyncSession = Depends(get_db)):
    s = Submission(assignment_id=body.assignment_id, student_id=student_id, content=body.content)
    db.add(s); await db.commit()
    return {"ok": True, "submission_id": s.id}
