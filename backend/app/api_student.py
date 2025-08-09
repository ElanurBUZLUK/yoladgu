from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.db import get_db
from app.models import TeacherStudentRequest, Assignment, Submission, User
from app.schemas import SubmissionCreate
from app.core.deps import require_roles

router = APIRouter(prefix="/student", tags=["student"])

@router.post("/request-teacher")
async def request_teacher(teacher_id: int, user: User = Depends(require_roles("student")), db: AsyncSession = Depends(get_db)):
    if teacher_id == user.id:
        raise HTTPException(400, "Cannot request yourself")
    r = TeacherStudentRequest(teacher_id=teacher_id, student_id=user.id, status="pending")
    db.add(r); await db.commit()
    return {"ok": True, "request_id": r.id}

@router.get("/assignments")
async def list_assignments(user: User = Depends(require_roles("student")), db: AsyncSession = Depends(get_db)):
    # Sadece onaylı öğretmen eşleşmelerinden gelen ödevler
    approved = await db.execute(select(TeacherStudentRequest.teacher_id).where(
        TeacherStudentRequest.student_id==user.id, TeacherStudentRequest.status=="approved"
    ))
    teacher_ids = [tid for (tid,) in approved.all()]
    if not teacher_ids:
        return []
    q = await db.execute(select(Assignment).where(Assignment.teacher_id.in_(teacher_ids)))
    rows = q.scalars().all()
    return [{"id": r.id, "title": r.title, "topic": r.topic} for r in rows]

@router.post("/submit")
async def submit(body: SubmissionCreate, user: User = Depends(require_roles("student")), db: AsyncSession = Depends(get_db)):
    # Assignment var mı ve öğrenci ilişkili mi kontrol et (temel)
    ares = await db.execute(select(Assignment).where(Assignment.id==body.assignment_id))
    a = ares.scalar_one_or_none()
    if not a:
        raise HTTPException(404, "Assignment not found")
    # Öğretmen-öğrenci eşleşmesi onaylı mı
    tcheck = await db.execute(select(TeacherStudentRequest).where(
        TeacherStudentRequest.teacher_id==a.teacher_id,
        TeacherStudentRequest.student_id==user.id,
        TeacherStudentRequest.status=="approved"
    ))
    if tcheck.scalar_one_or_none() is None:
        raise HTTPException(403, "No access to this assignment")
    s = Submission(assignment_id=body.assignment_id, student_id=user.id, content=body.content)
    db.add(s); await db.commit()
    return {"ok": True, "submission_id": s.id}
