from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from app.core.db import get_db
from app.models import User, TeacherStudentRequest, Assignment, Submission
from app.schemas import AssignmentCreate, SubmissionCreate

router = APIRouter(prefix="/teacher", tags=["teacher"])

@router.get("/students/requests")
async def list_requests(teacher_id: int, db: AsyncSession = Depends(get_db)):
    q = await db.execute(select(TeacherStudentRequest).where(TeacherStudentRequest.teacher_id==teacher_id))
    rows = q.scalars().all()
    return [{"id": r.id, "student_id": r.student_id, "status": r.status} for r in rows]

@router.post("/students/approve")
async def approve_request(request_id: int, approve: bool, db: AsyncSession = Depends(get_db)):
    q = await db.execute(select(TeacherStudentRequest).where(TeacherStudentRequest.id==request_id))
    r = q.scalar_one_or_none()
    if not r: raise HTTPException(404, "Request not found")
    r.status = "approved" if approve else "rejected"
    await db.commit()
    return {"ok": True}

@router.post("/assignments")
async def create_assignment(teacher_id: int, body: AssignmentCreate, db: AsyncSession = Depends(get_db)):
    a = Assignment(teacher_id=teacher_id, title=body.title, description=body.description, topic=body.topic)
    db.add(a); await db.commit()
    return {"id": a.id, "title": a.title}

@router.get("/assignments")
async def list_assignments(teacher_id: int, db: AsyncSession = Depends(get_db)):
    q = await db.execute(select(Assignment).where(Assignment.teacher_id==teacher_id))
    rows = q.scalars().all()
    return [{"id": r.id, "title": r.title, "topic": r.topic} for r in rows]

@router.get("/submissions")
async def list_submissions(teacher_id: int, db: AsyncSession = Depends(get_db)):
    q = await db.execute(select(Submission).join(Assignment).where(Assignment.teacher_id==teacher_id))
    rows = q.scalars().all()
    return [{"id": s.id, "assignment_id": s.assignment_id, "student_id": s.student_id, "score": float(s.score) if s.score is not None else None} for s in rows]
