from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from app.core.db import get_db
from app.models import User, TeacherStudentRequest, Assignment, Submission
from app.schemas import AssignmentCreate, SubmissionCreate
from app.core.deps import require_roles

router = APIRouter(prefix="/teacher", tags=["teacher"])

@router.get("/students/requests")
async def list_requests(user: User = Depends(require_roles("teacher")), db: AsyncSession = Depends(get_db)):
    q = await db.execute(select(TeacherStudentRequest).where(TeacherStudentRequest.teacher_id==user.id))
    rows = q.scalars().all()
    return [{"id": r.id, "student_id": r.student_id, "status": r.status} for r in rows]

@router.post("/students/approve")
async def approve_request(request_id: int, approve: bool, user: User = Depends(require_roles("teacher")), db: AsyncSession = Depends(get_db)):
    q = await db.execute(select(TeacherStudentRequest).where(TeacherStudentRequest.id==request_id))
    r = q.scalar_one_or_none()
    if not r: raise HTTPException(404, "Request not found")
    if r.teacher_id != user.id:
        raise HTTPException(403, "Not your request")
    r.status = "approved" if approve else "rejected"
    await db.commit()
    return {"ok": True}

@router.post("/assignments")
async def create_assignment(body: AssignmentCreate, user: User = Depends(require_roles("teacher")), db: AsyncSession = Depends(get_db)):
    a = Assignment(teacher_id=user.id, title=body.title, description=body.description, topic=body.topic)
    db.add(a); await db.commit()
    return {"id": a.id, "title": a.title}

@router.get("/assignments")
async def list_assignments(user: User = Depends(require_roles("teacher")), db: AsyncSession = Depends(get_db)):
    q = await db.execute(select(Assignment).where(Assignment.teacher_id==user.id))
    rows = q.scalars().all()
    return [{"id": r.id, "title": r.title, "topic": r.topic} for r in rows]

@router.get("/submissions")
async def list_submissions(user: User = Depends(require_roles("teacher")), db: AsyncSession = Depends(get_db)):
    q = await db.execute(select(Submission).join(Assignment).where(Assignment.teacher_id==user.id))
    rows = q.scalars().all()
    return [{"id": s.id, "assignment_id": s.assignment_id, "student_id": s.student_id, "score": float(s.score) if s.score is not None else None} for s in rows]


# Teacher-Student link endpoints
@router.post("/links/request")
async def link_request(target_student_id: int, user: User = Depends(require_roles("teacher")), db: AsyncSession = Depends(get_db)):
    r = TeacherStudentRequest(teacher_id=user.id, student_id=target_student_id, status="pending")
    db.add(r); await db.commit()
    return {"ok": True, "request_id": r.id}

@router.post("/links/approve")
async def link_approve(request_id: int, user: User = Depends(require_roles("teacher")), db: AsyncSession = Depends(get_db)):
    q = await db.execute(select(TeacherStudentRequest).where(TeacherStudentRequest.id==request_id))
    r = q.scalar_one_or_none()
    if not r or r.teacher_id != user.id:
        raise HTTPException(404, "Request not found")
    r.status = "approved"; await db.commit(); return {"ok": True}

@router.post("/links/reject")
async def link_reject(request_id: int, user: User = Depends(require_roles("teacher")), db: AsyncSession = Depends(get_db)):
    q = await db.execute(select(TeacherStudentRequest).where(TeacherStudentRequest.id==request_id))
    r = q.scalar_one_or_none()
    if not r or r.teacher_id != user.id:
        raise HTTPException(404, "Request not found")
    r.status = "rejected"; await db.commit(); return {"ok": True}

@router.post("/links/cancel")
async def link_cancel(request_id: int, user: User = Depends(require_roles("teacher")), db: AsyncSession = Depends(get_db)):
    q = await db.execute(select(TeacherStudentRequest).where(TeacherStudentRequest.id==request_id))
    r = q.scalar_one_or_none()
    if not r or r.teacher_id != user.id:
        raise HTTPException(404, "Request not found")
    r.status = "cancelled"; await db.commit(); return {"ok": True}

@router.post("/links/block")
async def link_block(student_id: int, user: User = Depends(require_roles("teacher")), db: AsyncSession = Depends(get_db)):
    # Basit blok: status=blocked kaydı oluştur ya da güncelle
    q = await db.execute(select(TeacherStudentRequest).where(TeacherStudentRequest.teacher_id==user.id, TeacherStudentRequest.student_id==student_id))
    r = q.scalar_one_or_none()
    if r:
        r.status = "blocked"
    else:
        r = TeacherStudentRequest(teacher_id=user.id, student_id=student_id, status="blocked"); db.add(r)
    await db.commit(); return {"ok": True}

@router.get("/links/pending")
async def links_pending(user: User = Depends(require_roles("teacher")), db: AsyncSession = Depends(get_db)):
    q = await db.execute(select(TeacherStudentRequest).where(TeacherStudentRequest.teacher_id==user.id, TeacherStudentRequest.status=="pending"))
    rows = q.scalars().all();
    return [{"id": r.id, "student_id": r.student_id, "status": r.status} for r in rows]

@router.get("/students")
async def list_students(user: User = Depends(require_roles("teacher")), db: AsyncSession = Depends(get_db)):
    q = await db.execute(select(TeacherStudentRequest).where(TeacherStudentRequest.teacher_id==user.id, TeacherStudentRequest.status=="approved"))
    rows = q.scalars().all();
    return [{"student_id": r.student_id} for r in rows]
