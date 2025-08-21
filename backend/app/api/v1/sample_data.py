from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any

from app.database import database_manager
# from app.core.auth import get_current_admin  # Temporarily disabled
from app.models.user import User
from app.services.sample_data_service import sample_data_service

router = APIRouter(prefix="/api/v1/sample-data", tags=["Sample Data"])


@router.post("/create", response_model=Dict[str, Any])
async def create_sample_data(
    # current_user: User = Depends(get_current_admin),  # Temporarily disabled
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Sample data oluştur (sadece admin kullanıcılar)"""
    
    try:
        results = await sample_data_service.create_sample_data(db)
        
        return {
            "success": True,
            "message": "Sample data başarıyla oluşturuldu",
            "data": results,
            "summary": {
                "users_created": len(results["users"]),
                "questions_created": len(results["questions"]),
                "attempts_created": len(results["attempts"]),
                "error_patterns_created": len(results["error_patterns"]),
                "pdf_uploads_created": len(results["pdf_uploads"])
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sample data oluşturma hatası: {str(e)}"
        )


@router.post("/users", response_model=Dict[str, Any])
async def create_sample_users(
    # current_user: User = Depends(get_current_admin),  # Temporarily disabled
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Sadece sample users oluştur"""
    
    try:
        users = await sample_data_service.create_sample_users(db)
        
        return {
            "success": True,
            "message": f"{len(users)} sample user oluşturuldu",
            "users": users
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sample users oluşturma hatası: {str(e)}"
        )


@router.post("/questions", response_model=Dict[str, Any])
async def create_sample_questions(
    # current_user: User = Depends(get_current_admin),  # Temporarily disabled
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Sadece sample questions oluştur"""
    
    try:
        questions = await sample_data_service.create_sample_questions(db)
        
        return {
            "success": True,
            "message": f"{len(questions)} sample question oluşturuldu",
            "questions": questions
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sample questions oluşturma hatası: {str(e)}"
        )


@router.post("/attempts", response_model=Dict[str, Any])
async def create_sample_attempts(
    # current_user: User = Depends(get_current_admin),  # Temporarily disabled
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Sadece sample attempts oluştur (users ve questions gerekli)"""
    
    try:
        # Önce users ve questions'ları al
        from app.models.user import User
        from app.models.question import Question
        from sqlalchemy import select
        
        users_result = await db.execute(select(User))
        users = users_result.scalars().all()
        
        questions_result = await db.execute(select(Question))
        questions = questions_result.scalars().all()
        
        if not users:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Önce sample users oluşturulmalı"
            )
        
        if not questions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Önce sample questions oluşturulmalı"
            )
        
        # Users ve questions'ları dict formatına çevir
        users_dict = [
            {
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "role": user.role.value
            }
            for user in users
        ]
        
        questions_dict = [
            {
                "id": str(question.id),
                "content": question.content,
                "subject": question.subject.value,
                "difficulty_level": question.difficulty_level
            }
            for question in questions
        ]
        
        attempts = await sample_data_service.create_sample_attempts(db, users_dict, questions_dict)
        
        return {
            "success": True,
            "message": f"{len(attempts)} sample attempt oluşturuldu",
            "attempts": attempts
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sample attempts oluşturma hatası: {str(e)}"
        )


@router.post("/error-patterns", response_model=Dict[str, Any])
async def create_sample_error_patterns(
    # current_user: User = Depends(get_current_admin),  # Temporarily disabled
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Sadece sample error patterns oluştur (users gerekli)"""
    
    try:
        # Önce users'ları al
        from app.models.user import User
        from sqlalchemy import select
        
        users_result = await db.execute(select(User))
        users = users_result.scalars().all()
        
        if not users:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Önce sample users oluşturulmalı"
            )
        
        # Users'ları dict formatına çevir
        users_dict = [
            {
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "role": user.role.value
            }
            for user in users
        ]
        
        # Boş questions listesi (error patterns için gerekli değil)
        questions_dict = []
        
        error_patterns = await sample_data_service.create_sample_error_patterns(db, users_dict, questions_dict)
        
        return {
            "success": True,
            "message": f"{len(error_patterns)} sample error pattern oluşturuldu",
            "error_patterns": error_patterns
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sample error patterns oluşturma hatası: {str(e)}"
        )


@router.post("/pdf-uploads", response_model=Dict[str, Any])
async def create_sample_pdf_uploads(
    # current_user: User = Depends(get_current_admin),  # Temporarily disabled
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Sadece sample PDF uploads oluştur (users gerekli)"""
    
    try:
        # Önce users'ları al
        from app.models.user import User
        from sqlalchemy import select
        
        users_result = await db.execute(select(User))
        users = users_result.scalars().all()
        
        if not users:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Önce sample users oluşturulmalı"
            )
        
        # Users'ları dict formatına çevir
        users_dict = [
            {
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "role": user.role.value
            }
            for user in users
        ]
        
        pdf_uploads = await sample_data_service.create_sample_pdf_uploads(db, users_dict)
        
        return {
            "success": True,
            "message": f"{len(pdf_uploads)} sample PDF upload oluşturuldu",
            "pdf_uploads": pdf_uploads
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sample PDF uploads oluşturma hatası: {str(e)}"
        )


@router.delete("/clear", response_model=Dict[str, Any])
async def clear_sample_data(
    # current_user: User = Depends(get_current_admin),  # Temporarily disabled
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Sample data'yı temizle (sadece admin kullanıcılar)"""
    
    try:
        deleted_counts = await sample_data_service.clear_sample_data(db)
        
        return {
            "success": True,
            "message": "Sample data başarıyla temizlendi",
            "deleted_counts": deleted_counts
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sample data temizleme hatası: {str(e)}"
        )


@router.get("/status", response_model=Dict[str, Any])
async def get_sample_data_status(
    # current_user: User = Depends(get_current_admin),  # Temporarily disabled
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Sample data durumunu kontrol et"""
    
    try:
        from app.models.user import User
        from app.models.question import Question
        from app.models.student_attempt import StudentAttempt
        from app.models.error_pattern import ErrorPattern
        from app.models.pdf_upload import PDFUpload
        from sqlalchemy import select, func
        
        # Her entity için count al
        user_count = await db.execute(select(func.count(User.id)))
        user_count = user_count.scalar()
        
        question_count = await db.execute(select(func.count(Question.id)))
        question_count = question_count.scalar()
        
        attempt_count = await db.execute(select(func.count(StudentAttempt.id)))
        attempt_count = attempt_count.scalar()
        
        error_pattern_count = await db.execute(select(func.count(ErrorPattern.id)))
        error_pattern_count = error_pattern_count.scalar()
        
        pdf_upload_count = await db.execute(select(func.count(PDFUpload.id)))
        pdf_upload_count = pdf_upload_count.scalar()
        
        # Role'lara göre user count
        admin_count = await db.execute(select(func.count(User.id)).where(User.role == "admin"))
        admin_count = admin_count.scalar()
        
        teacher_count = await db.execute(select(func.count(User.id)).where(User.role == "teacher"))
        teacher_count = teacher_count.scalar()
        
        student_count = await db.execute(select(func.count(User.id)).where(User.role == "student"))
        student_count = student_count.scalar()
        
        return {
            "success": True,
            "data_status": {
                "users": {
                    "total": user_count,
                    "admins": admin_count,
                    "teachers": teacher_count,
                    "students": student_count
                },
                "questions": question_count,
                "attempts": attempt_count,
                "error_patterns": error_pattern_count,
                "pdf_uploads": pdf_upload_count
            },
            "has_sample_data": user_count > 0 or question_count > 0 or attempt_count > 0
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sample data status kontrolü hatası: {str(e)}"
        )
