from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
from pydantic import BaseModel

from app.core.database import get_async_session
from app.services.analytics_service import analytics_service
from app.middleware.auth import get_current_student, get_current_teacher, get_current_admin
from app.models.user import User
from app.models.question import Subject

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


class PerformanceAnalysisRequest(BaseModel):
    subject: Subject
    days: int = 30


class SimilarStudentsRequest(BaseModel):
    subject: Subject
    limit: int = 5


class TrendAnalysisRequest(BaseModel):
    subject: Subject
    days: int = 30


@router.post("/performance/analyze")
async def analyze_student_performance(
    request: PerformanceAnalysisRequest,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Öğrenci performans analizi"""
    
    try:
        analysis = await analytics_service.analyze_student_performance(
            db, str(current_user.id), request.subject, request.days
        )
        
        return {
            "success": True,
            "analysis": analysis
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Performance analysis failed: {str(e)}"
        )


@router.post("/students/similar")
async def find_similar_students(
    request: SimilarStudentsRequest,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Benzer öğrencileri bul"""
    
    try:
        similar_students = await analytics_service.find_similar_students(
            db, str(current_user.id), request.subject, request.limit
        )
        
        return {
            "success": True,
            "similar_students": similar_students,
            "total_found": len(similar_students)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similar students search failed: {str(e)}"
        )


@router.post("/trends/calculate")
async def calculate_performance_trends(
    request: TrendAnalysisRequest,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Performans trendlerini hesapla"""
    
    try:
        trends = await analytics_service.calculate_performance_trends(
            db, str(current_user.id), request.subject, request.days
        )
        
        return {
            "success": True,
            "trends": trends
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trend calculation failed: {str(e)}"
        )


@router.post("/strengths-weaknesses/identify")
async def identify_strengths_weaknesses(
    request: PerformanceAnalysisRequest,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Güçlü ve zayıf yönleri belirle"""
    
    try:
        analysis = await analytics_service.identify_weaknesses_strengths(
            db, str(current_user.id), request.subject, request.days
        )
        
        return {
            "success": True,
            "analysis": analysis
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Strengths/weaknesses analysis failed: {str(e)}"
        )


@router.get("/performance/summary/{subject}")
async def get_performance_summary(
    subject: Subject,
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(get_async_session)
):
    """Performans özeti al"""
    
    try:
        # Performans analizi
        performance = await analytics_service.analyze_student_performance(
            db, str(current_user.id), subject, days
        )
        
        # Trend analizi
        trends = await analytics_service.calculate_performance_trends(
            db, str(current_user.id), subject, days
        )
        
        # Güçlü/zayıf yönler
        strengths_weaknesses = await analytics_service.identify_weaknesses_strengths(
            db, str(current_user.id), subject, days
        )
        
        return {
            "success": True,
            "summary": {
                "performance": performance,
                "trends": trends,
                "strengths_weaknesses": strengths_weaknesses,
                "subject": subject.value,
                "analysis_period_days": days
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Performance summary failed: {str(e)}"
        )


# Teacher endpoints
@router.get("/teacher/student/{user_id}/performance")
async def get_student_performance_teacher(
    user_id: str,
    subject: Subject = Query(...),
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_async_session)
):
    """Öğretmen için öğrenci performans analizi"""
    
    try:
        # Performans analizi
        performance = await analytics_service.analyze_student_performance(
            db, user_id, subject, days
        )
        
        # Benzer öğrenciler
        similar_students = await analytics_service.find_similar_students(
            db, user_id, subject, 3
        )
        
        # Trend analizi
        trends = await analytics_service.calculate_performance_trends(
            db, user_id, subject, days
        )
        
        return {
            "success": True,
            "student_performance": {
                "performance": performance,
                "similar_students": similar_students,
                "trends": trends,
                "student_id": user_id,
                "subject": subject.value,
                "analysis_period_days": days
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Student performance analysis failed: {str(e)}"
        )


@router.get("/teacher/class/comparison")
async def get_class_performance_comparison(
    subject: Subject = Query(...),
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_async_session)
):
    """Sınıf performans karşılaştırması"""
    
    try:
        from sqlalchemy import select
        
        # Öğretmenin öğrencilerini bul
        result = await db.execute(
            select(User).where(
                User.role == "student",
                User.is_active == "true"
            )
        )
        students = result.scalars().all()
        
        class_performance = []
        
        for student in students:
            try:
                # Her öğrenci için performans analizi
                performance = await analytics_service.analyze_student_performance(
                    db, str(student.id), subject, days
                )
                
                if performance.get("total_attempts", 0) > 0:
                    class_performance.append({
                        "student_id": str(student.id),
                        "username": student.username,
                        "performance": performance
                    })
            
            except Exception as e:
                # Hata durumunda öğrenciyi atla
                continue
        
        # Performansa göre sırala
        class_performance.sort(
            key=lambda x: x["performance"].get("accuracy_rate", 0), 
            reverse=True
        )
        
        # Sınıf ortalaması hesapla
        if class_performance:
            avg_accuracy = sum(
                p["performance"].get("accuracy_rate", 0) 
                for p in class_performance
            ) / len(class_performance)
        else:
            avg_accuracy = 0
        
        return {
            "success": True,
            "class_comparison": {
                "subject": subject.value,
                "analysis_period_days": days,
                "total_students": len(class_performance),
                "average_accuracy": round(avg_accuracy, 2),
                "student_performances": class_performance
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Class comparison failed: {str(e)}"
        )


# Admin endpoints
@router.get("/admin/system/overview")
async def get_system_analytics_overview(
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_async_session)
):
    """Sistem analitik genel bakış (admin)"""
    
    try:
        from sqlalchemy import select, func
        from app.models.student_attempt import StudentAttempt
        from app.models.user import User
        from datetime import datetime, timedelta
        
        # Son 30 günlük istatistikler
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        
        # Toplam deneme sayısı
        result = await db.execute(
            select(func.count(StudentAttempt.id)).where(
                StudentAttempt.created_at >= cutoff_date
            )
        )
        total_attempts = result.scalar() or 0
        
        # Aktif öğrenci sayısı
        result = await db.execute(
            select(func.count(User.id)).where(
                User.role == "student",
                User.is_active == "true"
            )
        )
        active_students = result.scalar() or 0
        
        # Doğru cevap oranı
        result = await db.execute(
            select(
                func.count(StudentAttempt.id).label('total'),
                func.sum(func.case((StudentAttempt.is_correct == True, 1), else_=0)).label('correct')
            ).where(
                StudentAttempt.created_at >= cutoff_date
            )
        )
        stats = result.first()
        
        if stats and stats.total > 0:
            accuracy_rate = (stats.correct / stats.total) * 100
        else:
            accuracy_rate = 0
        
        # Konu bazında dağılım
        result = await db.execute(
            select(
                StudentAttempt.subject,
                func.count(StudentAttempt.id).label('attempts')
            ).where(
                StudentAttempt.created_at >= cutoff_date
            ).group_by(StudentAttempt.subject)
        )
        subject_distribution = [
            {"subject": row.subject.value, "attempts": row.attempts}
            for row in result.all()
        ]
        
        return {
            "success": True,
            "system_overview": {
                "period_days": 30,
                "total_attempts": total_attempts,
                "active_students": active_students,
                "overall_accuracy_rate": round(accuracy_rate, 2),
                "subject_distribution": subject_distribution,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"System overview failed: {str(e)}"
        )


@router.get("/admin/performance/leaderboard")
async def get_performance_leaderboard(
    subject: Subject = Query(...),
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_async_session)
):
    """Performans liderlik tablosu (admin)"""
    
    try:
        from sqlalchemy import select, func
        from app.models.student_attempt import StudentAttempt
        from app.models.user import User
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # En yüksek performanslı öğrencileri bul
        result = await db.execute(
            select(
                StudentAttempt.user_id,
                func.count(StudentAttempt.id).label('total_attempts'),
                func.sum(func.case((StudentAttempt.is_correct == True, 1), else_=0)).label('correct_attempts'),
                func.avg(StudentAttempt.difficulty_level).label('avg_difficulty')
            ).where(
                StudentAttempt.subject == subject,
                StudentAttempt.created_at >= cutoff_date
            ).group_by(StudentAttempt.user_id)
            .having(func.count(StudentAttempt.id) >= 5)  # En az 5 deneme
            .order_by(
                func.sum(func.case((StudentAttempt.is_correct == True, 1), else_=0)).desc()
            )
            .limit(limit)
        )
        
        leaderboard_data = result.all()
        
        leaderboard = []
        for row in leaderboard_data:
            accuracy = (row.correct_attempts / row.total_attempts) * 100
            
            # Öğrenci bilgilerini al
            user_result = await db.execute(
                select(User).where(User.id == row.user_id)
            )
            user = user_result.scalar_one_or_none()
            
            if user:
                leaderboard.append({
                    "rank": len(leaderboard) + 1,
                    "user_id": str(row.user_id),
                    "username": user.username,
                    "total_attempts": row.total_attempts,
                    "correct_attempts": row.correct_attempts,
                    "accuracy_rate": round(accuracy, 2),
                    "avg_difficulty": round(row.avg_difficulty, 2)
                })
        
        return {
            "success": True,
            "leaderboard": {
                "subject": subject.value,
                "period_days": days,
                "top_students": leaderboard
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Leaderboard generation failed: {str(e)}"
        )


@router.get("/health")
async def analytics_health_check():
    """Analytics health check endpoint"""
    return {
        "status": "healthy",
        "service": "analytics",
        "features": [
            "student_performance_analysis",
            "similar_students_detection",
            "performance_trends",
            "strengths_weaknesses_identification",
            "class_comparison",
            "system_overview"
        ]
    }
