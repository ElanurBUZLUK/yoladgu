from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List
import asyncio

from app.database import database_manager
from app.services.pdf_upload_service import pdf_upload_service
from app.schemas.pdf_upload import (
    PDFUploadRequest, PDFUploadResponse, PDFUploadProgress, PDFUploadSummary,
    PDFUploadList, PDFUploadListRequest, Subject, ProcessingStatus,
    DeleteUploadRequest, PDFReprocessRequest, PDFUploadStatistics
)
from app.middleware.auth import get_current_student, get_current_teacher, get_current_admin
from app.core.error_handling import ErrorHandler, ErrorCode, ErrorSeverity
from app.models.user import User
from app.models.question import Subject as ModelSubject

router = APIRouter(prefix="/api/v1/pdf", tags=["pdf"])
error_handler = ErrorHandler()


@router.post("/upload", response_model=PDFUploadResponse)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to upload"),
    subject: Subject = Form(..., description="Subject for the PDF content"),
    description: Optional[str] = Form(None, description="Optional description"),
    current_user: User = Depends(get_current_teacher),  # Only teachers can upload PDFs
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Upload a PDF file for question extraction"""
    
    try:
        # Check daily upload limit
        can_upload = await pdf_upload_service.check_daily_upload_limit(db, str(current_user.id))
        if not can_upload:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Daily upload limit exceeded. Please try again tomorrow."
            )
        
        # Validate file
        validation_result = await pdf_upload_service.validate_file(file)
        if not validation_result["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "File validation failed",
                    "errors": validation_result["errors"],
                    "warnings": validation_result["warnings"]
                }
            )
        
        # Convert subject to model enum
        model_subject = ModelSubject.MATH if subject == Subject.MATH else ModelSubject.ENGLISH
        
        # Save file
        file_info = await pdf_upload_service.save_file(file, str(current_user.id), model_subject)
        
        # Create database record
        pdf_upload = await pdf_upload_service.create_upload_record(
            db, str(current_user.id), model_subject, file_info
        )
        
        # Start virus scan if enabled
        if pdf_upload_service.VIRUS_SCAN_ENABLED:
            # Update status to virus scanning
            await pdf_upload_service.update_upload_status(
                db, str(pdf_upload.id), "virus_scanning"
            )
            
            # Perform virus scan (in background)
            asyncio.create_task(
                _background_virus_scan(str(pdf_upload.id), file_info["file_path"])
            )
        
        return PDFUploadResponse(
            upload_id=str(pdf_upload.id),
            filename=pdf_upload.filename,
            subject=pdf_upload.subject.value,
            file_size=file_info["file_size"],
            status=ProcessingStatus.PENDING,
            virus_scan_status=pdf_upload.virus_scan_status,
            message="File uploaded successfully. Processing will begin shortly.",
            created_at=pdf_upload.created_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@router.get("/upload/{upload_id}/progress", response_model=PDFUploadProgress)
async def get_upload_progress(
    upload_id: str,
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Get upload processing progress"""
    
    try:
        progress = await pdf_upload_service.get_upload_progress(db, upload_id)
        
        # Check if user has access to this upload
        from sqlalchemy import select
        from app.models.pdf_upload import PDFUpload
        
        result = await db.execute(
            select(PDFUpload).where(PDFUpload.id == upload_id)
        )
        pdf_upload = result.scalar_one_or_none()
        
        if not pdf_upload:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        # Allow access if user is the uploader or has teacher/admin role
        if (str(pdf_upload.uploaded_by) != str(current_user.id) and 
            current_user.role not in ['teacher', 'admin']):
            raise HTTPException(status_code=403, detail="Access denied")
        
        return PDFUploadProgress(**progress)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get progress: {str(e)}"
        )


@router.get("/uploads", response_model=PDFUploadList)
async def list_uploads(
    subject: Optional[Subject] = Query(None, description="Filter by subject"),
    status: Optional[ProcessingStatus] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100, description="Number of uploads to return"),
    skip: int = Query(0, ge=0, description="Number of uploads to skip"),
    current_user: User = Depends(get_current_student),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Get list of user's PDF uploads"""
    
    try:
        uploads = await pdf_upload_service.get_user_uploads(
            db, str(current_user.id), limit, skip
        )
        
        # Filter by subject and status if provided
        filtered_uploads = uploads
        if subject:
            filtered_uploads = [u for u in filtered_uploads if u["subject"] == subject.value]
        if status:
            filtered_uploads = [u for u in filtered_uploads if u["status"] == status.value]
        
        # Convert to response format
        upload_summaries = [
            PDFUploadSummary(
                upload_id=upload["upload_id"],
                filename=upload["filename"],
                subject=upload["subject"],
                status=ProcessingStatus(upload["status"]),
                questions_extracted=upload["questions_extracted"],
                quality_score=upload["quality_score"],
                created_at=upload["created_at"],
                file_size=upload["file_size"]
            )
            for upload in filtered_uploads
        ]
        
        return PDFUploadList(
            uploads=upload_summaries,
            total_count=len(upload_summaries),
            page=skip // limit + 1,
            page_size=limit,
            has_next=len(uploads) == limit
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list uploads: {str(e)}"
        )


@router.delete("/upload/{upload_id}")
async def delete_upload(
    upload_id: str,
    confirm: bool = Query(False, description="Confirmation flag"),
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Delete an uploaded PDF and its extracted questions"""
    
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Deletion must be confirmed with confirm=true parameter"
        )
    
    try:
        success = await pdf_upload_service.delete_upload(db, upload_id, str(current_user.id))
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Upload not found or access denied"
            )
        
        return {
            "success": True,
            "message": "Upload deleted successfully",
            "upload_id": upload_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete upload: {str(e)}"
        )


@router.post("/upload/{upload_id}/reprocess")
async def reprocess_pdf(
    upload_id: str,
    force: bool = Query(False, description="Force reprocessing"),
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Reprocess a PDF to extract questions again"""
    
    try:
        # Check if upload exists and user has access
        from sqlalchemy import select
        from app.models.pdf_upload import PDFUpload
        
        result = await db.execute(
            select(PDFUpload).where(
                PDFUpload.id == upload_id,
                PDFUpload.uploaded_by == str(current_user.id)
            )
        )
        
        pdf_upload = result.scalar_one_or_none()
        if not pdf_upload:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Upload not found or access denied"
            )
        
        # Check if reprocessing is allowed
        if pdf_upload.processing_status == "processing" and not force:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Upload is currently being processed. Use force=true to override."
            )
        
        # Reset status to pending
        await pdf_upload_service.update_upload_status(
            db, upload_id, "pending", {"questions_extracted": 0}
        )
        
        # Trigger reprocessing workflow
        from app.services.pdf_processing_service import pdf_processing_service
        
        # Background task olarak yeniden işleme başlat
        asyncio.create_task(
            pdf_processing_service.reprocess_pdf_upload(
                db, upload_id, str(current_user.id)
            )
        )
        
        return {
            "success": True,
            "message": "Reprocessing started",
            "upload_id": upload_id,
            "status": "pending"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reprocess: {str(e)}"
        )


@router.post("/{upload_id}/process", response_model=dict)
async def process_pdf_upload(
    upload_id: str,
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Manually trigger PDF processing"""
    
    try:
        from app.services.pdf_processing_service import pdf_processing_service
        
        # Background task olarak işleme başlat
        asyncio.create_task(
            pdf_processing_service.process_pdf_upload(
                db, upload_id, str(current_user.id)
            )
        )
        
        return {
            "success": True,
            "message": "PDF processing started",
            "upload_id": upload_id,
            "status": "processing"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start processing: {str(e)}"
        )


@router.get("/statistics", response_model=PDFUploadStatistics)
async def get_upload_statistics(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    current_user: User = Depends(get_current_teacher),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Get PDF upload statistics"""
    
    try:
        from sqlalchemy import select, func
        from app.models.pdf_upload import PDFUpload
        from datetime import datetime, timedelta
        
        # Date filters
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Get all uploads for the user in the period
        result = await db.execute(
            select(PDFUpload).where(
                PDFUpload.uploaded_by == str(current_user.id),
                PDFUpload.created_at >= cutoff_date
            )
        )
        uploads = result.scalars().all()
        
        # Calculate statistics
        total_uploads = len(uploads)
        successful_uploads = len([u for u in uploads if u.processing_status == "completed"])
        failed_uploads = len([u for u in uploads if u.processing_status == "failed"])
        pending_uploads = len([u for u in uploads if u.processing_status in ["pending", "processing"]])
        
        total_questions = sum(u.questions_extracted for u in uploads)
        quality_scores = [u.quality_score for u in uploads if u.quality_score is not None]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else None
        
        # Subject distribution
        subject_dist = {}
        for upload in uploads:
            subject = upload.subject.value
            subject_dist[subject] = subject_dist.get(subject, 0) + 1
        
        # Daily and monthly counts
        daily_uploads = len([u for u in uploads if u.created_at >= today_start])
        monthly_uploads = len([u for u in uploads if u.created_at >= month_start])
        
        return PDFUploadStatistics(
            total_uploads=total_uploads,
            successful_uploads=successful_uploads,
            failed_uploads=failed_uploads,
            pending_uploads=pending_uploads,
            total_questions_extracted=total_questions,
            average_quality_score=avg_quality,
            upload_size_stats={
                "period_days": days,
                "uploads_per_day": total_uploads / days if days > 0 else 0
            },
            subject_distribution=subject_dist,
            daily_upload_count=daily_uploads,
            monthly_upload_count=monthly_uploads
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )


# Admin endpoints
@router.get("/admin/uploads", response_model=PDFUploadList)
async def list_all_uploads(
    subject: Optional[Subject] = Query(None, description="Filter by subject"),
    status: Optional[ProcessingStatus] = Query(None, description="Filter by status"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    limit: int = Query(20, ge=1, le=100, description="Number of uploads to return"),
    skip: int = Query(0, ge=0, description="Number of uploads to skip"),
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Get list of all PDF uploads (Admin only)"""
    
    try:
        from sqlalchemy import select, desc
        from app.models.pdf_upload import PDFUpload
        
        # Build query
        query = select(PDFUpload)
        
        # Apply filters
        if user_id:
            query = query.where(PDFUpload.uploaded_by == user_id)
        if subject:
            model_subject = ModelSubject.MATH if subject == Subject.MATH else ModelSubject.ENGLISH
            query = query.where(PDFUpload.subject == model_subject)
        if status:
            query = query.where(PDFUpload.processing_status == status.value)
        
        # Apply pagination and ordering
        query = query.order_by(desc(PDFUpload.created_at)).offset(skip).limit(limit)
        
        result = await db.execute(query)
        uploads = result.scalars().all()
        
        # Convert to response format
        upload_summaries = [
            PDFUploadSummary(
                upload_id=str(upload.id),
                filename=upload.filename,
                subject=upload.subject.value,
                status=ProcessingStatus(upload.processing_status.value),
                questions_extracted=upload.questions_extracted,
                quality_score=upload.quality_score,
                created_at=upload.created_at.isoformat()
            )
            for upload in uploads
        ]
        
        return PDFUploadList(
            uploads=upload_summaries,
            total_count=len(upload_summaries),
            page=skip // limit + 1,
            page_size=limit,
            has_next=len(uploads) == limit
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list uploads: {str(e)}"
        )


@router.get("/admin/statistics")
async def get_system_upload_statistics(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    current_user: User = Depends(get_current_admin),
    db: AsyncSession = Depends(database_manager.get_session)
):
    """Get system-wide PDF upload statistics (Admin only)"""
    
    try:
        from sqlalchemy import select, func
        from app.models.pdf_upload import PDFUpload
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get all uploads in the period
        result = await db.execute(
            select(PDFUpload).where(PDFUpload.created_at >= cutoff_date)
        )
        uploads = result.scalars().all()
        
        # Calculate system statistics
        total_uploads = len(uploads)
        successful_uploads = len([u for u in uploads if u.processing_status == "completed"])
        failed_uploads = len([u for u in uploads if u.processing_status == "failed"])
        
        # User statistics
        unique_users = len(set(str(u.uploaded_by) for u in uploads))
        
        # Subject distribution
        subject_dist = {}
        for upload in uploads:
            subject = upload.subject.value
            subject_dist[subject] = subject_dist.get(subject, 0) + 1
        
        # Quality statistics
        quality_scores = [u.quality_score for u in uploads if u.quality_score is not None]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else None
        
        return {
            "period_days": days,
            "total_uploads": total_uploads,
            "successful_uploads": successful_uploads,
            "failed_uploads": failed_uploads,
            "success_rate": (successful_uploads / total_uploads * 100) if total_uploads > 0 else 0,
            "unique_users": unique_users,
            "average_quality_score": avg_quality,
            "subject_distribution": subject_dist,
            "uploads_per_day": total_uploads / days if days > 0 else 0,
            "total_questions_extracted": sum(u.questions_extracted for u in uploads)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system statistics: {str(e)}"
        )


# Background task functions
async def _background_virus_scan(upload_id: str, file_path: str):
    """Background virus scan task"""
    
    try:
        from pathlib import Path
        
        # Perform virus scan
        scan_result = await pdf_upload_service.virus_scan_file(Path(file_path))
        
        # Update database with scan result
        async with database_manager.get_session() as db:
            if scan_result["clean"]:
                await pdf_upload_service.update_upload_status(
                    db, upload_id, "pending", {"virus_scan_status": "clean"}
                )
            else:
                await pdf_upload_service.update_upload_status(
                    db, upload_id, "failed", {"virus_scan_status": "infected"}
                )
            
    except Exception as e:
        # Update status to failed on error
        async with database_manager.get_session() as db:
            await pdf_upload_service.update_upload_status(
                db, upload_id, "failed", {"virus_scan_status": "failed"}
            )


@router.get("/health")
async def pdf_health():
    """PDF upload module health check"""
    return {"status": "ok", "module": "pdf_upload"}