import os
import uuid
import hashlib
import mimetypes
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import aiofiles
from fastapi import UploadFile, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.user import User
from app.models.pdf_upload import PDFUpload
from app.models.question import Subject
from app.core.config import settings


class PDFUploadService:
    """Service for handling PDF file uploads with security and validation"""
    
    def __init__(self):
        # File validation settings
        self.MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
        self.ALLOWED_MIME_TYPES = [
            'application/pdf',
            'application/x-pdf'
        ]
        self.ALLOWED_EXTENSIONS = ['.pdf']
        
        # Storage settings
        self.UPLOAD_DIR = Path(settings.upload_directory) if hasattr(settings, 'upload_directory') else Path("uploads")
        self.PDF_DIR = self.UPLOAD_DIR / "pdfs"
        
        # Security settings
        self.VIRUS_SCAN_ENABLED = getattr(settings, 'virus_scan_enabled', False)
        self.MAX_DAILY_UPLOADS = 20
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure upload directories exist"""
        self.UPLOAD_DIR.mkdir(exist_ok=True)
        self.PDF_DIR.mkdir(exist_ok=True)
        
        # Create subdirectories by date for organization
        today = datetime.now().strftime("%Y/%m/%d")
        (self.PDF_DIR / today).mkdir(parents=True, exist_ok=True)
    
    async def validate_file(self, file: UploadFile) -> Dict[str, Any]:
        """Validate uploaded PDF file"""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "file_info": {}
        }
        
        # Check file extension
        if not file.filename:
            validation_result["valid"] = False
            validation_result["errors"].append("No filename provided")
            return validation_result
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.ALLOWED_EXTENSIONS:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Invalid file extension: {file_ext}. Only PDF files are allowed.")
        
        # Check MIME type
        if file.content_type not in self.ALLOWED_MIME_TYPES:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Invalid MIME type: {file.content_type}. Only PDF files are allowed.")
        
        # Check file size
        file_size = 0
        if hasattr(file, 'size') and file.size:
            file_size = file.size
        else:
            # Read file to get size
            content = await file.read()
            file_size = len(content)
            await file.seek(0)  # Reset file pointer
        
        if file_size > self.MAX_FILE_SIZE:
            validation_result["valid"] = False
            validation_result["errors"].append(f"File too large: {file_size} bytes. Maximum allowed: {self.MAX_FILE_SIZE} bytes.")
        
        if file_size == 0:
            validation_result["valid"] = False
            validation_result["errors"].append("Empty file not allowed")
        
        # Store file info
        validation_result["file_info"] = {
            "filename": file.filename,
            "size": file_size,
            "content_type": file.content_type,
            "extension": file_ext
        }
        
        # Check filename for security issues
        if self._has_security_issues(file.filename):
            validation_result["valid"] = False
            validation_result["errors"].append("Filename contains potentially dangerous characters")
        
        return validation_result
    
    def _has_security_issues(self, filename: str) -> bool:
        """Check filename for security issues"""
        dangerous_chars = ['..', '/', '\\', '<', '>', ':', '"', '|', '?', '*']
        return any(char in filename for char in dangerous_chars)
    
    async def check_daily_upload_limit(self, db: AsyncSession, user_id: str) -> bool:
        """Check if user has exceeded daily upload limit"""
        
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        result = await db.execute(
            select(PDFUpload).where(
                PDFUpload.uploaded_by == user_id,
                PDFUpload.created_at >= today_start
            )
        )
        
        today_uploads = result.scalars().all()
        return len(today_uploads) < self.MAX_DAILY_UPLOADS
    
    async def save_file(self, file: UploadFile, user_id: str, subject: Subject) -> Dict[str, Any]:
        """Save uploaded file to storage"""
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        original_name = file.filename
        safe_filename = self._sanitize_filename(original_name)
        
        # Create file path with date organization
        today = datetime.now().strftime("%Y/%m/%d")
        file_dir = self.PDF_DIR / today
        file_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = file_dir / f"{file_id}_{safe_filename}"
        
        # Calculate file hash while saving
        hasher = hashlib.sha256()
        file_size = 0
        
        try:
            async with aiofiles.open(file_path, 'wb') as f:
                while chunk := await file.read(8192):  # Read in 8KB chunks
                    await f.write(chunk)
                    hasher.update(chunk)
                    file_size += len(chunk)
            
            file_hash = hasher.hexdigest()
            
            # Extract basic metadata
            metadata = await self._extract_metadata(file_path)
            
            return {
                "file_id": file_id,
                "original_filename": original_name,
                "safe_filename": safe_filename,
                "file_path": str(file_path),
                "relative_path": str(file_path.relative_to(self.UPLOAD_DIR)),
                "file_size": file_size,
                "file_hash": file_hash,
                "metadata": metadata,
                "upload_success": True
            }
            
        except Exception as e:
            # Clean up file if save failed
            if file_path.exists():
                file_path.unlink()
            
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save file: {str(e)}"
            )
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage"""
        # Remove dangerous characters
        safe_chars = []
        for char in filename:
            if char.isalnum() or char in '.-_':
                safe_chars.append(char)
            else:
                safe_chars.append('_')
        
        safe_filename = ''.join(safe_chars)
        
        # Ensure it's not too long
        if len(safe_filename) > 100:
            name_part = safe_filename[:90]
            ext_part = safe_filename[-10:] if '.' in safe_filename[-10:] else '.pdf'
            safe_filename = name_part + ext_part
        
        return safe_filename
    
    async def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract basic metadata from PDF file"""
        
        metadata = {
            "file_size": file_path.stat().st_size,
            "created_at": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
        }
        
        try:
            # Try to extract PDF-specific metadata using PyPDF2 if available
            import PyPDF2
            
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                metadata.update({
                    "page_count": len(pdf_reader.pages),
                    "pdf_version": pdf_reader.pdf_header if hasattr(pdf_reader, 'pdf_header') else None,
                    "encrypted": pdf_reader.is_encrypted,
                })
                
                # Extract document info if available
                if pdf_reader.metadata:
                    doc_info = {}
                    for key, value in pdf_reader.metadata.items():
                        if isinstance(value, str):
                            doc_info[key.replace('/', '')] = value
                    metadata["document_info"] = doc_info
                
        except ImportError:
            # PyPDF2 not available, use basic metadata only
            metadata["extraction_method"] = "basic"
        except Exception as e:
            # PDF parsing failed, but file is still valid
            metadata["pdf_parsing_error"] = str(e)
            metadata["extraction_method"] = "basic"
        
        return metadata
    
    async def create_upload_record(
        self, 
        db: AsyncSession, 
        user_id: str,
        subject: Subject,
        file_info: Dict[str, Any]
    ) -> PDFUpload:
        """Create database record for uploaded PDF"""
        
        pdf_upload = PDFUpload(
            filename=file_info["original_filename"],
            file_path=file_info["relative_path"],
            uploaded_by=user_id,
            subject=subject,
            questions_extracted=0,
            processing_status="pending",
            quality_score=None,
            virus_scan_status="pending" if self.VIRUS_SCAN_ENABLED else "skipped"
        )
        
        db.add(pdf_upload)
        await db.commit()
        await db.refresh(pdf_upload)
        
        return pdf_upload
    
    async def update_upload_status(
        self, 
        db: AsyncSession, 
        upload_id: str,
        status: str,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """Update upload processing status"""
        
        result = await db.execute(
            select(PDFUpload).where(PDFUpload.id == upload_id)
        )
        
        pdf_upload = result.scalar_one_or_none()
        if not pdf_upload:
            raise HTTPException(status_code=404, detail="Upload record not found")
        
        pdf_upload.processing_status = status
        
        if additional_data:
            if "questions_extracted" in additional_data:
                pdf_upload.questions_extracted = additional_data["questions_extracted"]
            if "quality_score" in additional_data:
                pdf_upload.quality_score = additional_data["quality_score"]
            if "virus_scan_status" in additional_data:
                pdf_upload.virus_scan_status = additional_data["virus_scan_status"]
        
        await db.commit()
        return pdf_upload
    
    async def get_upload_progress(self, db: AsyncSession, upload_id: str) -> Dict[str, Any]:
        """Get upload processing progress"""
        
        result = await db.execute(
            select(PDFUpload).where(PDFUpload.id == upload_id)
        )
        
        pdf_upload = result.scalar_one_or_none()
        if not pdf_upload:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        # Calculate progress percentage based on status
        progress_map = {
            "pending": 10,
            "virus_scanning": 20,
            "processing": 50,
            "extracting_questions": 70,
            "validating_questions": 85,
            "completed": 100,
            "failed": 0
        }
        
        progress_percentage = progress_map.get(pdf_upload.processing_status, 0)
        
        return {
            "upload_id": str(pdf_upload.id),
            "filename": pdf_upload.filename,
            "status": pdf_upload.processing_status,
            "progress_percentage": progress_percentage,
            "questions_extracted": pdf_upload.questions_extracted,
            "quality_score": pdf_upload.quality_score,
            "virus_scan_status": pdf_upload.virus_scan_status,
            "created_at": pdf_upload.created_at.isoformat(),
            "subject": pdf_upload.subject.value
        }
    
    async def get_user_uploads(
        self, 
        db: AsyncSession, 
        user_id: str,
        limit: int = 20,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """Get user's upload history"""
        
        result = await db.execute(
            select(PDFUpload).where(
                PDFUpload.uploaded_by == user_id
            ).order_by(PDFUpload.created_at.desc()).offset(skip).limit(limit)
        )
        
        uploads = result.scalars().all()
        
        return [
            {
                "upload_id": str(upload.id),
                "filename": upload.filename,
                "subject": upload.subject.value,
                "status": upload.processing_status,
                "questions_extracted": upload.questions_extracted,
                "quality_score": upload.quality_score,
                "created_at": upload.created_at.isoformat(),
                "file_size": None  # Would need to be stored or calculated
            }
            for upload in uploads
        ]
    
    async def delete_upload(self, db: AsyncSession, upload_id: str, user_id: str) -> bool:
        """Delete uploaded PDF and its record"""
        
        result = await db.execute(
            select(PDFUpload).where(
                PDFUpload.id == upload_id,
                PDFUpload.uploaded_by == user_id
            )
        )
        
        pdf_upload = result.scalar_one_or_none()
        if not pdf_upload:
            return False
        
        # Delete physical file
        file_path = self.UPLOAD_DIR / pdf_upload.file_path
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as e:
                # Log error but continue with database deletion
                print(f"Failed to delete file {file_path}: {e}")
        
        # Delete database record
        await db.delete(pdf_upload)
        await db.commit()
        
        return True
    
    async def virus_scan_file(self, file_path: Path) -> Dict[str, Any]:
        """Perform virus scan on uploaded file (placeholder implementation)"""
        
        if not self.VIRUS_SCAN_ENABLED:
            return {
                "scan_result": "skipped",
                "clean": True,
                "message": "Virus scanning disabled"
            }
        
        # Placeholder implementation
        # In a real system, you would integrate with ClamAV or similar
        try:
            # Simulate virus scan
            import time
            await asyncio.sleep(1)  # Simulate scan time
            
            # Basic checks
            file_size = file_path.stat().st_size
            
            # Very basic heuristics (not real virus detection)
            if file_size > 100 * 1024 * 1024:  # Files over 100MB are suspicious
                return {
                    "scan_result": "suspicious",
                    "clean": False,
                    "message": "File size exceeds normal limits"
                }
            
            return {
                "scan_result": "clean",
                "clean": True,
                "message": "No threats detected"
            }
            
        except Exception as e:
            return {
                "scan_result": "error",
                "clean": False,
                "message": f"Scan failed: {str(e)}"
            }


# Global PDF upload service instance
pdf_upload_service = PDFUploadService()