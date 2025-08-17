import asyncio
import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import pdfplumber
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.pdf_upload import PDFUpload, ProcessingStatus
from app.models.question import Question, Subject, QuestionType, SourceType
from app.services.llm_gateway import llm_gateway
from app.core.config import settings


class PDFProcessingService:
    """PDF işleme servisi - PDF'lerden soru çıkarma ve işleme"""
    
    def __init__(self):
        self.upload_dir = settings.upload_dir
        self.max_file_size = settings.max_file_size
    
    async def process_pdf_upload(
        self, 
        db: AsyncSession, 
        upload_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """PDF yüklemesini işle ve soruları çıkar"""
        
        try:
            # Upload kaydını al
            result = await db.execute(
                select(PDFUpload).where(PDFUpload.id == upload_id)
            )
            upload = result.scalar_one_or_none()
            
            if not upload:
                raise ValueError(f"Upload {upload_id} not found")
            
            # Status'u processing olarak güncelle
            upload.processing_status = ProcessingStatus.PROCESSING
            upload.processing_started_at = datetime.utcnow()
            await db.commit()
            
            # PDF dosyasını oku
            pdf_path = os.path.join(self.upload_dir, upload.file_path)
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # PDF'den metin çıkar
            extracted_text = await self._extract_text_from_pdf(pdf_path)
            
            # LLM ile soruları çıkar
            questions = await self._extract_questions_from_text(
                extracted_text, 
                upload.subject,
                user_id
            )
            
            # Soruları veritabanına kaydet
            saved_questions = await self._save_questions_to_db(
                db, questions, upload, user_id
            )
            
            # Upload'u güncelle
            upload.processing_status = ProcessingStatus.COMPLETED
            upload.processing_completed_at = datetime.utcnow()
            upload.questions_extracted = len(saved_questions)
            upload.quality_score = self._calculate_quality_score(questions)
            upload.processing_metadata = {
                "extracted_questions": len(saved_questions),
                "processing_time": (datetime.utcnow() - upload.processing_started_at).total_seconds(),
                "text_length": len(extracted_text),
                "quality_score": upload.quality_score
            }
            
            await db.commit()
            
            return {
                "success": True,
                "upload_id": upload_id,
                "questions_extracted": len(saved_questions),
                "quality_score": upload.quality_score,
                "processing_time": upload.processing_metadata["processing_time"]
            }
            
        except Exception as e:
            # Hata durumunda status'u failed olarak güncelle
            if upload:
                upload.processing_status = ProcessingStatus.FAILED
                upload.processing_metadata = {
                    "error": str(e),
                    "failed_at": datetime.utcnow().isoformat()
                }
                await db.commit()
            
            raise e
    
    async def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF'den metin çıkar"""
        try:
            text_content = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_content.append(text)
            
            return "\n".join(text_content)
            
        except Exception as e:
            raise Exception(f"PDF text extraction failed: {str(e)}")
    
    async def _extract_questions_from_text(
        self, 
        text: str, 
        subject: Subject,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Metinden soruları çıkar"""
        
        try:
            # LLM Gateway kullanarak soru çıkarma
            llm_result = await llm_gateway.extract_questions_from_text(
                text=text,
                subject=subject.value,
                user_id=user_id
            )
            
            if llm_result.get("success"):
                return llm_result.get("questions", [])
            else:
                # Fallback: basit soru çıkarma
                return await self._fallback_question_extraction(text, subject)
                
        except Exception as e:
            print(f"LLM question extraction failed: {e}")
            # Fallback kullan
            return await self._fallback_question_extraction(text, subject)
    
    async def _fallback_question_extraction(
        self, 
        text: str, 
        subject: Subject
    ) -> List[Dict[str, Any]]:
        """Basit soru çıkarma (LLM başarısız olduğunda)"""
        
        questions = []
        lines = text.split('\n')
        
        # Basit soru tespiti
        question_indicators = ['?', 'soru', 'problem', 'question']
        
        for i, line in enumerate(lines):
            line = line.strip()
            if any(indicator in line.lower() for indicator in question_indicators):
                if len(line) > 20:  # Minimum uzunluk kontrolü
                    questions.append({
                        "content": line,
                        "question_type": "multiple_choice",
                        "difficulty_level": 3,
                        "topic_category": "general",
                        "correct_answer": "",
                        "options": [],
                        "source_type": "pdf"
                    })
        
        return questions[:10]  # Maksimum 10 soru
    
    async def _save_questions_to_db(
        self, 
        db: AsyncSession, 
        questions: List[Dict[str, Any]], 
        upload: PDFUpload,
        user_id: str
    ) -> List[Question]:
        """Soruları veritabanına kaydet"""
        
        saved_questions = []
        
        for question_data in questions:
            try:
                question = Question(
                    subject=upload.subject,
                    content=question_data.get("content", ""),
                    question_type=QuestionType(question_data.get("question_type", "multiple_choice")),
                    difficulty_level=question_data.get("difficulty_level", 3),
                    original_difficulty=question_data.get("difficulty_level", 3),
                    topic_category=question_data.get("topic_category", "general"),
                    correct_answer=question_data.get("correct_answer"),
                    options=question_data.get("options", []),
                    source_type=SourceType.PDF,
                    pdf_source_path=upload.file_path,
                    question_metadata={
                        "extracted_from_pdf": True,
                        "upload_id": str(upload.id),
                        "user_id": user_id,
                        "extraction_method": question_data.get("extraction_method", "llm"),
                        "confidence_score": question_data.get("confidence_score", 0.5)
                    }
                )
                
                db.add(question)
                saved_questions.append(question)
                
            except Exception as e:
                print(f"Failed to save question: {e}")
                continue
        
        await db.commit()
        return saved_questions
    
    def _calculate_quality_score(self, questions: List[Dict[str, Any]]) -> float:
        """Soru kalitesi skorunu hesapla"""
        
        if not questions:
            return 0.0
        
        total_score = 0.0
        
        for question in questions:
            score = 0.0
            
            # İçerik uzunluğu kontrolü
            content = question.get("content", "")
            if len(content) > 50:
                score += 0.3
            
            # Soru tipi kontrolü
            if question.get("question_type") in ["multiple_choice", "open_ended"]:
                score += 0.2
            
            # Zorluk seviyesi kontrolü
            difficulty = question.get("difficulty_level", 3)
            if 1 <= difficulty <= 5:
                score += 0.2
            
            # Seçenek kontrolü (çoktan seçmeli için)
            if question.get("question_type") == "multiple_choice":
                options = question.get("options", [])
                if len(options) >= 2:
                    score += 0.2
            
            # Doğru cevap kontrolü
            if question.get("correct_answer"):
                score += 0.1
            
            total_score += score
        
        return total_score / len(questions)
    
    async def reprocess_pdf_upload(
        self, 
        db: AsyncSession, 
        upload_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """PDF yüklemesini yeniden işle"""
        
        # Önceki soruları sil
        await self._delete_questions_from_upload(db, upload_id)
        
        # Yeniden işle
        return await self.process_pdf_upload(db, upload_id, user_id)
    
    async def _delete_questions_from_upload(
        self, 
        db: AsyncSession, 
        upload_id: str
    ):
        """Upload'a ait soruları sil"""
        
        result = await db.execute(
            select(Question).where(
                Question.pdf_source_path.like(f"%{upload_id}%")
            )
        )
        questions = result.scalars().all()
        
        for question in questions:
            await db.delete(question)
        
        await db.commit()


# Singleton instance
pdf_processing_service = PDFProcessingService()
