import asyncio
import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import pdfplumber
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.pdf_upload import PDFUpload, ProcessingStatus
from app.models.question import Question, Subject, QuestionType, SourceType
from app.services.llm_gateway import llm_gateway
from app.services.embedding_service import embedding_service
from app.services.metadata_schema_service import metadata_schema_service, ContentType, Domain
from app.core.config import settings

logger = logging.getLogger(__name__)


class PDFProcessingService:
    """PDF işleme servisi - PDF'lerden soru çıkarma ve işleme"""
    
    def __init__(self):
        self.upload_dir = settings.upload_dir
        self.max_file_size = settings.max_file_size
        
        # Domain-specific question indicators
        self.question_indicators = {
            Subject.MATH: ['?', 'soru', 'problem', 'question', 'hesapla', 'çöz', 'bul', '=', '+', '-', '*', '/'],
            Subject.ENGLISH: ['?', 'soru', 'question', 'fill', 'blank', 'choose', 'correct', 'grammar', 'vocabulary'],
            Subject.GENERAL: ['?', 'soru', 'question', 'problem', 'test']
        }
        
        # Domain-specific metadata patterns
        self.metadata_patterns = {
            Subject.MATH: {
                'topics': ['algebra', 'geometry', 'calculus', 'trigonometry', 'statistics', 'arithmetic'],
                'difficulty_keywords': {
                    'easy': ['basit', 'kolay', 'temel', 'basic'],
                    'medium': ['orta', 'normal', 'standard'],
                    'hard': ['zor', 'difficult', 'advanced', 'ileri']
                }
            },
            Subject.ENGLISH: {
                'topics': ['grammar', 'vocabulary', 'reading', 'writing', 'listening', 'speaking'],
                'difficulty_keywords': {
                    'easy': ['beginner', 'elementary', 'basic', 'temel'],
                    'medium': ['intermediate', 'orta', 'normal'],
                    'hard': ['advanced', 'upper', 'ileri', 'difficult']
                }
            }
        }
    
    async def process_pdf_upload(
        self, 
        db: AsyncSession, 
        upload_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """PDF yüklemesini işle ve soruları çıkar"""
        
        try:
            logger.info(f"🚀 Starting PDF processing for upload {upload_id}")
            
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
            logger.info(f"📄 Extracted {len(extracted_text)} characters from PDF")
            
            # Domain-specific soru çıkarma
            questions = await self._extract_questions_from_text(
                extracted_text, 
                upload.subject,
                user_id
            )
            
            logger.info(f"❓ Extracted {len(questions)} questions from PDF")
            
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
                "quality_score": upload.quality_score,
                "domain": upload.subject.value,
                "extraction_method": "enhanced_domain_specific"
            }
            
            await db.commit()
            
            logger.info(f"✅ PDF processing completed successfully for upload {upload_id}")
            
            return {
                "success": True,
                "upload_id": upload_id,
                "questions_extracted": len(saved_questions),
                "quality_score": upload.quality_score,
                "processing_time": upload.processing_metadata["processing_time"],
                "domain": upload.subject.value
            }
            
        except Exception as e:
            logger.error(f"❌ PDF processing failed for upload {upload_id}: {e}", exc_info=True)
            
            # Hata durumunda status'u failed olarak güncelle
            if upload:
                upload.processing_status = ProcessingStatus.FAILED
                upload.processing_metadata = {
                    "error": str(e),
                    "failed_at": datetime.utcnow().isoformat(),
                    "error_type": type(e).__name__
                }
                await db.commit()
            
            raise e
    
    async def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF'den metin çıkar"""
        try:
            text_content = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        # Sayfa numarası ekle
                        text_content.append(f"--- PAGE {page_num + 1} ---\n{text}")
            
            extracted_text = "\n".join(text_content)
            logger.info(f"📄 Successfully extracted text from {len(pdf.pages)} pages")
            return extracted_text
            
        except Exception as e:
            logger.error(f"❌ PDF text extraction failed: {e}", exc_info=True)
            raise Exception(f"PDF text extraction failed: {str(e)}")
    
    async def _extract_questions_from_text(
        self, 
        text: str, 
        subject: Subject,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Metinden domain-specific soruları çıkar"""
        
        try:
            logger.info(f"🔍 Extracting questions for subject: {subject.value}")
            
            # LLM Gateway kullanarak domain-specific soru çıkarma
            llm_result = await llm_gateway.extract_questions_from_text(
                text=text,
                subject=subject.value,
                user_id=user_id
            )
            
            if llm_result.get("success"):
                questions = llm_result.get("questions", [])
                # LLM sonuçlarını domain-specific olarak işle
                processed_questions = await self._process_llm_questions(questions, subject)
                logger.info(f"✅ LLM extracted {len(processed_questions)} questions")
                return processed_questions
            else:
                logger.warning(f"⚠️ LLM question extraction failed, using fallback")
                # Fallback: domain-specific soru çıkarma
                return await self._fallback_question_extraction(text, subject)
                
        except Exception as e:
            logger.error(f"❌ LLM question extraction failed: {e}", exc_info=True)
            # Fallback kullan
            return await self._fallback_question_extraction(text, subject)
    
    async def _process_llm_questions(
        self, 
        questions: List[Dict[str, Any]], 
        subject: Subject
    ) -> List[Dict[str, Any]]:
        """LLM'den gelen soruları domain-specific olarak işle"""
        
        processed_questions = []
        
        for question in questions:
            try:
                # Domain-specific metadata ekle
                enhanced_question = await self._enhance_question_metadata(question, subject)
                processed_questions.append(enhanced_question)
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to process question: {e}")
                continue
        
        return processed_questions
    
    async def _enhance_question_metadata(
        self, 
        question: Dict[str, Any], 
        subject: Subject
    ) -> Dict[str, Any]:
        """Soru metadata'sını domain-specific olarak geliştir"""
        
        try:
            content = question.get("content", "")
            
            # Domain-specific topic detection
            topic = self._detect_topic(content, subject)
            
            # Domain-specific difficulty detection
            difficulty = self._detect_difficulty(content, subject)
            
            # Enhanced metadata
            enhanced_question = {
                **question,
                "topic_category": topic,
                "difficulty_level": difficulty,
                "domain": subject.value,
                "extraction_method": "llm_enhanced",
                "confidence_score": question.get("confidence_score", 0.8),
                "metadata_version": "2.0.0"
            }
            
            return enhanced_question
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to enhance question metadata: {e}")
            return question
    
    def _detect_topic(self, content: str, subject: Subject) -> str:
        """İçerikten domain-specific topic tespit et"""
        
        content_lower = content.lower()
        
        if subject == Subject.MATH:
            for topic in self.metadata_patterns[Subject.MATH]['topics']:
                if topic in content_lower:
                    return topic
            return "general_math"
            
        elif subject == Subject.ENGLISH:
            for topic in self.metadata_patterns[Subject.ENGLISH]['topics']:
                if topic in content_lower:
                    return topic
            return "general_english"
        
        return "general"
    
    def _detect_difficulty(self, content: str, subject: Subject) -> int:
        """İçerikten domain-specific difficulty tespit et"""
        
        content_lower = content.lower()
        difficulty_keywords = self.metadata_patterns.get(subject, {}).get('difficulty_keywords', {})
        
        # Difficulty keyword detection
        for level, keywords in difficulty_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                if level == 'easy':
                    return 1
                elif level == 'medium':
                    return 3
                elif level == 'hard':
                    return 5
        
        # Content-based difficulty estimation
        if subject == Subject.MATH:
            # Matematik için content complexity
            if any(op in content for op in ['+', '-', '*', '/', '=']):
                if len(content) < 100:
                    return 2
                elif len(content) < 200:
                    return 3
                else:
                    return 4
            else:
                return 3
                
        elif subject == Subject.ENGLISH:
            # İngilizce için content complexity
            if len(content) < 80:
                return 2
            elif len(content) < 150:
                return 3
            else:
                return 4
        
        return 3  # Default difficulty
    
    async def _fallback_question_extraction(
        self, 
        text: str, 
        subject: Subject
    ) -> List[Dict[str, Any]]:
        """Domain-specific basit soru çıkarma (LLM başarısız olduğunda)"""
        
        try:
            logger.info(f"🔄 Using fallback question extraction for {subject.value}")
            
            questions = []
            lines = text.split('\n')
            
            # Domain-specific question indicators
            indicators = self.question_indicators.get(subject, self.question_indicators[Subject.GENERAL])
            
            for i, line in enumerate(lines):
                line = line.strip()
                if any(indicator in line.lower() for indicator in indicators):
                    if len(line) > 20:  # Minimum uzunluk kontrolü
                        
                        # Domain-specific metadata
                        topic = self._detect_topic(line, subject)
                        difficulty = self._detect_difficulty(line, subject)
                        
                        question = {
                            "content": line,
                            "question_type": "multiple_choice",
                            "difficulty_level": difficulty,
                            "topic_category": topic,
                            "correct_answer": "",
                            "options": [],
                            "source_type": "pdf",
                            "domain": subject.value,
                            "extraction_method": "fallback_pattern",
                            "confidence_score": 0.6
                        }
                        
                        questions.append(question)
            
            logger.info(f"🔄 Fallback extraction found {len(questions)} questions")
            return questions[:10]  # Maksimum 10 soru
            
        except Exception as e:
            logger.error(f"❌ Fallback question extraction failed: {e}")
            return []
    
    async def _save_questions_to_db(
        self, 
        db: AsyncSession, 
        questions: List[Dict[str, Any]], 
        upload: PDFUpload,
        user_id: str
    ) -> List[Question]:
        """Soruları veritabanına kaydet ve embedding oluştur"""
        
        saved_questions = []
        
        for question_data in questions:
            try:
                # Question object oluştur
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
                        "confidence_score": question_data.get("confidence_score", 0.5),
                        "domain": question_data.get("domain", upload.subject.value),
                        "topic_category": question_data.get("topic_category", "general"),
                        "difficulty_level": question_data.get("difficulty_level", 3),
                        "metadata_version": "2.0.0"
                    }
                )
                
                db.add(question)
                saved_questions.append(question)
                
                logger.debug(f"✅ Question saved: {question.id}")
                
            except Exception as e:
                logger.error(f"❌ Failed to save question: {e}")
                continue
        
        await db.commit()
        logger.info(f"💾 Saved {len(saved_questions)} questions to database")
        return saved_questions
    
    def _calculate_quality_score(self, questions: List[Dict[str, Any]]) -> float:
        """Domain-specific soru kalitesi skorunu hesapla"""
        
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
            
            # Domain-specific bonus
            if question.get("domain") and question.get("topic_category"):
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
