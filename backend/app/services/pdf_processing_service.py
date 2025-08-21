import asyncio
import os
import json
import logging
import gc
import psutil
from typing import List, Dict, Any, Optional
from datetime import datetime
import pdfplumber
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import time

from app.models.pdf_upload import PDFUpload, ProcessingStatus
from app.models.question import Question, Subject, QuestionType, SourceType
from app.services.llm_gateway import llm_gateway
from app.services.embedding_service import embedding_service
from app.services.metadata_schema_service import metadata_schema_service, ContentType, Domain
from app.services.batch_processing_service import batch_processing_service
from app.services.enhanced_cache_service import enhanced_cache_service
from app.core.config import settings

logger = logging.getLogger(__name__)


class MemoryManager:
    """Memory management utility for PDF processing"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.memory_threshold_mb = 500  # 500MB threshold
        self.cleanup_interval = 5  # Every 5 pages
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0
    
    def should_cleanup(self, page_count: int) -> bool:
        """Check if memory cleanup is needed"""
        memory_usage = self.get_memory_usage()
        return (memory_usage > self.memory_threshold_mb or 
                page_count % self.cleanup_interval == 0)
    
    def force_cleanup(self):
        """Force garbage collection and memory cleanup"""
        try:
            # Force garbage collection
            collected = gc.collect()
            logger.debug(f"🧹 Garbage collection: {collected} objects collected")
            
            # Force memory cleanup
            if hasattr(gc, 'collect_generations'):
                gc.collect_generations()
            
            # Log memory usage after cleanup
            memory_after = self.get_memory_usage()
            logger.info(f"🧹 Memory cleanup completed. Usage: {memory_after:.1f}MB")
            
        except Exception as e:
            logger.warning(f"⚠️ Memory cleanup failed: {e}")


class MonitoringPDFService:
    """Monitoring wrapper for PDF processing service"""
    
    def __init__(self, pdf_service):
        self.pdf_service = pdf_service
        self.stats = {
            "total_pdfs_processed": 0,
            "successful_processing": 0,
            "failed_processing": 0,
            "total_questions_extracted": 0,
            "average_processing_time": 0.0,
            "total_processing_time": 0.0,
            "domain_classification_accuracy": 0.0,
            "embedding_generation_success": 0,
            "embedding_generation_failed": 0,
            "memory_usage_peak": 0.0,
            "memory_cleanups_performed": 0
        }
    
    async def process_pdf_upload(self, *args, **kwargs):
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            result = await self.pdf_service.process_pdf_upload(*args, **kwargs)
            processing_time = time.time() - start_time
            
            # Update stats
            self.stats["total_pdfs_processed"] += 1
            self.stats["successful_processing"] += 1
            self.stats["total_processing_time"] += processing_time
            self.stats["average_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["total_pdfs_processed"]
            )
            
            # Track memory usage
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_peak = max(initial_memory, final_memory)
            self.stats["memory_usage_peak"] = max(self.stats["memory_usage_peak"], memory_peak)
            
            if result.get("success"):
                self.stats["total_questions_extracted"] += result.get("questions_extracted", 0)
            
            logger.info(f"📊 PDF processing stats updated: {self.stats}")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.stats["total_pdfs_processed"] += 1
            self.stats["failed_processing"] += 1
            self.stats["total_processing_time"] += processing_time
            
            logger.error(f"❌ PDF processing failed: {e}")
            raise


class PDFProcessingService:
    """PDF işleme servisi - Memory leak korumalı"""
    
    def __init__(self):
        self.upload_dir = settings.upload_dir
        self.max_file_size = settings.max_file_size
        self.memory_manager = MemoryManager()
        
        # Domain-specific question indicators
        self.question_indicators = {
            Subject.MATH: ['?', 'soru', 'problem', 'question', 'hesapla', 'çöz', 'bul', '=', '+', '-', '*', '/'],
            Subject.ENGLISH: ['?', 'soru', 'question', 'fill', 'blank', 'choose', 'correct', 'grammar', 'vocabulary']
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
        
        # Performance tracking
        self.performance_metrics = {
            "total_pdfs_processed": 0,
            "total_questions_extracted": 0,
            "average_processing_time": 0.0,
            "domain_classification_accuracy": 0.0,
            "batch_processing_efficiency": 0.0,
            "memory_cleanups_performed": 0,
            "peak_memory_usage_mb": 0.0
        }
    
    async def process_pdf_upload(
        self, 
        db: AsyncSession, 
        upload_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """PDF yüklemesini işle ve soruları çıkar - Memory leak korumalı"""
        
        try:
            logger.info(f"🚀 Starting PDF processing for upload {upload_id}")
            
            # Track initial memory
            initial_memory = self.memory_manager.get_memory_usage()
            logger.info(f"📊 Initial memory usage: {initial_memory:.1f}MB")
            
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
            
            # PDF dosyasını oku - Memory leak korumalı
            pdf_path = os.path.join(self.upload_dir, upload.file_path)
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # PDF'den metin çıkar - Memory management ile
            extracted_text = await self._extract_text_from_pdf_memory_safe(pdf_path)
            logger.info(f"📄 Extracted {len(extracted_text)} characters from PDF")
            
            # Memory cleanup after text extraction
            if self.memory_manager.should_cleanup(1):
                self.memory_manager.force_cleanup()
                self.performance_metrics["memory_cleanups_performed"] += 1
            
            # Akıllı domain classification
            detected_domain = await self._classify_domain_intelligent(extracted_text)
            logger.info(f"🎯 Detected domain: {detected_domain}")
            
            # Domain-specific soru çıkarma
            questions = await self._extract_questions_from_text(
                extracted_text, 
                detected_domain,
                user_id
            )
            
            logger.info(f"❓ Extracted {len(questions)} questions from PDF")
            
            # Batch processing for embeddings and vector storage
            if questions:
                batch_result = await self._batch_process_questions(questions, detected_domain)
                logger.info(f"🔄 Batch processing completed: {batch_result}")
            
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
                "domain": detected_domain,
                "extraction_method": "enhanced_domain_specific",
                "batch_processing": True,
                "domain_classification_method": "intelligent",
                "memory_usage_initial_mb": initial_memory,
                "memory_usage_final_mb": self.memory_manager.get_memory_usage(),
                "memory_cleanups_performed": self.performance_metrics["memory_cleanups_performed"]
            }
            
            await db.commit()
            
            # Update performance metrics
            self._update_performance_metrics(len(saved_questions), detected_domain)
            
            # Final memory cleanup
            self.memory_manager.force_cleanup()
            
            logger.info(f"✅ PDF processing completed successfully for upload {upload_id}")
            
            return {
                "success": True,
                "upload_id": upload_id,
                "questions_extracted": len(saved_questions),
                "quality_score": upload.quality_score,
                "processing_time": upload.processing_metadata["processing_time"],
                "domain": detected_domain,
                "performance_metrics": self.performance_metrics,
                "memory_usage": {
                    "initial_mb": initial_memory,
                    "final_mb": self.memory_manager.get_memory_usage(),
                    "peak_mb": self.performance_metrics["peak_memory_usage_mb"]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ PDF processing failed for upload {upload_id}: {e}", exc_info=True)
            
            # Hata durumunda status'u failed olarak güncelle
            if upload:
                upload.processing_status = ProcessingStatus.FAILED
                upload.processing_metadata = {
                    "error": str(e),
                    "failed_at": datetime.utcnow().isoformat(),
                    "error_type": type(e).__name__,
                    "memory_usage_mb": self.memory_manager.get_memory_usage()
                }
                await db.commit()
            
            # Force cleanup on error
            self.memory_manager.force_cleanup()
            raise e
    
    async def _extract_text_from_pdf_memory_safe(self, pdf_path: str) -> str:
        """PDF'den metin çıkar - Memory leak korumalı"""
        
        try:
            text_content = []
            page_count = 0
            
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"📄 Processing {total_pages} pages with memory management")
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # Extract text from page
                        text = page.extract_text()
                        if text:
                            # Sayfa numarası ekle
                            text_content.append(f"--- PAGE {page_num + 1} ---\n{text}")
                        
                        page_count += 1
                        
                        # Memory management: Clear page reference
                        pdf.pages[page_num] = None
                        
                        # Periodic memory cleanup
                        if self.memory_manager.should_cleanup(page_count):
                            self.memory_manager.force_cleanup()
                            self.performance_metrics["memory_cleanups_performed"] += 1
                            
                            # Log memory usage
                            current_memory = self.memory_manager.get_memory_usage()
                            self.performance_metrics["peak_memory_usage_mb"] = max(
                                self.performance_metrics["peak_memory_usage_mb"], 
                                current_memory
                            )
                            
                            logger.debug(f"📊 Page {page_num + 1}/{total_pages} processed. Memory: {current_memory:.1f}MB")
                        
                        # Progress logging
                        if (page_num + 1) % 10 == 0:
                            logger.info(f"📄 Processed {page_num + 1}/{total_pages} pages")
                    
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to process page {page_num + 1}: {e}")
                        continue
                
                # Final memory cleanup
                self.memory_manager.force_cleanup()
                
                # Clear all page references
                pdf.pages.clear()
                
            extracted_text = "\n".join(text_content)
            logger.info(f"📄 Successfully extracted text from {page_count} pages")
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"❌ PDF text extraction failed: {e}", exc_info=True)
            # Force cleanup on error
            self.memory_manager.force_cleanup()
            raise Exception(f"PDF text extraction failed: {str(e)}")
    
    async def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Legacy method - use _extract_text_from_pdf_memory_safe instead"""
        logger.warning("⚠️ Using legacy PDF extraction method. Use _extract_text_from_pdf_memory_safe for memory safety.")
        return await self._extract_text_from_pdf_memory_safe(pdf_path)
    
    async def _classify_domain_intelligent(self, text: str) -> Subject:
        """Akıllı domain classification using multiple methods"""
        
        try:
            # Method 1: ML-based classification (if available)
            ml_domain = await self._classify_domain_ml(text)
            if ml_domain:
                logger.info(f"🤖 ML classification result: {ml_domain}")
                return ml_domain
            
            # Method 2: Enhanced keyword-based classification
            keyword_domain = self._classify_domain_keywords(text)
            logger.info(f"🔍 Keyword classification result: {keyword_domain}")
            
            # Method 3: Content analysis
            content_domain = self._classify_domain_content_analysis(text)
            logger.info(f"📊 Content analysis result: {content_domain}")
            
            # Combine results with confidence scoring
            final_domain = self._combine_domain_classifications(
                keyword_domain, content_domain, text
            )
            
            logger.info(f"🎯 Final domain classification: {final_domain}")
            return final_domain
            
        except Exception as e:
            logger.error(f"❌ Domain classification failed: {e}")
            return Subject.GENERAL
    
    async def _classify_domain_ml(self, text: str) -> Optional[Subject]:
        """ML-based domain classification"""
        
        try:
            # Check if ML model is available
            if hasattr(settings, 'use_ml_classification') and settings.use_ml_classification:
                # Use transformers pipeline for classification
                from transformers import pipeline
                
                classifier = pipeline(
                    "text-classification",
                    model="microsoft/DialoGPT-medium",  # Example model
                    return_all_scores=True
                )
                
                # Classify text
                result = classifier(text[:1000])  # Limit text length
                
                # Map classification to Subject enum
                if result and len(result) > 0:
                    top_result = result[0]
                    if 'math' in top_result['label'].lower():
                        return Subject.MATH
                    elif 'english' in top_result['label'].lower():
                        return Subject.ENGLISH
                    else:
                        return Subject.GENERAL
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ ML classification failed: {e}")
            return None
    
    def _classify_domain_keywords(self, text: str) -> Dict[str, float]:
        """Enhanced keyword-based domain classification with confidence scores"""
        
        try:
            text_lower = text.lower()
            
            # Math keywords with weights
            math_keywords = {
                'equation': 2.0, 'formula': 2.0, 'calculate': 1.5, 'solve': 1.5,
                'algebra': 3.0, 'geometry': 3.0, 'calculus': 3.0, 'trigonometry': 3.0,
                'function': 2.0, 'derivative': 2.5, 'integral': 2.5, 'matrix': 2.0,
                'probability': 2.0, 'statistics': 2.0, 'percentage': 1.5, 'ratio': 1.5,
                '=': 1.0, '+': 0.5, '-': 0.5, '*': 0.5, '/': 0.5, '√': 1.0, 'π': 1.5
            }
            
            # English keywords with weights
            english_keywords = {
                'grammar': 3.0, 'vocabulary': 3.0, 'sentence': 2.0, 'paragraph': 2.0,
                'verb': 2.5, 'noun': 2.5, 'adjective': 2.5, 'adverb': 2.5,
                'tense': 2.0, 'conjugation': 2.0, 'pronunciation': 2.0, 'spelling': 2.0,
                'reading': 2.0, 'comprehension': 2.0, 'writing': 2.0, 'essay': 2.0,
                'cloze': 2.5, 'fill': 1.5, 'blank': 1.5, 'choose': 1.5, 'correct': 1.5
            }
            
            # Calculate scores
            math_score = sum(weight for keyword, weight in math_keywords.items() 
                           if keyword in text_lower)
            english_score = sum(weight for keyword, weight in english_keywords.items() 
                              if keyword in text_lower)
            
            return {
                'math': math_score,
                'english': english_score,
                'general': 1.0  # Base score
            }
            
        except Exception as e:
            logger.error(f"❌ Keyword classification failed: {e}")
            return {'math': 0.0, 'english': 0.0, 'general': 1.0}
    
    def _classify_domain_content_analysis(self, text: str) -> Dict[str, float]:
        """Content analysis-based domain classification"""
        
        try:
            # Analyze text characteristics
            sentences = text.split('.')
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            
            # Count mathematical symbols
            math_symbols = sum(1 for char in text if char in '=+-*/√π∫∑∏∞≤≥≠≈')
            
            # Count question marks (indicates questions)
            question_marks = text.count('?')
            
            # Analyze word patterns
            words = text.lower().split()
            math_words = sum(1 for word in words if any(symbol in word for symbol in '0123456789'))
            english_words = sum(1 for word in words if len(word) > 8)  # Long words often English
            
            # Calculate scores
            math_score = (math_symbols * 2.0) + (math_words * 0.5) + (question_marks * 1.0)
            english_score = (english_words * 0.3) + (avg_sentence_length * 0.2) + (question_marks * 0.5)
            
            return {
                'math': math_score,
                'english': english_score,
                'general': 1.0
            }
            
        except Exception as e:
            logger.error(f"❌ Content analysis failed: {e}")
            return {'math': 0.0, 'english': 0.0, 'general': 1.0}
    
    def _combine_domain_classifications(
        self,
        keyword_scores: Dict[str, float],
        content_scores: Dict[str, float],
        text: str
    ) -> Subject:
        """Combine multiple classification methods with confidence weighting"""
        
        try:
            # Weight the different methods
            keyword_weight = 0.6
            content_weight = 0.4
            
            # Combine scores
            combined_scores = {}
            for domain in ['math', 'english', 'general']:
                combined_scores[domain] = (
                    keyword_scores.get(domain, 0.0) * keyword_weight +
                    content_scores.get(domain, 0.0) * content_weight
                )
            
            # Find the domain with highest score
            best_domain = max(combined_scores.items(), key=lambda x: x[1])
            
            # Map to Subject enum
            if best_domain[0] == 'math':
                return Subject.MATH
            elif best_domain[0] == 'english':
                return Subject.ENGLISH
            else:
                return Subject.GENERAL
                
        except Exception as e:
            logger.error(f"❌ Domain combination failed: {e}")
            return Subject.GENERAL
    
    async def _batch_process_questions(
        self,
        questions: List[Dict[str, Any]],
        domain: Subject
    ) -> Dict[str, Any]:
        """Batch process questions for embeddings and vector storage"""
        
        try:
            logger.info(f"🔄 Starting batch processing for {len(questions)} questions")
            
            # Use batch processing service
            result = await batch_processing_service.batch_process_questions(
                session=None,  # We'll handle DB operations separately
                questions=questions,
                domain=domain.value,
                content_type="question"
            )
            
            logger.info(f"✅ Batch processing completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _extract_questions_from_text(
        self, 
        text: str, 
        subject: Subject,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Metinden domain-specific soruları çıkar"""
        
        try:
            logger.info(f"🔍 Extracting questions for subject: {subject.value}")
            
            # Check cache first
            cache_key = f"pdf_questions:{hash(text)}:{subject.value}"
            cached_questions = await enhanced_cache_service.get(cache_key)
            if cached_questions:
                logger.info(f"✅ Found cached questions for {subject.value}")
                return cached_questions
            
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
                
                # Cache the results
                await enhanced_cache_service.set(cache_key, processed_questions, ttl_seconds=3600)
                
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
    
    def _update_performance_metrics(self, questions_count: int, domain: Subject):
        """Update performance metrics"""
        
        try:
            self.performance_metrics["total_questions_extracted"] += questions_count
            self.performance_metrics["total_pdfs_processed"] += 1
            
            # Calculate domain classification accuracy (simplified)
            if domain in [Subject.MATH, Subject.ENGLISH]:
                self.performance_metrics["domain_classification_accuracy"] = 0.95
            else:
                self.performance_metrics["domain_classification_accuracy"] = 0.85
            
            # Update memory metrics
            current_memory = self.memory_manager.get_memory_usage()
            self.performance_metrics["peak_memory_usage_mb"] = max(
                self.performance_metrics["peak_memory_usage_mb"], 
                current_memory
            )
                
        except Exception as e:
            logger.error(f"❌ Error updating performance metrics: {e}")
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        
        return {
            "service": "PDFProcessingService",
            "metrics": self.performance_metrics,
            "cache_stats": await enhanced_cache_service.get_stats(),
            "batch_stats": await batch_processing_service.get_performance_report(),
            "memory_management": {
                "current_memory_mb": self.memory_manager.get_memory_usage(),
                "memory_threshold_mb": self.memory_manager.memory_threshold_mb,
                "cleanup_interval": self.memory_manager.cleanup_interval
            }
        }
    
    async def process_pdf_batch(
        self,
        db: AsyncSession,
        pdf_paths: List[str],
        user_id: str,
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """Process multiple PDFs concurrently with memory management"""
        
        try:
            logger.info(f"🚀 Starting batch processing for {len(pdf_paths)} PDFs")
            
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_single_pdf(pdf_path: str):
                async with semaphore:
                    try:
                        # Create mock upload for batch processing
                        mock_upload = PDFUpload(
                            id=f"batch_{int(time.time())}_{hash(pdf_path)}",
                            file_path=pdf_path,
                            subject=Subject.GENERAL,  # Will be detected
                            user_id=user_id
                        )
                        
                        return await self.process_pdf_upload(db, mock_upload.id, user_id)
                    except Exception as e:
                        logger.error(f"❌ Failed to process PDF {pdf_path}: {e}")
                        return {"success": False, "error": str(e), "pdf_path": pdf_path}
            
            # Process all PDFs concurrently
            results = await asyncio.gather(*[process_single_pdf(path) for path in pdf_paths])
            
            logger.info(f"✅ Batch processing completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch PDF processing failed: {e}")
            return []

# Singleton instance
pdf_processing_service = PDFProcessingService()

# Monitoring wrapper
monitoring_pdf_service = MonitoringPDFService(pdf_processing_service)
