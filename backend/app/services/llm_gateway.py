import logging
from typing import Dict, Any, List, Optional
from .llm_providers.llm_router import llm_router
from .llm_providers.base import LLMResponse

logger = logging.getLogger(__name__)


class LLMGatewayService:
    """LLM Gateway Service - Tüm LLM işlemleri için tek giriş noktası"""
    
    def __init__(self):
        self.router = llm_router
    
    async def generate_english_question(
        self,
        user_id: str,
        error_patterns: List[str],
        difficulty_level: int,
        question_type: str = "multiple_choice",
        topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """İngilizce soru üretimi"""
        
        # Prompt oluştur
        prompt = self._create_question_generation_prompt(
            error_patterns, difficulty_level, question_type, topic
        )
        
        system_prompt = """You are an expert English teacher creating personalized questions for students. 
        Focus on the student's specific error patterns and create engaging, educational questions.
        Always respond in Turkish for instructions and explanations."""
        
        # Schema tanımla
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "content": {"type": "string"},
                "question_type": {"type": "string"},
                "difficulty_level": {"type": "integer"},
                "options": {"type": "array", "items": {"type": "string"}},
                "correct_answer": {"type": "string"},
                "explanation": {"type": "string"},
                "focus_areas": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["content", "question_type", "correct_answer"]
        }
        
        result = await self.router.generate_structured_with_fallback(
            task_type="question_generation",
            prompt=prompt,
            schema=schema,
            system_prompt=system_prompt,
            complexity="medium",
            error_patterns=error_patterns,
            difficulty_level=difficulty_level
        )
        
        return result
    
    async def evaluate_student_answer(
        self,
        question_content: str,
        correct_answer: str,
        student_answer: str,
        subject: str,
        question_type: str = "multiple_choice"
    ) -> Dict[str, Any]:
        """Öğrenci cevabını değerlendir"""
        
        prompt = self._create_evaluation_prompt(
            question_content, correct_answer, student_answer, subject, question_type
        )
        
        system_prompt = """You are an expert teacher evaluating student answers. 
        Provide detailed, constructive feedback in Turkish. 
        Be encouraging while pointing out areas for improvement."""
        
        schema = {
            "type": "object",
            "properties": {
                "is_correct": {"type": "boolean"},
                "score": {"type": "number"},
                "feedback": {"type": "string"},
                "error_analysis": {
                    "type": "object",
                    "properties": {
                        "grammar_errors": {"type": "array", "items": {"type": "string"}},
                        "vocabulary_errors": {"type": "array", "items": {"type": "string"}},
                        "conceptual_errors": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "recommendations": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["is_correct", "score", "feedback"]
        }
        
        result = await self.router.generate_structured_with_fallback(
            task_type="answer_evaluation",
            prompt=prompt,
            schema=schema,
            system_prompt=system_prompt,
            complexity="low",
            question=question_content,
            student_answer=student_answer,
            correct_answer=correct_answer
        )
        
        return result
    
    async def analyze_student_performance(
        self,
        user_id: str,
        recent_attempts: List[Dict[str, Any]],
        subject: str
    ) -> Dict[str, Any]:
        """Öğrenci performansını analiz et"""
        
        prompt = self._create_performance_analysis_prompt(
            user_id, recent_attempts, subject
        )
        
        system_prompt = """You are an educational data analyst. 
        Analyze student performance patterns and provide actionable insights in Turkish.
        Focus on learning trends and improvement recommendations."""
        
        schema = {
            "type": "object",
            "properties": {
                "current_level": {"type": "integer"},
                "confidence": {"type": "number"},
                "strengths": {"type": "array", "items": {"type": "string"}},
                "weaknesses": {"type": "array", "items": {"type": "string"}},
                "recommendations": {"type": "array", "items": {"type": "string"}},
                "next_difficulty": {"type": "integer"}
            },
            "required": ["current_level", "confidence", "recommendations"]
        }
        
        result = await self.router.generate_structured_with_fallback(
            task_type="content_analysis",
            prompt=prompt,
            schema=schema,
            system_prompt=system_prompt,
            complexity="medium"
        )
        
        return result
    
    async def generate_feedback(
        self,
        question: str,
        student_answer: str,
        is_correct: bool,
        subject: str
    ) -> Dict[str, Any]:
        """Öğrenci için geri bildirim üret"""
        
        prompt = f"""
        Soru: {question}
        Öğrenci Cevabı: {student_answer}
        Doğru mu: {is_correct}
        Ders: {subject}
        
        Bu öğrenci için yapıcı, teşvik edici ve eğitici bir geri bildirim oluştur.
        Eğer yanlışsa, doğru cevabı açıkla ve neden yanlış olduğunu belirt.
        Eğer doğruysa, tebrik et ve konuyla ilgili ek bilgi ver.
        """
        
        system_prompt = "Sen deneyimli bir öğretmensin. Öğrencilere yapıcı geri bildirim veriyorsun."
        
        result = await self.router.generate_with_fallback(
            task_type="quick_classification",
            prompt=prompt,
            system_prompt=system_prompt,
            complexity="low"
        )
        
        return result
    
    def _create_question_generation_prompt(
        self,
        error_patterns: List[str],
        difficulty_level: int,
        question_type: str,
        topic: Optional[str] = None
    ) -> str:
        """Soru üretimi prompt'u oluştur"""
        
        error_text = ", ".join(error_patterns) if error_patterns else "genel grammar"
        topic_text = f" konusunda" if topic else ""
        
        prompt = f"""
        Bir İngilizce öğrencisi için {question_type} tipinde soru oluştur.
        
        Öğrencinin hata yaptığı alanlar: {error_text}
        Zorluk seviyesi: {difficulty_level}/5
        Konu{topic_text}: {topic or "genel"}
        
        Soru şu özelliklere sahip olmalı:
        - Öğrencinin hata yaptığı alanlara odaklanmalı
        - Zorluk seviyesine uygun olmalı
        - Eğitici ve net olmalı
        - Türkçe açıklamalar içermeli
        
        {self._get_question_type_instructions(question_type)}
        """
        
        return prompt
    
    def _create_evaluation_prompt(
        self,
        question: str,
        correct_answer: str,
        student_answer: str,
        subject: str,
        question_type: str
    ) -> str:
        """Değerlendirme prompt'u oluştur"""
        
        return f"""
        Aşağıdaki öğrenci cevabını değerlendir:
        
        Soru: {question}
        Doğru Cevap: {correct_answer}
        Öğrenci Cevabı: {student_answer}
        Ders: {subject}
        Soru Tipi: {question_type}
        
        Değerlendirme kriterleri:
        - Cevabın doğruluğu
        - Yapılan hatalar (grammar, vocabulary, kavramsal)
        - Puan (0-100 arası)
        - Yapıcı geri bildirim
        - İyileştirme önerileri
        
        Geri bildirim Türkçe olmalı ve teşvik edici olmalı.
        """
    
    def _create_performance_analysis_prompt(
        self,
        user_id: str,
        recent_attempts: List[Dict[str, Any]],
        subject: str
    ) -> str:
        """Performans analizi prompt'u oluştur"""
        
        attempts_summary = []
        for attempt in recent_attempts[-10:]:  # Son 10 deneme
            attempts_summary.append(
                f"Doğru: {attempt.get('is_correct', False)}, "
                f"Zorluk: {attempt.get('difficulty_level', 1)}, "
                f"Süre: {attempt.get('time_spent', 0)}s"
            )
        
        attempts_text = "\n".join(attempts_summary)
        
        return f"""
        Öğrenci performans analizi:
        
        Öğrenci ID: {user_id}
        Ders: {subject}
        Son denemeler:
        {attempts_text}
        
        Analiz et:
        - Mevcut seviye (1-5 arası)
        - Güven seviyesi (0-1 arası)
        - Güçlü yönler
        - Zayıf yönler
        - Gelişim önerileri
        - Önerilen sonraki zorluk seviyesi
        
        Analiz Türkçe olmalı ve yapıcı olmalı.
        """
    
    def _get_question_type_instructions(self, question_type: str) -> str:
        """Soru tipine göre özel talimatlar"""
        
        instructions = {
            "multiple_choice": """
            Çoktan seçmeli soru oluştur:
            - 4 seçenek (A, B, C, D)
            - Sadece bir doğru cevap
            - Çeldirici seçenekler mantıklı olmalı
            - Seçenekler benzer uzunlukta olmalı
            """,
            "fill_blank": """
            Boşluk doldurma sorusu oluştur:
            - Boşluk _____ ile gösterilmeli
            - Tek kelime veya kısa ifade cevabı
            - Bağlam yeterli ipucu vermeli
            - Doğru cevap net olmalı
            """,
            "open_ended": """
            Açık uçlu soru oluştur:
            - Yaratıcı düşünmeyi teşvik etmeli
            - Birden fazla doğru cevap olabilir
            - Değerlendirme kriterleri belirtilmeli
            - Öğrenciyi düşündürmeli
            """
        }
        
        return instructions.get(question_type, "")
    
    async def extract_questions_from_text(
        self,
        text: str,
        subject: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Metinden soruları çıkar"""
        
        prompt = self._create_question_extraction_prompt(text, subject)
        
        system_prompt = """You are an expert teacher extracting questions from educational content. 
        Analyze the text and identify potential questions that can be used for assessment.
        Always respond in Turkish for instructions and explanations."""
        
        schema = {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "question_type": {"type": "string"},
                            "difficulty_level": {"type": "integer"},
                            "topic_category": {"type": "string"},
                            "correct_answer": {"type": "string"},
                            "options": {"type": "array", "items": {"type": "string"}},
                            "extraction_method": {"type": "string"},
                            "confidence_score": {"type": "number"}
                        },
                        "required": ["content", "question_type", "difficulty_level"]
                    }
                },
                "extraction_summary": {
                    "type": "object",
                    "properties": {
                        "total_questions": {"type": "integer"},
                        "subject": {"type": "string"},
                        "text_length": {"type": "integer"},
                        "extraction_quality": {"type": "string"}
                    }
                }
            },
            "required": ["questions"]
        }
        
        result = await self.router.generate_structured_with_fallback(
            task_type="content_analysis",
            prompt=prompt,
            schema=schema,
            system_prompt=system_prompt,
            complexity="high",
            text_length=len(text)
        )
        
        return result
    
    def _create_question_extraction_prompt(self, text: str, subject: str) -> str:
        """Soru çıkarma prompt'u oluştur"""
        
        return f"""
        Aşağıdaki metinden {subject} dersi için sorular çıkar:
        
        Metin:
        {text[:2000]}...
        
        Talimatlar:
        1. Metindeki ana konuları belirle
        2. Her konu için 1-2 soru oluştur
        3. Sorular farklı zorluk seviyelerinde olmalı (1-5)
        4. Çoktan seçmeli, boşluk doldurma ve açık uçlu sorular dahil et
        5. Her soru için doğru cevap belirt
        6. Çoktan seçmeli sorular için 4 seçenek oluştur
        
        Soru kalitesi kriterleri:
        - Net ve anlaşılır olmalı
        - Öğrenmeyi ölçmeli
        - Yaş grubuna uygun olmalı
        - Çeldirici seçenekler mantıklı olmalı
        
        Maksimum 10 soru çıkar.
        """
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Servis durumu"""
        return await self.router.get_provider_status()
    
    async def health_check(self) -> Dict[str, Any]:
        """Sağlık kontrolü"""
        return await self.router.health_check()

    # RAG için gerekli metodlar
    async def embed_query(self, query: str) -> List[float]:
        """Sorgu için embedding üret"""
        try:
            from app.services.embedding_service import embedding_service
            embeddings = await embedding_service.embed_texts([query])
            logger.info(f"embed_query successful for query: {query[:50]}...")
            return embeddings[0] if embeddings else []
        except Exception as e:
            logger.error(f"Error in embed_query for query '{query[:50]}...': {e}", exc_info=True)
            return []

    async def compress_context(self, context: str, max_len: int = 1600) -> str:
        """Bağlamı sıkıştır"""
        if len(context) <= max_len:
            logger.info("Context already within max_len, no compression needed.")
            return context

        prompt = f"""
        Aşağıdaki metni {max_len} karaktere sıkıştır, önemli bilgileri koru:

        {context}
        """

        system_prompt = "Sen bir metin sıkıştırma uzmanısın. Önemli bilgileri koruyarak metni kısalt."

        try:
            result = await self.router.generate_with_fallback(
                task_type="content_analysis",
                prompt=prompt,
                system_prompt=system_prompt,
                complexity="low"
            )
            compressed_text = result.get("text", context[:max_len])
            logger.info(f"compress_context successful. Original length: {len(context)}, Compressed length: {len(compressed_text)}")
            return compressed_text
        except Exception as e:
            logger.error(f"Error in compress_context: {e}", exc_info=True)
            return context[:max_len] # Fallback to simple truncation on error

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Metin üretimi"""
        try:
            full_prompt = f"Context: {context}\n\n{prompt}" if context else prompt
            result = await self.router.generate_with_fallback(
                task_type="content_generation",
                prompt=full_prompt,
                system_prompt=system_prompt,
                complexity="medium",
                temperature=temperature,
                max_tokens=max_tokens
            )
            logger.info(f"generate_text successful. Usage: {result.get('usage', {})}")
            return {
                "success": True,
                "text": result.get("text", ""),
                "usage": result.get("usage", {}),
                "error": None
            }
        except Exception as e:
            logger.error(f"Error in generate_text: {e}", exc_info=True)
            return {
                "success": False,
                "text": "",
                "usage": {},
                "error": str(e)
            }

    async def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        schema: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """JSON üretimi"""
        try:
            full_prompt = f"Context: {context}\n\n{prompt}" if context else prompt
            result = await self.router.generate_structured_with_fallback(
                task_type="structured_generation",
                prompt=full_prompt,
                schema=schema or {},
                system_prompt=system_prompt,
                complexity="medium",
                temperature=temperature,
                max_tokens=max_tokens
            )
            logger.info(f"generate_json successful. Usage: {result.get('usage', {})}")
            return {
                "success": True,
                "parsed_json": result.get("parsed_data", {}),
                "usage": result.get("usage", {}),
                "error": None
            }
        except Exception as e:
            logger.error(f"Error in generate_json: {e}", exc_info=True)
            return {
                "success": False,
                "parsed_json": {},
                "usage": {},
                "error": str(e)
            }


# Global LLM gateway instance
llm_gateway = LLMGatewayService()
