import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..schemas.cloze_question_context import ClozeQuestionGenerationContext
from .base_context_builder import BaseContextBuilder
from app.models.error_pattern import ErrorPattern
from app.models.student_attempt import StudentAttempt
from app.repositories.error_pattern_repository import ErrorPatternRepository
from app.services.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class ClozeQuestionContextBuilder(BaseContextBuilder):
    """Cloze soru üretimi için context builder"""
    
    def __init__(
        self,
        retriever: HybridRetriever,
        error_pattern_repo: ErrorPatternRepository
    ):
        super().__init__()
        self.retriever = retriever
        self.error_pattern_repo = error_pattern_repo
        self.logger = logger
    
    async def build(
        self,
        session: AsyncSession,
        user_id: str,
        num_questions: int = 1,
        last_n_errors: int = 5,
        difficulty_level: Optional[int] = None
    ) -> ClozeQuestionGenerationContext:
        """Cloze soru üretimi için context oluştur"""
        
        try:
            # 1. Kullanıcı bağlamını al
            user_context = await self._get_user_context(session, user_id)
            
            # 2. Hata pattern'larını al
            error_patterns = await self._get_user_error_patterns(
                session, user_id, last_n_errors
            )
            
            # 3. Bilgi bağlamını al
            knowledge_context = await self._get_knowledge_context(
                session, error_patterns=error_patterns
            )
            
            # 4. Çıktı format bağlamını al
            output_context = await self._get_output_format_context(
                num_questions=num_questions
            )
            
            # 5. Task definition oluştur
            task_definition = self._create_task_definition(
                num_questions, difficulty_level or user_context.get("user_level", 3)
            )
            
            # 6. Context'i oluştur
            context = ClozeQuestionGenerationContext(
                task_definition=task_definition,
                num_questions=num_questions,
                difficulty_level=difficulty_level or user_context.get("user_level", 3),
                user_id=user_id,
                user_error_patterns=error_patterns,
                user_level=user_context.get("user_level"),
                user_preferences=user_context.get("user_preferences"),
                grammar_rules=knowledge_context.get("grammar_rules", []),
                vocabulary_context=knowledge_context.get("vocabulary_context"),
                topic_context=knowledge_context.get("topic_context"),
                output_schema=output_context.get("output_schema", {}),
                output_format_context=output_context.get("format_instructions", "")
            )
            
            self.logger.info(f"ClozeQuestionContext created for user {user_id}")
            return context
            
        except Exception as e:
            self.logger.error(f"Error building ClozeQuestionContext: {e}")
            raise
    
    async def _get_user_context(
        self, 
        session: AsyncSession, 
        user_id: str
    ) -> Dict[str, Any]:
        """Kullanıcı bağlamını al"""
        
        try:
            # Kullanıcının son denemelerini al
            result = await session.execute(
                select(StudentAttempt)
                .where(StudentAttempt.user_id == user_id)
                .order_by(StudentAttempt.created_at.desc())
                .limit(10)
            )
            recent_attempts = result.scalars().all()
            
            # Kullanıcı seviyesini hesapla
            user_level = 3  # Default level
            if recent_attempts:
                correct_attempts = [a for a in recent_attempts if a.is_correct]
                if correct_attempts:
                    success_rate = len(correct_attempts) / len(recent_attempts)
                    if success_rate > 0.8:
                        user_level = 4
                    elif success_rate > 0.6:
                        user_level = 3
                    elif success_rate > 0.4:
                        user_level = 2
                    else:
                        user_level = 1
            
            return {
                "user_id": user_id,
                "user_level": user_level,
                "user_preferences": {
                    "preferred_difficulty": user_level,
                    "recent_attempts_count": len(recent_attempts)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting user context: {e}")
            return {
                "user_id": user_id,
                "user_level": 3,
                "user_preferences": {}
            }
    
    async def _get_user_error_patterns(
        self,
        session: AsyncSession,
        user_id: str,
        last_n_errors: int
    ) -> List[str]:
        """Kullanıcının son hata pattern'larını al"""
        
        try:
            # Son yanlış denemeleri al
            result = await session.execute(
                select(StudentAttempt)
                .where(
                    StudentAttempt.user_id == user_id,
                    StudentAttempt.is_correct == False
                )
                .order_by(StudentAttempt.created_at.desc())
                .limit(last_n_errors)
            )
            error_attempts = result.scalars().all()
            
            # Hata pattern'larını çıkar
            error_patterns = []
            for attempt in error_attempts:
                if attempt.error_analysis:
                    patterns = attempt.error_analysis.get("patterns", [])
                    error_patterns.extend(patterns)
            
            # Tekrarları kaldır ve en sık olanları al
            unique_patterns = list(set(error_patterns))
            return unique_patterns[:5]  # En fazla 5 pattern
            
        except Exception as e:
            self.logger.error(f"Error getting user error patterns: {e}")
            return []
    
    async def _get_knowledge_context(
        self,
        session: AsyncSession,
        error_patterns: List[str]
    ) -> Dict[str, Any]:
        """Bilgi bağlamını al"""
        
        try:
            grammar_rules = []
            vocabulary_context = None
            topic_context = None
            
            # Hata pattern'larına göre ilgili kuralları al
            if error_patterns:
                for pattern in error_patterns:
                    # RAG ile ilgili kuralları ara
                    rules = await self.retriever.retrieve_grammar_rules(pattern)
                    grammar_rules.extend(rules)
                
                # Kelime bilgisi bağlamını al
                vocabulary_context = await self.retriever.retrieve_vocabulary_context(
                    error_patterns
                )
                
                # Konu bağlamını al
                topic_context = await self.retriever.retrieve_topic_context(
                    error_patterns
                )
            
            return {
                "grammar_rules": grammar_rules[:10],  # En fazla 10 kural
                "vocabulary_context": vocabulary_context,
                "topic_context": topic_context
            }
            
        except Exception as e:
            self.logger.error(f"Error getting knowledge context: {e}")
            return {
                "grammar_rules": [],
                "vocabulary_context": None,
                "topic_context": None
            }
    
    async def _get_output_format_context(
        self,
        num_questions: int
    ) -> Dict[str, Any]:
        """Çıktı format bağlamını al"""
        
        output_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "original_sentence": {"type": "string"},
                    "cloze_sentence": {"type": "string"},
                    "correct_answer": {"type": "string"},
                    "distractors": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "explanation": {"type": "string"},
                    "difficulty_level": {"type": "integer"},
                    "error_type_addressed": {"type": "string"}
                },
                "required": [
                    "original_sentence", "cloze_sentence", "correct_answer",
                    "distractors", "explanation", "difficulty_level", "error_type_addressed"
                ]
            }
        }
        
        format_instructions = f"""
        Lütfen {num_questions} adet cloze sorusu oluştur ve JSON formatında döndür.
        
        Her soru için şu bilgileri sağla:
        - original_sentence: Orijinal cümle
        - cloze_sentence: Boşluklu cümle (___ ile boşlukları göster)
        - correct_answer: Doğru cevap
        - distractors: Yanlış şıklar (en az 3 tane)
        - explanation: Sorunun açıklaması
        - difficulty_level: Zorluk seviyesi (1-5)
        - error_type_addressed: Odaklanılan hata türü
        """
        
        return {
            "output_schema": output_schema,
            "format_instructions": format_instructions
        }
    
    def _create_task_definition(
        self,
        num_questions: int,
        difficulty_level: int
    ) -> str:
        """Task definition oluştur"""
        
        return f"""
        {num_questions} adet cloze sorusu oluştur.
        
        Soru özellikleri:
        - Zorluk seviyesi: {difficulty_level}/5
        - Soru tipi: Cloze (boşluk doldurma)
        - Ders: İngilizce
        - Odak: Öğrencinin hata yaptığı alanlar
        
        Her soru için:
        1. Orijinal cümleyi oluştur
        2. Bir kelimeyi boşluk olarak değiştir
        3. Doğru cevabı belirle
        4. Yanlış şıkları oluştur
        5. Açıklama ekle
        """
