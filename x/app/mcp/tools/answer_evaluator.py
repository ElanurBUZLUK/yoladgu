from typing import Dict, Any, List
from .base import BaseMCPTool


class AnswerEvaluatorTool(BaseMCPTool):
    """Öğrenci cevaplarını değerlendirme için MCP tool"""
    
    def get_name(self) -> str:
        return "evaluate_answer"
    
    def get_description(self) -> str:
        return "Öğrenci cevabını değerlendirir ve detaylı hata analizi yapar"
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question_content": {
                    "type": "string",
                    "description": "Soru metni"
                },
                "correct_answer": {
                    "type": "string",
                    "description": "Doğru cevap"
                },
                "student_answer": {
                    "type": "string",
                    "description": "Öğrencinin verdiği cevap"
                },
                "subject": {
                    "type": "string",
                    "enum": ["math", "english"],
                    "description": "Ders konusu"
                },
                "question_type": {
                    "type": "string",
                    "enum": ["multiple_choice", "fill_blank", "open_ended", "true_false"],
                    "description": "Soru tipi"
                },
                "difficulty_level": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Soru zorluk seviyesi"
                }
            },
            "required": ["question_content", "correct_answer", "student_answer", "subject"]
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Cevap değerlendirme mantığı"""
        
        question_content = arguments["question_content"]
        correct_answer = arguments["correct_answer"]
        student_answer = arguments["student_answer"]
        subject = arguments["subject"]
        question_type = arguments.get("question_type", "multiple_choice")
        difficulty_level = arguments.get("difficulty_level", 3)
        
        # Temel doğruluk kontrolü
        is_correct = self._check_correctness(correct_answer, student_answer, question_type)
        
        # Hata analizi
        error_analysis = await self._analyze_errors(
            question_content, correct_answer, student_answer, subject, question_type
        )
        
        # Puan hesaplama
        score = self._calculate_score(is_correct, error_analysis, difficulty_level)
        
        # Geri bildirim üretme
        feedback = self._generate_feedback(
            is_correct, error_analysis, correct_answer, student_answer, subject
        )
        
        return {
            "success": True,
            "evaluation": {
                "is_correct": is_correct,
                "score": score,
                "max_score": 100,
                "feedback": feedback,
                "error_analysis": error_analysis,
                "recommendations": self._generate_recommendations(error_analysis, subject),
                "next_difficulty": self._suggest_next_difficulty(is_correct, difficulty_level, error_analysis)
            },
            "metadata": {
                "question_type": question_type,
                "subject": subject,
                "difficulty_level": difficulty_level,
                "evaluation_timestamp": "2024-01-01T00:00:00Z"
            }
        }
    
    def _check_correctness(self, correct: str, student: str, question_type: str) -> bool:
        """Temel doğruluk kontrolü"""
        if question_type == "multiple_choice":
            return correct.strip().upper() == student.strip().upper()
        elif question_type == "true_false":
            return correct.strip().lower() == student.strip().lower()
        else:
            # Metin tabanlı cevaplar için daha esnek karşılaştırma
            return correct.strip().lower() in student.strip().lower()
    
    async def _analyze_errors(self, question: str, correct: str, student: str, subject: str, question_type: str) -> Dict[str, Any]:
        """Detaylı hata analizi"""
        
        errors = {
            "grammar_errors": [],
            "vocabulary_errors": [],
            "spelling_errors": [],
            "syntax_errors": [],
            "conceptual_errors": []
        }
        
        if subject == "english":
            # İngilizce hata analizi
            if question_type == "fill_blank" or question_type == "open_ended":
                # Grammar hataları
                if "went" in correct and "go" in student:
                    errors["grammar_errors"].append({
                        "type": "past_tense_error",
                        "description": "Past tense kullanımı hatası",
                        "expected": "went",
                        "found": "go"
                    })
                
                # Vocabulary hataları
                if len(student.split()) < len(correct.split()) * 0.7:
                    errors["vocabulary_errors"].append({
                        "type": "insufficient_vocabulary",
                        "description": "Yetersiz kelime kullanımı"
                    })
        
        elif subject == "math":
            # Matematik hata analizi
            if not self._check_correctness(correct, student, question_type):
                errors["conceptual_errors"].append({
                    "type": "calculation_error",
                    "description": "Hesaplama hatası",
                    "expected": correct,
                    "found": student
                })
        
        return errors
    
    def _calculate_score(self, is_correct: bool, error_analysis: Dict, difficulty_level: int) -> int:
        """Puan hesaplama"""
        base_score = 100 if is_correct else 0
        
        if not is_correct:
            # Kısmi puan hesaplama
            total_errors = sum(len(errors) for errors in error_analysis.values())
            if total_errors > 0:
                # Her hata için puan düşürme
                penalty = min(total_errors * 10, 80)
                base_score = max(100 - penalty, 10)
        
        # Zorluk seviyesine göre bonus
        difficulty_bonus = difficulty_level * 2 if is_correct else 0
        
        return min(base_score + difficulty_bonus, 100)
    
    def _generate_feedback(self, is_correct: bool, error_analysis: Dict, correct: str, student: str, subject: str) -> str:
        """Geri bildirim üretme"""
        if is_correct:
            return "Tebrikler! Cevabınız doğru."
        
        feedback_parts = ["Cevabınızda bazı hatalar var:"]
        
        for error_type, errors in error_analysis.items():
            if errors:
                if error_type == "grammar_errors":
                    feedback_parts.append("• Grammar hatası: Doğru zaman kullanımına dikkat edin.")
                elif error_type == "vocabulary_errors":
                    feedback_parts.append("• Kelime seçimi: Daha uygun kelimeler kullanmaya çalışın.")
                elif error_type == "spelling_errors":
                    feedback_parts.append("• Yazım hatası: Kelimelerin doğru yazımını kontrol edin.")
                elif error_type == "conceptual_errors":
                    feedback_parts.append("• Kavram hatası: Konuyu tekrar gözden geçirin.")
        
        feedback_parts.append(f"Doğru cevap: {correct}")
        
        return " ".join(feedback_parts)
    
    def _generate_recommendations(self, error_analysis: Dict, subject: str) -> List[str]:
        """Öneriler üretme"""
        recommendations = []
        
        for error_type, errors in error_analysis.items():
            if errors:
                if error_type == "grammar_errors":
                    recommendations.append("Grammar kurallarını tekrar çalışın")
                elif error_type == "vocabulary_errors":
                    recommendations.append("Kelime dağarcığınızı geliştirin")
                elif error_type == "spelling_errors":
                    recommendations.append("Yazım kurallarını pratik yapın")
                elif error_type == "conceptual_errors":
                    recommendations.append("Temel kavramları tekrar edin")
        
        if not recommendations:
            recommendations.append("Mükemmel! Çalışmaya devam edin.")
        
        return recommendations
    
    def _suggest_next_difficulty(self, is_correct: bool, current_difficulty: int, error_analysis: Dict) -> int:
        """Sonraki zorluk seviyesi önerisi"""
        total_errors = sum(len(errors) for errors in error_analysis.values())
        
        if is_correct and total_errors == 0:
            # Mükemmel cevap - zorluk artırılabilir
            return min(current_difficulty + 1, 5)
        elif is_correct and total_errors <= 2:
            # İyi cevap - aynı seviye
            return current_difficulty
        else:
            # Zayıf cevap - zorluk azaltılabilir
            return max(current_difficulty - 1, 1)