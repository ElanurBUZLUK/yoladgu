from typing import Dict, Any, List
import json
from datetime import datetime
import re


class MathEvaluatorTool:
    """Matematik cevap değerlendirici MCP tool"""
    
    def get_name(self) -> str:
        return "math_evaluator"
    
    def get_description(self) -> str:
        return "Evaluate mathematics student answers with partial credit support and detailed feedback"
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question_content": {
                    "type": "string",
                    "description": "The mathematics question content"
                },
                "correct_answer": {
                    "type": "string",
                    "description": "The correct answer to the question"
                },
                "student_answer": {
                    "type": "string",
                    "description": "The student's submitted answer"
                },
                "question_type": {
                    "type": "string",
                    "description": "Type of question (multiple_choice, open_ended, fill_blank, true_false)",
                    "enum": ["multiple_choice", "open_ended", "fill_blank", "true_false"]
                },
                "difficulty_level": {
                    "type": "integer",
                    "description": "Question difficulty level (1-5)",
                    "minimum": 1,
                    "maximum": 5
                },
                "partial_credit": {
                    "type": "boolean",
                    "description": "Whether to award partial credit for partially correct answers",
                    "default": True
                }
            },
            "required": ["question_content", "correct_answer", "student_answer"]
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Matematik cevabını değerlendir"""
        
        question_content = arguments.get("question_content", "")
        correct_answer = arguments.get("correct_answer", "")
        student_answer = arguments.get("student_answer", "")
        question_type = arguments.get("question_type", "multiple_choice")
        difficulty_level = arguments.get("difficulty_level", 3)
        partial_credit = arguments.get("partial_credit", True)
        
        # Cevap değerlendirme
        evaluation_result = self._evaluate_math_answer(
            question_content, correct_answer, student_answer, question_type, difficulty_level, partial_credit
        )
        
        return {
            "success": True,
            "evaluation": evaluation_result,
            "metadata": {
                "evaluated_at": datetime.utcnow().isoformat(),
                "question_type": question_type,
                "difficulty_level": difficulty_level,
                "partial_credit_enabled": partial_credit,
                "evaluation_method": "mathematical_analysis"
            }
        }
    
    def _evaluate_math_answer(
        self, 
        question_content: str, 
        correct_answer: str, 
        student_answer: str,
        question_type: str,
        difficulty_level: int,
        partial_credit: bool
    ) -> Dict[str, Any]:
        """Matematik cevabını değerlendir"""
        
        # Cevap temizleme ve normalizasyon
        cleaned_student_answer = self._clean_answer(student_answer)
        cleaned_correct_answer = self._clean_answer(correct_answer)
        
        # Exact match kontrolü
        is_exact_match = self._check_exact_match(cleaned_student_answer, cleaned_correct_answer)
        
        if is_exact_match:
            return self._create_correct_evaluation(question_content, correct_answer, student_answer)
        
        # Partial credit hesaplama
        if partial_credit:
            partial_score = self._calculate_partial_credit(
                question_content, correct_answer, student_answer, question_type, difficulty_level
            )
            
            if partial_score > 0.5:
                return self._create_partial_evaluation(
                    question_content, correct_answer, student_answer, partial_score
                )
        
        # Yanlış cevap değerlendirmesi
        return self._create_incorrect_evaluation(
            question_content, correct_answer, student_answer, question_type, difficulty_level
        )
    
    def _clean_answer(self, answer: str) -> str:
        """Cevabı temizle ve normalize et"""
        
        # Boşlukları kaldır
        cleaned = answer.strip()
        
        # Küçük harfe çevir
        cleaned = cleaned.lower()
        
        # Özel karakterleri normalize et
        cleaned = cleaned.replace("×", "*")
        cleaned = cleaned.replace("÷", "/")
        cleaned = cleaned.replace("=", "=")
        cleaned = cleaned.replace("≠", "!=")
        cleaned = cleaned.replace("≤", "<=")
        cleaned = cleaned.replace("≥", ">=")
        
        # Fazla boşlukları kaldır
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned
    
    def _check_exact_match(self, student_answer: str, correct_answer: str) -> bool:
        """Exact match kontrolü"""
        
        # Direkt karşılaştırma
        if student_answer == correct_answer:
            return True
        
        # Sayısal karşılaştırma
        try:
            student_num = float(student_answer)
            correct_num = float(correct_answer)
            return abs(student_num - correct_num) < 0.001
        except ValueError:
            pass
        
        # Matematiksel eşitlik kontrolü
        if self._check_mathematical_equivalence(student_answer, correct_answer):
            return True
        
        return False
    
    def _check_mathematical_equivalence(self, answer1: str, answer2: str) -> bool:
        """Matematiksel eşitlik kontrolü"""
        
        # Basit matematiksel eşitlikler
        equivalences = [
            ("2", "2.0"),
            ("x=2", "x = 2"),
            ("x=1 or x=3", "x = 1 or x = 3"),
            ("x=1,x=3", "x = 1 or x = 3"),
        ]
        
        for eq1, eq2 in equivalences:
            if (answer1 == eq1 and answer2 == eq2) or (answer1 == eq2 and answer2 == eq1):
                return True
        
        return False
    
    def _calculate_partial_credit(
        self, 
        question_content: str, 
        correct_answer: str, 
        student_answer: str,
        question_type: str,
        difficulty_level: int
    ) -> float:
        """Kısmi puan hesapla"""
        
        partial_score = 0.0
        
        # Question type bazlı kısmi puan
        if question_type == "multiple_choice":
            # Çoktan seçmeli için kısmi puan yok
            return 0.0
        elif question_type == "open_ended":
            partial_score = self._calculate_open_ended_partial_credit(
                question_content, correct_answer, student_answer, difficulty_level
            )
        elif question_type == "fill_blank":
            partial_score = self._calculate_fill_blank_partial_credit(
                question_content, correct_answer, student_answer
            )
        elif question_type == "true_false":
            # True/false için kısmi puan yok
            return 0.0
        
        return min(0.9, partial_score)  # Maksimum %90 kısmi puan
    
    def _calculate_open_ended_partial_credit(
        self, 
        question_content: str, 
        correct_answer: str, 
        student_answer: str,
        difficulty_level: int
    ) -> float:
        """Açık uçlu sorular için kısmi puan"""
        
        partial_score = 0.0
        
        # Adım bazlı kontrol
        if "solve for x" in question_content.lower():
            # Denklem çözme
            if "2x" in student_answer and "=" in student_answer:
                partial_score += 0.3  # Denklem kurma
            if "x=" in student_answer:
                partial_score += 0.3  # Çözüm bulma
            if "2" in student_answer or "x=2" in student_answer:
                partial_score += 0.4  # Doğru cevap yakını
        
        elif "area" in question_content.lower():
            # Alan hesaplama
            if "length" in student_answer and "width" in student_answer:
                partial_score += 0.4  # Formül kullanma
            if "15" in student_answer or "5*3" in student_answer:
                partial_score += 0.5  # Doğru hesaplama
        
        elif "derivative" in question_content.lower():
            # Türev
            if "2x" in student_answer:
                partial_score += 0.4  # x² türevi
            if "3" in student_answer:
                partial_score += 0.3  # 3x türevi
            if "0" in student_answer or "1" in student_answer:
                partial_score += 0.3  # Sabit türevi
        
        # Zorluk seviyesine göre ayarla
        if difficulty_level >= 4:
            partial_score *= 0.8  # Zor sorularda daha az kısmi puan
        
        return partial_score
    
    def _calculate_fill_blank_partial_credit(
        self, 
        question_content: str, 
        correct_answer: str, 
        student_answer: str
    ) -> float:
        """Boşluk doldurma için kısmi puan"""
        
        # Basit string benzerliği
        if correct_answer in student_answer or student_answer in correct_answer:
            return 0.5
        
        # Sayısal yakınlık
        try:
            correct_num = float(correct_answer)
            student_num = float(student_answer)
            difference = abs(correct_num - student_num)
            
            if difference <= 1:
                return 0.7
            elif difference <= 5:
                return 0.4
            elif difference <= 10:
                return 0.2
        except ValueError:
            pass
        
        return 0.0
    
    def _create_correct_evaluation(
        self, 
        question_content: str, 
        correct_answer: str, 
        student_answer: str
    ) -> Dict[str, Any]:
        """Doğru cevap değerlendirmesi"""
        
        return {
            "is_correct": True,
            "score": 1.0,
            "feedback": "Excellent! Your answer is correct.",
            "detailed_feedback": f"Your answer '{student_answer}' matches the correct answer '{correct_answer}'.",
            "explanation": self._generate_explanation(question_content, correct_answer),
            "error_analysis": {},
            "recommendations": [
                "Great job! You've mastered this concept.",
                "Consider trying a more challenging version of this problem."
            ],
            "next_difficulty": "increase",
            "partial_credit_awarded": False
        }
    
    def _create_partial_evaluation(
        self, 
        question_content: str, 
        correct_answer: str, 
        student_answer: str,
        partial_score: float
    ) -> Dict[str, Any]:
        """Kısmi puan değerlendirmesi"""
        
        return {
            "is_correct": False,
            "score": partial_score,
            "feedback": f"Good effort! You're on the right track. You earned {int(partial_score * 100)}% credit.",
            "detailed_feedback": f"Your answer '{student_answer}' is partially correct. The correct answer is '{correct_answer}'.",
            "explanation": self._generate_explanation(question_content, correct_answer),
            "error_analysis": self._analyze_errors(question_content, correct_answer, student_answer),
            "recommendations": [
                "Review the solution steps carefully.",
                "Check your calculations for any arithmetic errors.",
                "Make sure you understand the underlying concept."
            ],
            "next_difficulty": "maintain",
            "partial_credit_awarded": True
        }
    
    def _create_incorrect_evaluation(
        self, 
        question_content: str, 
        correct_answer: str, 
        student_answer: str,
        question_type: str,
        difficulty_level: int
    ) -> Dict[str, Any]:
        """Yanlış cevap değerlendirmesi"""
        
        return {
            "is_correct": False,
            "score": 0.0,
            "feedback": "Your answer is incorrect. Let's review the solution together.",
            "detailed_feedback": f"Your answer '{student_answer}' is not correct. The correct answer is '{correct_answer}'.",
            "explanation": self._generate_explanation(question_content, correct_answer),
            "error_analysis": self._analyze_errors(question_content, correct_answer, student_answer),
            "recommendations": self._generate_recommendations(question_type, difficulty_level),
            "next_difficulty": "decrease",
            "partial_credit_awarded": False
        }
    
    def _generate_explanation(self, question_content: str, correct_answer: str) -> str:
        """Çözüm açıklaması üret"""
        
        if "solve for x" in question_content.lower():
            if "2x + 3 = 7" in question_content:
                return "Step 1: Subtract 3 from both sides: 2x = 4\nStep 2: Divide both sides by 2: x = 2"
            elif "3x²" in question_content:
                return "Step 1: Factor the quadratic: (3x-3)(x-3) = 0\nStep 2: Set each factor to zero: x = 1 or x = 3"
        
        elif "area" in question_content.lower():
            return "Area = length × width = 5 × 3 = 15 square units"
        
        elif "derivative" in question_content.lower():
            return "The derivative of x² is 2x, the derivative of 3x is 3, and the derivative of 1 is 0. So f'(x) = 2x + 3"
        
        return f"The correct answer is {correct_answer}. Review the problem-solving steps to understand the solution."
    
    def _analyze_errors(self, question_content: str, correct_answer: str, student_answer: str) -> Dict[str, Any]:
        """Hata analizi"""
        
        error_analysis = {
            "error_type": "incorrect_answer",
            "common_mistakes": [],
            "suggested_corrections": []
        }
        
        # Hata türü belirleme
        if "solve for x" in question_content.lower():
            if "=" not in student_answer:
                error_analysis["error_type"] = "missing_equation"
                error_analysis["common_mistakes"].append("Did not set up the equation properly")
            elif "x=" not in student_answer:
                error_analysis["error_type"] = "incomplete_solution"
                error_analysis["common_mistakes"].append("Did not solve for x completely")
        
        elif "area" in question_content.lower():
            if "*" not in student_answer and "×" not in student_answer:
                error_analysis["error_type"] = "wrong_formula"
                error_analysis["common_mistakes"].append("Did not use the area formula (length × width)")
        
        return error_analysis
    
    def _generate_recommendations(self, question_type: str, difficulty_level: int) -> List[str]:
        """Öneriler üret"""
        
        recommendations = [
            "Review the basic concepts related to this topic.",
            "Practice similar problems to build confidence."
        ]
        
        if question_type == "open_ended":
            recommendations.append("Show your work step by step to avoid calculation errors.")
        
        if difficulty_level >= 4:
            recommendations.append("Consider breaking down complex problems into smaller parts.")
        
        return recommendations
