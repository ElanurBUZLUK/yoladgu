from typing import Dict, Any, List
import json
from datetime import datetime


class MathGeneratorTool:
    """Matematik soru üretici MCP tool"""
    
    def get_name(self) -> str:
        return "math_generator"
    
    def get_description(self) -> str:
        return "Generate mathematics questions for students based on difficulty level, topic, and learning objectives"
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Student user ID"
                },
                "difficulty_level": {
                    "type": "integer",
                    "description": "Question difficulty level (1-5)",
                    "minimum": 1,
                    "maximum": 5
                },
                "topic_category": {
                    "type": "string",
                    "description": "Mathematics topic category (e.g., algebra, geometry, calculus)"
                },
                "question_type": {
                    "type": "string",
                    "description": "Type of question (multiple_choice, open_ended, fill_blank, true_false)",
                    "enum": ["multiple_choice", "open_ended", "fill_blank", "true_false"]
                },
                "context": {
                    "type": "string",
                    "description": "Context for question generation"
                },
                "error_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Common error patterns to focus on"
                },
                "learning_objectives": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Learning objectives for the question"
                }
            },
            "required": ["user_id", "difficulty_level", "topic_category"]
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Matematik sorusu üret"""
        
        user_id = arguments.get("user_id")
        difficulty_level = arguments.get("difficulty_level", 3)
        topic_category = arguments.get("topic_category", "general")
        question_type = arguments.get("question_type", "multiple_choice")
        context = arguments.get("context", "")
        error_patterns = arguments.get("error_patterns", [])
        learning_objectives = arguments.get("learning_objectives", [])
        
        # Mock matematik soru üretimi
        question_data = self._generate_mock_math_question(
            difficulty_level, topic_category, question_type, context, error_patterns, learning_objectives
        )
        
        return {
            "success": True,
            "question": question_data,
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "difficulty_level": difficulty_level,
                "topic_category": topic_category,
                "question_type": question_type,
                "context_used": bool(context),
                "error_patterns_focused": len(error_patterns),
                "learning_objectives_addressed": len(learning_objectives)
            }
        }
    
    def _generate_mock_math_question(
        self, 
        difficulty_level: int, 
        topic_category: str, 
        question_type: str,
        context: str,
        error_patterns: List[str],
        learning_objectives: List[str]
    ) -> Dict[str, Any]:
        """Mock matematik sorusu üret"""
        
        # Topic bazlı soru şablonları
        topic_questions = {
            "algebra": {
                "easy": "Solve for x: 2x + 3 = 7",
                "medium": "Solve the equation: 3x² - 12x + 9 = 0",
                "hard": "Find all real solutions of: x³ - 6x² + 11x - 6 = 0"
            },
            "geometry": {
                "easy": "Find the area of a rectangle with length 5 and width 3",
                "medium": "Calculate the volume of a cylinder with radius 4 and height 6",
                "hard": "Find the surface area of a cone with radius 3 and slant height 5"
            },
            "calculus": {
                "easy": "Find the derivative of f(x) = x² + 3x + 1",
                "medium": "Calculate the integral ∫(2x + 3)dx",
                "hard": "Find the limit: lim(x→0) (sin(x)/x)"
            },
            "arithmetic": {
                "easy": "What is 15 + 27?",
                "medium": "Calculate 234 × 56",
                "hard": "Find the greatest common divisor of 48 and 72"
            }
        }
        
        # Difficulty mapping
        difficulty_map = {1: "easy", 2: "easy", 3: "medium", 4: "medium", 5: "hard"}
        difficulty_key = difficulty_map.get(difficulty_level, "medium")
        
        # Topic seçimi
        if topic_category in topic_questions:
            base_question = topic_questions[topic_category][difficulty_key]
        else:
            base_question = topic_questions["arithmetic"][difficulty_key]
        
        # Question type'a göre formatla
        if question_type == "multiple_choice":
            question_content = base_question
            options = self._generate_options(base_question, difficulty_level)
            correct_answer = options[0]  # İlk seçenek doğru
        elif question_type == "open_ended":
            question_content = base_question
            options = None
            correct_answer = self._calculate_answer(base_question)
        elif question_type == "fill_blank":
            question_content = base_question.replace("x", "___")
            options = None
            correct_answer = self._calculate_answer(base_question)
        else:  # true_false
            question_content = f"Is the following statement true: {base_question} = {self._calculate_answer(base_question)}?"
            options = ["True", "False"]
            correct_answer = "True"
        
        return {
            "content": question_content,
            "question_type": question_type,
            "difficulty_level": difficulty_level,
            "topic_category": topic_category,
            "correct_answer": correct_answer,
            "options": options,
            "explanation": self._generate_explanation(base_question, correct_answer),
            "hints": self._generate_hints(base_question, difficulty_level),
            "estimated_difficulty": difficulty_level + (0.1 if context else 0.0),
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "context": context,
                "error_patterns": error_patterns,
                "learning_objectives": learning_objectives,
                "generation_info": {
                    "based_on_topic": topic_category,
                    "difficulty_adjusted": difficulty_level,
                    "error_patterns_used": error_patterns,
                    "context_used": context,
                }
            }
        }
    
    def _generate_options(self, question: str, difficulty_level: int) -> List[str]:
        """Çoktan seçmeli seçenekler üret"""
        
        correct_answer = self._calculate_answer(question)
        
        if difficulty_level <= 2:
            # Kolay seviye - yakın seçenekler
            if isinstance(correct_answer, (int, float)):
                options = [
                    correct_answer,
                    correct_answer + 1,
                    correct_answer - 1,
                    correct_answer * 2
                ]
            else:
                options = [correct_answer, "A", "B", "C"]
        else:
            # Zor seviye - daha karmaşık seçenekler
            if isinstance(correct_answer, (int, float)):
                options = [
                    correct_answer,
                    correct_answer + 2,
                    correct_answer - 2,
                    correct_answer * 1.5
                ]
            else:
                options = [correct_answer, "X", "Y", "Z"]
        
        # Seçenekleri karıştır
        import random
        random.shuffle(options)
        
        return options
    
    def _calculate_answer(self, question: str) -> Any:
        """Soru için cevap hesapla"""
        
        # Basit hesaplama (gerçek implementasyonda daha karmaşık olacak)
        if "2x + 3 = 7" in question:
            return 2
        elif "3x² - 12x + 9 = 0" in question:
            return "x = 1 or x = 3"
        elif "area of a rectangle" in question:
            return 15
        elif "volume of a cylinder" in question:
            return 96
        elif "derivative" in question:
            return "2x + 3"
        elif "15 + 27" in question:
            return 42
        elif "234 × 56" in question:
            return 13104
        else:
            return "Answer"
    
    def _generate_explanation(self, question: str, answer: Any) -> str:
        """Açıklama üret"""
        
        if "2x + 3 = 7" in question:
            return "Subtract 3 from both sides: 2x = 4. Then divide by 2: x = 2"
        elif "area of a rectangle" in question:
            return "Area = length × width = 5 × 3 = 15 square units"
        elif "derivative" in question:
            return "The derivative of x² is 2x, the derivative of 3x is 3, and the derivative of 1 is 0"
        else:
            return f"The answer is {answer}. This can be solved using standard mathematical methods."
    
    def _generate_hints(self, question: str, difficulty_level: int) -> List[str]:
        """İpuçları üret"""
        
        hints = []
        
        if difficulty_level <= 2:
            hints.append("Take your time and read the question carefully")
            hints.append("Check your calculations step by step")
        elif difficulty_level <= 4:
            hints.append("Consider what mathematical concepts are involved")
            hints.append("Try to break the problem into smaller parts")
        else:
            hints.append("This is an advanced problem - use your knowledge of multiple concepts")
            hints.append("Consider alternative approaches if your first method doesn't work")
        
        return hints
