from typing import Dict, Any, List
from .base import BaseMCPTool


class QuestionGeneratorTool(BaseMCPTool):
    """İngilizce soru üretimi için MCP tool"""
    
    def get_name(self) -> str:
        return "generate_english_question"
    
    def get_description(self) -> str:
        return "İngilizce soru üretir, öğrencinin hata geçmişine göre kişiselleştirilmiş sorular oluşturur"
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Öğrenci ID'si"
                },
                "error_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Öğrencinin yaptığı hata türleri"
                },
                "difficulty_level": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Soru zorluk seviyesi (1-5)"
                },
                "question_type": {
                    "type": "string",
                    "enum": ["multiple_choice", "fill_blank", "open_ended"],
                    "description": "Soru tipi"
                },
                "focus_areas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Odaklanılacak grammar/vocabulary alanları"
                },
                "topic": {
                    "type": "string",
                    "description": "Soru konusu (opsiyonel)"
                }
            },
            "required": ["user_id", "error_patterns", "difficulty_level"]
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """İngilizce soru üretimi mantığı"""
        
        user_id = arguments["user_id"]
        error_patterns = arguments["error_patterns"]
        difficulty_level = arguments["difficulty_level"]
        question_type = arguments.get("question_type", "multiple_choice")
        focus_areas = arguments.get("focus_areas", [])
        topic = arguments.get("topic", "general")
        
        # Şimdilik mock data döndürüyoruz
        # Gerçek implementasyonda LLM API'si çağrılacak
        
        if question_type == "multiple_choice":
            question_data = {
                "id": f"gen_{user_id}_{hash(str(arguments))}",
                "content": self._generate_multiple_choice_question(error_patterns, difficulty_level, focus_areas),
                "question_type": question_type,
                "difficulty_level": difficulty_level,
                "correct_answer": "B",
                "options": ["A) went", "B) go", "C) going", "D) goes"],
                "explanation": "Past tense kullanımı için 'went' doğru cevaptır.",
                "focus_areas": focus_areas,
                "generated_for_errors": error_patterns,
                "metadata": {
                    "generated_at": "2024-01-01T00:00:00Z",
                    "user_id": user_id,
                    "topic": topic
                }
            }
        elif question_type == "fill_blank":
            question_data = {
                "id": f"gen_{user_id}_{hash(str(arguments))}",
                "content": self._generate_fill_blank_question(error_patterns, difficulty_level, focus_areas),
                "question_type": question_type,
                "difficulty_level": difficulty_level,
                "correct_answer": "went",
                "explanation": "Past tense kullanımı için 'went' doğru cevaptır.",
                "focus_areas": focus_areas,
                "generated_for_errors": error_patterns,
                "metadata": {
                    "generated_at": "2024-01-01T00:00:00Z",
                    "user_id": user_id,
                    "topic": topic
                }
            }
        else:  # open_ended
            question_data = {
                "id": f"gen_{user_id}_{hash(str(arguments))}",
                "content": self._generate_open_ended_question(error_patterns, difficulty_level, focus_areas),
                "question_type": question_type,
                "difficulty_level": difficulty_level,
                "sample_answer": "I went to school yesterday because I had an important exam.",
                "evaluation_criteria": ["Past tense usage", "Sentence structure", "Vocabulary"],
                "focus_areas": focus_areas,
                "generated_for_errors": error_patterns,
                "metadata": {
                    "generated_at": "2024-01-01T00:00:00Z",
                    "user_id": user_id,
                    "topic": topic
                }
            }
        
        return {
            "success": True,
            "question": question_data,
            "generation_info": {
                "based_on_errors": error_patterns,
                "difficulty_adjusted": difficulty_level,
                "focus_areas_used": focus_areas
            }
        }
    
    def _generate_multiple_choice_question(self, error_patterns: List[str], difficulty: int, focus_areas: List[str]) -> str:
        """Çoktan seçmeli soru üret"""
        if "past_tense" in error_patterns:
            return "I ____ to school yesterday."
        elif "present_perfect" in error_patterns:
            return "I ____ never been to Paris."
        else:
            return "She ____ English every day."
    
    def _generate_fill_blank_question(self, error_patterns: List[str], difficulty: int, focus_areas: List[str]) -> str:
        """Boşluk doldurma sorusu üret"""
        if "past_tense" in error_patterns:
            return "Yesterday, I _____ (go) to the market with my mother."
        elif "present_perfect" in error_patterns:
            return "I _____ (finish) my homework already."
        else:
            return "She _____ (study) English for three years."
    
    def _generate_open_ended_question(self, error_patterns: List[str], difficulty: int, focus_areas: List[str]) -> str:
        """Açık uçlu soru üret"""
        if "past_tense" in error_patterns:
            return "Write a paragraph about what you did yesterday. Use at least 5 past tense verbs."
        elif "present_perfect" in error_patterns:
            return "Describe your experiences using present perfect tense. Include at least 3 different activities."
        else:
            return "Write about your daily routine using present tense verbs."