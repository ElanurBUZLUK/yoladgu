import json
import random
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path

from app.core.config import settings
from app.services.llm_gateway import llm_gateway
from app.services.llm_utils import llm_utils

logger = logging.getLogger(__name__)

class QuestionGenerator:
    """Hybrid question generation service using templates and GPT"""
    
    def __init__(self):
        self.templates_path = Path("app/data/question_templates")
        self.templates_data = {}
        self.daily_gpt_usage = {}
        self.load_templates()
        
    def load_templates(self):
        """Load question templates from JSON files"""
        try:
            # Load English grammar templates
            english_path = self.templates_path / "english_grammar_templates.json"
            if english_path.exists():
                with open(english_path, 'r', encoding='utf-8') as f:
                    self.templates_data['english'] = json.load(f)
                logger.info(f"Loaded {len(self.templates_data['english'])} English grammar templates")
            
            # Load Math problem templates
            math_path = self.templates_path / "math_problem_templates.json"
            if math_path.exists():
                with open(math_path, 'r', encoding='utf-8') as f:
                    self.templates_data['math'] = json.load(f)
                logger.info(f"Loaded {len(self.templates_data['math'])} Math problem templates")
                
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            self.templates_data = {'english': {}, 'math': {}}

    async def generate_question(
        self,
        subject: str,
        error_type: str,
        difficulty_level: int = 3,
        use_gpt: bool = True,
        student_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a question using hybrid approach (templates + GPT)
        
        Args:
            subject: 'english' or 'math'
            error_type: Specific error type (e.g., 'present_perfect_error', 'algebra_equation')
            difficulty_level: 1-5 difficulty scale
            use_gpt: Whether to use GPT as fallback
            student_context: Student's error context for GPT
            
        Returns:
            Dict containing question, answer, and metadata
        """
        try:
            # First, try template-based generation
            template_question = self._generate_template_question(subject, error_type, difficulty_level)
            
            if template_question:
                logger.info(f"Generated template question for {subject}/{error_type}")
                return template_question
            
            # If template fails and GPT is enabled, try GPT
            if use_gpt and self._can_use_gpt():
                gpt_question = await self._generate_gpt_question(
                    subject, error_type, difficulty_level, student_context
                )
                
                if gpt_question:
                    self._increment_gpt_usage()
                    logger.info(f"Generated GPT question for {subject}/{error_type}")
                    return gpt_question
            
            # Fallback to basic template
            fallback_question = self._generate_fallback_question(subject, error_type, difficulty_level)
            logger.warning(f"Using fallback question for {subject}/{error_type}")
            return fallback_question
            
        except Exception as e:
            logger.error(f"Error generating question: {e}")
            return self._generate_error_fallback_question(subject, error_type)

    def _generate_template_question(
        self, 
        subject: str, 
        error_type: str, 
        difficulty_level: int
    ) -> Optional[Dict[str, Any]]:
        """Generate question using predefined templates"""
        try:
            subject_templates = self.templates_data.get(subject, {})
            template_list = subject_templates.get(error_type, [])
            
            if not template_list:
                return None
            
            # Filter templates by difficulty level
            suitable_templates = [
                t for t in template_list 
                if t.get('difficulty_level', 3) == difficulty_level
            ]
            
            if not suitable_templates:
                # Use any template if difficulty doesn't match
                suitable_templates = template_list
            
            # Select random template
            chosen_template = random.choice(suitable_templates)
            
            # Process template
            question_text = chosen_template["template"]
            answer = chosen_template.get("answer", "")
            
            # Handle word sets for variety
            if "word_sets" in chosen_template:
                word_set = random.choice(chosen_template["word_sets"])
                for key, value in word_set.items():
                    placeholder = "{" + key + "}"
                    question_text = question_text.replace(placeholder, str(value))
                
                # Extract answer from word set if available
                if not answer and "correct_answer" in word_set:
                    answer = word_set["correct_answer"]
            
            # Generate options if not provided
            options = chosen_template.get("options", [])
            if not options and chosen_template.get("generate_options", False):
                options = self._generate_options_for_question(question_text, answer, subject)
            
            return {
                "question": question_text,
                "answer": answer,
                "options": options,
                "type": "template",
                "difficulty_level": difficulty_level,
                "error_type": error_type,
                "subject": subject,
                "metadata": {
                    "template_id": chosen_template.get("id", ""),
                    "generation_method": "template",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error in template generation: {e}")
            return None

    async def _generate_gpt_question(
        self,
        subject: str,
        error_type: str,
        difficulty_level: int,
        student_context: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Generate question using GPT"""
        try:
            # Create prompt for GPT
            prompt = self._create_gpt_prompt(subject, error_type, difficulty_level, student_context)
            
            # Define response schema
            schema = {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The generated question"},
                    "answer": {"type": "string", "description": "Correct answer"},
                    "options": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "description": "Multiple choice options (if applicable)"
                    },
                    "explanation": {"type": "string", "description": "Brief explanation of the answer"},
                    "difficulty_reasoning": {"type": "string", "description": "Why this question matches the difficulty level"}
                },
                "required": ["question", "answer"]
            }
            
            # Generate using LLM utils with repair
            response = await llm_utils.generate_json_with_repair(
                prompt=prompt,
                schema=schema,
                system_prompt="You are an expert educational content creator. Generate clear, accurate practice questions.",
                retries=2
            )
            
            if response.get("success") and response.get("parsed_json"):
                gpt_data = response["parsed_json"]
                
                return {
                    "question": gpt_data["question"],
                    "answer": gpt_data["answer"],
                    "options": gpt_data.get("options", []),
                    "explanation": gpt_data.get("explanation", ""),
                    "type": "gpt",
                    "difficulty_level": difficulty_level,
                    "error_type": error_type,
                    "subject": subject,
                    "metadata": {
                        "generation_method": "gpt",
                        "gpt_model": "gpt-3.5-turbo",
                        "timestamp": datetime.utcnow().isoformat(),
                        "student_context": student_context
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in GPT generation: {e}")
            return None

    def _create_gpt_prompt(
        self,
        subject: str,
        error_type: str,
        difficulty_level: int,
        student_context: Optional[str]
    ) -> str:
        """Create GPT prompt for question generation"""
        
        if subject == "english":
            prompt = f"""
            Generate a practice question for an English learner who struggles with {error_type}.
            
            Difficulty Level: {difficulty_level}/5
            - Level 1-2: Basic concepts, simple sentences
            - Level 3: Intermediate, compound sentences
            - Level 4-5: Advanced, complex structures
            
            Student Context: {student_context or 'General practice needed'}
            
            Requirements:
            1. Question should be clear and focused on the specific grammar rule
            2. Include 4 multiple choice options (A, B, C, D)
            3. Provide a brief explanation of why the answer is correct
            4. Match the specified difficulty level
            
            Format your response as JSON with: question, answer, options, explanation, difficulty_reasoning
            """
        
        elif subject == "math":
            prompt = f"""
            Generate a math practice problem for a student who struggles with {error_type}.
            
            Difficulty Level: {difficulty_level}/5
            - Level 1-2: Basic operations, simple word problems
            - Level 3: Intermediate, multi-step problems
            - Level 4-5: Advanced, complex problem-solving
            
            Student Context: {student_context or 'General practice needed'}
            
            Requirements:
            1. Problem should be clear and focused on the specific math concept
            2. Include step-by-step solution approach
            3. Provide a brief explanation of the key concepts
            4. Match the specified difficulty level
            
            Format your response as JSON with: question, answer, explanation, difficulty_reasoning
            """
        
        else:
            prompt = f"""
            Generate a practice question for {subject} focusing on {error_type}.
            Difficulty Level: {difficulty_level}/5
            Student Context: {student_context or 'General practice needed'}
            
            Format your response as JSON with: question, answer, explanation
            """
        
        return prompt

    def _generate_fallback_question(
        self,
        subject: str,
        error_type: str,
        difficulty_level: int
    ) -> Dict[str, Any]:
        """Generate a basic fallback question when templates and GPT fail"""
        
        if subject == "english":
            fallback_questions = {
                "present_perfect_error": {
                    "question": "Complete: I _____ never _____ (see) a ghost.",
                    "answer": "have, seen",
                    "explanation": "Use present perfect for experiences up to now"
                },
                "preposition_error": {
                    "question": "Choose: He is afraid _____ spiders. A) of B) from C) with",
                    "answer": "A",
                    "explanation": "'Afraid of' is the correct preposition"
                }
            }
        elif subject == "math":
            fallback_questions = {
                "algebra_equation": {
                    "question": "Solve: 2x + 5 = 13",
                    "answer": "x = 4",
                    "explanation": "Subtract 5 from both sides, then divide by 2"
                },
                "fraction_operation": {
                    "question": "Calculate: 1/2 + 1/3",
                    "answer": "5/6",
                    "explanation": "Find common denominator (6), then add numerators"
                }
            }
        else:
            fallback_questions = {
                "general": {
                    "question": "This is a fallback question. Please contact support.",
                    "answer": "N/A",
                    "explanation": "Template generation failed"
                }
            }
        
        # Get fallback question or use general one
        fallback = fallback_questions.get(error_type, fallback_questions.get("general", {}))
        
        return {
            "question": fallback.get("question", "Question generation failed"),
            "answer": fallback.get("answer", "N/A"),
            "explanation": fallback.get("explanation", ""),
            "type": "fallback",
            "difficulty_level": difficulty_level,
            "error_type": error_type,
            "subject": subject,
            "metadata": {
                "generation_method": "fallback",
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    def _generate_error_fallback_question(
        self,
        subject: str,
        error_type: str
    ) -> Dict[str, Any]:
        """Generate question when all methods fail"""
        return {
            "question": f"Practice question for {error_type} in {subject}",
            "answer": "Please try again later",
            "type": "error_fallback",
            "error_type": error_type,
            "subject": subject,
            "metadata": {
                "generation_method": "error_fallback",
                "error": "All generation methods failed",
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    def _generate_options_for_question(
        self,
        question_text: str,
        correct_answer: str,
        subject: str
    ) -> List[str]:
        """Generate multiple choice options for a question"""
        # This is a simple implementation - could be enhanced with more logic
        if subject == "english":
            # For English, create plausible distractors
            distractors = ["Option A", "Option B", "Option C"]
            options = [correct_answer] + distractors
            random.shuffle(options)
            return options
        else:
            # For math, generate numerical distractors
            try:
                correct_num = float(correct_answer)
                distractors = [correct_num + 1, correct_num - 1, correct_num * 2]
                options = [correct_answer] + [str(d) for d in distractors]
                random.shuffle(options)
                return options
            except:
                return [correct_answer, "Option A", "Option B", "Option C"]

    def _can_use_gpt(self) -> bool:
        """Check if GPT can be used based on daily limits"""
        today = datetime.now().date()
        daily_limit = settings.question_generation_params.get('max_gpt_questions_per_day', 100)
        
        today_usage = self.daily_gpt_usage.get(today, 0)
        return today_usage < daily_limit

    def _increment_gpt_usage(self):
        """Increment daily GPT usage counter"""
        today = datetime.now().date()
        self.daily_gpt_usage[today] = self.daily_gpt_usage.get(today, 0) + 1

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about question generation"""
        today = datetime.now().date()
        return {
            "today_gpt_usage": self.daily_gpt_usage.get(today, 0),
            "daily_gpt_limit": settings.question_generation_params.get('max_gpt_questions_per_day', 100),
            "templates_loaded": {
                subject: len(templates) for subject, templates in self.templates_data.items()
            },
            "generation_methods": ["template", "gpt", "fallback"]
        }

# Global instance
question_generator = QuestionGenerator()
