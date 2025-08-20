import json
import random
import logging
import hashlib
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict

from app.core.config import settings
from app.services.llm_gateway import llm_gateway
from app.services.llm_utils import llm_utils

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class QuestionCache:
    """Cache entry for generated questions"""
    question_data: Dict[str, Any]
    created_at: datetime
    usage_count: int = 0
    last_used: Optional[datetime] = None

class QuestionGenerator:
    """Enhanced hybrid question generation service with orchestration, caching, and logging"""
    
    def __init__(self):
        self.templates_path = Path("app/data/question_templates")
        self.templates_data = {}
        self.daily_gpt_usage = {}
        
        # Enhanced caching system
        self.question_cache: Dict[str, QuestionCache] = {}
        self.cache_ttl_hours = 24  # Cache questions for 24 hours
        
        # Orchestration settings
        self.template_gpt_ratio = 0.7  # 70% template, 30% GPT when both available
        self.max_cache_size = 1000  # Maximum number of cached questions
        
        # Secure API key management
        self._validate_api_keys()
        self.load_templates()
        
        logger.info("QuestionGenerator initialized with enhanced orchestration and caching")
        
    def _validate_api_keys(self):
        """Validate that required API keys are available for GPT functionality"""
        if not settings.openai_api_key and not settings.anthropic_api_key:
            logger.warning("No LLM API keys configured - GPT question generation will be disabled")
            self.gpt_enabled = False
        else:
            self.gpt_enabled = True
            logger.info("LLM API keys configured - GPT question generation enabled")
        
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

    def _generate_cache_key(self, subject: str, error_type: str, difficulty_level: int, student_context: Optional[str] = None) -> str:
        """Generate a unique cache key for the question request"""
        key_data = f"{subject}:{error_type}:{difficulty_level}"
        if student_context:
            key_data += f":{student_context}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cached_question(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached question if it exists and is still valid"""
        if cache_key not in self.question_cache:
            return None
            
        cache_entry = self.question_cache[cache_key]
        now = datetime.utcnow()
        
        # Check if cache entry is expired
        if now - cache_entry.created_at > timedelta(hours=self.cache_ttl_hours):
            logger.debug(f"Cache entry expired for key: {cache_key}")
            del self.question_cache[cache_key]
            return None
            
        # Update usage statistics
        cache_entry.usage_count += 1
        cache_entry.last_used = now
        
        logger.info(f"Retrieved cached question for key: {cache_key} (usage count: {cache_entry.usage_count})")
        return cache_entry.question_data

    def _cache_question(self, cache_key: str, question_data: Dict[str, Any]):
        """Cache a generated question"""
        # Implement cache size management
        if len(self.question_cache) >= self.max_cache_size:
            # Remove oldest entries
            oldest_keys = sorted(
                self.question_cache.keys(),
                key=lambda k: self.question_cache[k].created_at
            )[:len(self.question_cache) // 4]  # Remove 25% oldest entries
            
            for key in oldest_keys:
                del self.question_cache[key]
            logger.info(f"Cleaned up {len(oldest_keys)} old cache entries")
        
        # Add new entry
        self.question_cache[cache_key] = QuestionCache(
            question_data=question_data,
            created_at=datetime.utcnow()
        )
        logger.debug(f"Cached question with key: {cache_key}")

    def _orchestrate_generation_strategy(
        self, 
        subject: str, 
        error_type: str, 
        difficulty_level: int,
        student_context: Optional[str] = None
    ) -> str:
        """
        Central orchestration function that decides generation strategy
        
        Returns:
            'template', 'gpt', 'hybrid', or 'fallback'
        """
        # Check if template exists for this error type
        template_exists = self._has_template_for_error_type(subject, error_type)
        
        if not template_exists:
            logger.info(f"No template found for {subject}/{error_type}, using GPT directly")
            return 'gpt' if self.gpt_enabled else 'fallback'
        
        if not self.gpt_enabled:
            logger.info(f"GPT disabled, using template for {subject}/{error_type}")
            return 'template'
        
        # Hybrid strategy: sometimes template, sometimes GPT
        if random.random() < self.template_gpt_ratio:
            logger.info(f"Orchestrator chose template for {subject}/{error_type}")
            return 'template'
        else:
            logger.info(f"Orchestrator chose GPT for {subject}/{error_type}")
            return 'gpt'

    def _has_template_for_error_type(self, subject: str, error_type: str) -> bool:
        """Check if a template exists for the given error type"""
        subject_templates = self.templates_data.get(subject, {})
        return error_type in subject_templates and len(subject_templates[error_type]) > 0

    async def generate_question(
        self,
        subject: str,
        error_type: str,
        difficulty_level: int = 3,
        use_gpt: bool = True,
        student_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhanced question generation with orchestration, caching, and comprehensive logging
        
        Args:
            subject: 'english' or 'math'
            error_type: Specific error type (e.g., 'present_perfect_error', 'algebra_equation')
            difficulty_level: 1-5 difficulty scale
            use_gpt: Whether to use GPT as fallback
            student_context: Student's error context for GPT
            
        Returns:
            Dict containing question, answer, and metadata
        """
        start_time = datetime.utcnow()
        cache_key = self._generate_cache_key(subject, error_type, difficulty_level, student_context)
        
        logger.info(f"Starting question generation for {subject}/{error_type} (difficulty: {difficulty_level})")
        
        try:
            # Check cache first
            cached_question = self._get_cached_question(cache_key)
            if cached_question:
                logger.info(f"Cache hit for {subject}/{error_type}")
                return cached_question
            
            # Orchestrate generation strategy
            strategy = self._orchestrate_generation_strategy(subject, error_type, difficulty_level, student_context)
            logger.info(f"Generation strategy: {strategy}")
            
            question_data = None
            
            if strategy == 'template':
                question_data = self._generate_template_question(subject, error_type, difficulty_level)
                if question_data:
                    question_data['metadata']['generation_method'] = 'template'
                    question_data['metadata']['orchestration_strategy'] = strategy
                    logger.info(f"Template generation successful for {subject}/{error_type}")
                else:
                    logger.warning(f"Template generation failed for {subject}/{error_type}, falling back to GPT")
                    strategy = 'gpt'
            
            if strategy == 'gpt' and use_gpt and self._can_use_gpt():
                try:
                    question_data = await self._generate_gpt_question(
                        subject, error_type, difficulty_level, student_context
                    )
                    if question_data:
                        question_data['metadata']['generation_method'] = 'gpt'
                        question_data['metadata']['orchestration_strategy'] = strategy
                        self._increment_gpt_usage()
                        logger.info(f"GPT generation successful for {subject}/{error_type}")
                    else:
                        logger.warning(f"GPT generation returned empty result for {subject}/{error_type}")
                except Exception as e:
                    logger.error(f"GPT generation failed for {subject}/{error_type}: {e}")
                    # Fallback to template
                    logger.info(f"Falling back to template for {subject}/{error_type}")
                    question_data = self._generate_template_question(subject, error_type, difficulty_level)
                    if question_data:
                        question_data['metadata']['generation_method'] = 'template_fallback'
                        question_data['metadata']['orchestration_strategy'] = 'fallback'
                        question_data['metadata']['fallback_reason'] = f"GPT failed: {str(e)}"
            
            # Final fallback if everything else fails
            if not question_data:
                logger.warning(f"All generation methods failed for {subject}/{error_type}, using error fallback")
                question_data = self._generate_error_fallback_question(subject, error_type)
                question_data['metadata']['generation_method'] = 'error_fallback'
                question_data['metadata']['orchestration_strategy'] = 'fallback'
            
            # Add common metadata
            question_data['metadata']['generation_time_ms'] = (datetime.utcnow() - start_time).total_seconds() * 1000
            question_data['metadata']['cache_key'] = cache_key
            question_data['metadata']['timestamp'] = datetime.utcnow().isoformat()
            
            # Cache the successful question
            if question_data.get('metadata', {}).get('generation_method') != 'error_fallback':
                self._cache_question(cache_key, question_data)
                logger.info(f"Question cached with key: {cache_key}")
            
            # Log final result
            generation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(
                f"Question generation completed for {subject}/{error_type} "
                f"in {generation_time:.2f}ms using {question_data['metadata']['generation_method']}"
            )
            
            return question_data
            
        except Exception as e:
            logger.error(f"Unexpected error in question generation for {subject}/{error_type}: {e}")
            error_question = self._generate_error_fallback_question(subject, error_type)
            error_question['metadata']['error'] = str(e)
            error_question['metadata']['generation_time_ms'] = (datetime.utcnow() - start_time).total_seconds() * 1000
            return error_question

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
                logger.debug(f"No templates found for {subject}/{error_type}")
                return None
            
            # Filter templates by difficulty level
            suitable_templates = [
                t for t in template_list 
                if t.get('difficulty_level', 3) == difficulty_level
            ]
            
            if not suitable_templates:
                # Use any template if difficulty doesn't match
                suitable_templates = template_list
                logger.debug(f"No difficulty {difficulty_level} templates for {subject}/{error_type}, using any available")
            
            # Select random template
            chosen_template = random.choice(suitable_templates)
            logger.debug(f"Selected template: {chosen_template.get('id', 'unknown')} for {subject}/{error_type}")
            
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
        """Generate question using GPT with enhanced logging and error handling"""
        try:
            logger.info(f"Starting GPT question generation for {subject}/{error_type}")
            
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
        """Check if GPT can be used based on daily limits and configuration"""
        # Check if GPT is enabled (API keys available)
        if not self.gpt_enabled:
            return False
            
        if not settings.question_generation_params.get('use_gpt', True):
            return False
        
        today = datetime.now().date()
        daily_limit = settings.question_generation_params.get('max_gpt_questions_per_day', 100)
        
        today_usage = self.daily_gpt_usage.get(today, 0)
        return today_usage < daily_limit

    def _increment_gpt_usage(self):
        """Increment daily GPT usage counter"""
        today = datetime.now().date()
        self.daily_gpt_usage[today] = self.daily_gpt_usage.get(today, 0) + 1

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about question generation including cache performance"""
        today = datetime.now().date()
        
        # Calculate cache statistics
        cache_stats = {
            "total_cached_questions": len(self.question_cache),
            "cache_hit_rate": 0.0,
            "average_cache_usage": 0.0,
            "oldest_cache_entry": None,
            "newest_cache_entry": None
        }
        
        if self.question_cache:
            usage_counts = [entry.usage_count for entry in self.question_cache.values()]
            cache_stats["average_cache_usage"] = sum(usage_counts) / len(usage_counts)
            
            timestamps = [entry.created_at for entry in self.question_cache.values()]
            cache_stats["oldest_cache_entry"] = min(timestamps).isoformat()
            cache_stats["newest_cache_entry"] = max(timestamps).isoformat()
        
        return {
            "today_gpt_usage": self.daily_gpt_usage.get(today, 0),
            "daily_gpt_limit": settings.question_generation_params.get('max_gpt_questions_per_day', 100),
            "templates_loaded": {
                subject: len(templates) for subject, templates in self.templates_data.items()
            },
            "generation_methods": ["template", "gpt", "fallback"],
            "cache_performance": cache_stats,
            "orchestration_settings": {
                "template_gpt_ratio": self.template_gpt_ratio,
                "cache_ttl_hours": self.cache_ttl_hours,
                "max_cache_size": self.max_cache_size
            }
        }
    
    def clear_expired_cache(self) -> int:
        """Clear expired cache entries and return number of cleared entries"""
        now = datetime.utcnow()
        expired_keys = []
        
        for key, entry in self.question_cache.items():
            if now - entry.created_at > timedelta(hours=self.cache_ttl_hours):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.question_cache[key]
        
        if expired_keys:
            logger.info(f"Cleared {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def clear_all_cache(self) -> int:
        """Clear all cache entries and return number of cleared entries"""
        cache_size = len(self.question_cache)
        self.question_cache.clear()
        logger.info(f"Cleared all {cache_size} cache entries")
        return cache_size

# Global instance
question_generator = QuestionGenerator()
