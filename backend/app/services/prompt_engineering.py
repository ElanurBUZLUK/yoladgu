from typing import List, Dict, Any, Optional, Union
import logging
import json
from datetime import datetime

from app.models.question import QuestionType, DifficultyLevel
from app.core.config import settings

logger = logging.getLogger(__name__)


class PromptEngineeringService:
    """Advanced prompt engineering service for question generation"""
    
    def __init__(self):
        self.cache_ttl = 3600  # 1 hour
        self.max_context_length = 2000
        self.max_examples = 3
        
        # Prompt templates for different question formats
        self.prompt_templates = {
            "mcq": self._get_mcq_prompt_template(),
            "cloze": self._get_cloze_prompt_template(),
            "error_correction": self._get_error_correction_prompt_template(),
            "fill_blank": self._get_fill_blank_prompt_template(),
            "true_false": self._get_true_false_prompt_template()
        }
        
        # CEFR-specific prompt adjustments
        self.cefr_adjustments = {
            "A1": {"complexity": "very_simple", "vocabulary": "basic", "grammar": "simple_present"},
            "A2": {"complexity": "simple", "vocabulary": "elementary", "grammar": "basic_tenses"},
            "B1": {"complexity": "intermediate", "vocabulary": "intermediate", "grammar": "mixed_tenses"},
            "B2": {"complexity": "upper_intermediate", "vocabulary": "advanced", "grammar": "complex_structures"},
            "C1": {"complexity": "advanced", "vocabulary": "sophisticated", "grammar": "advanced_structures"},
            "C2": {"complexity": "expert", "vocabulary": "academic", "grammar": "complex_advanced"}
        }
    
    async def generate_prompt(
        self,
        question_format: str,
        target_cefr: str,
        topic: str,
        error_focus: List[str],
        context: str,
        user_level: int,
        learning_style: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive prompt for question generation"""
        
        try:
            # Get base template
            base_template = self.prompt_templates.get(question_format, self.prompt_templates["mcq"])
            
            # Apply CEFR-specific adjustments
            cefr_adjustment = self.cefr_adjustments.get(target_cefr, self.cefr_adjustments["B1"])
            
            # Build context section
            context_section = self._build_context_section(context, error_focus, topic)
            
            # Build examples section
            examples_section = self._build_examples_section(question_format, target_cefr, topic)
            
            # Build constraints section
            constraints_section = self._build_constraints_section(constraints, user_level, learning_style)
            
            # Build error focus section
            error_focus_section = self._build_error_focus_section(error_focus, target_cefr)
            
            # Assemble final prompt
            final_prompt = base_template.format(
                cefr_level=target_cefr,
                complexity=cefr_adjustment["complexity"],
                vocabulary=cefr_adjustment["vocabulary"],
                grammar=cefr_adjustment["grammar"],
                topic=topic,
                context=context_section,
                examples=examples_section,
                constraints=constraints_section,
                error_focus=error_focus_section
            )
            
            # Generate metadata
            metadata = {
                "format": question_format,
                "cefr_level": target_cefr,
                "topic": topic,
                "error_focus": error_focus,
                "user_level": user_level,
                "learning_style": learning_style,
                "constraints": constraints,
                "context_length": len(context),
                "prompt_length": len(final_prompt),
                "generated_at": datetime.now().isoformat()
            }
            
            return {
                "prompt": final_prompt,
                "metadata": metadata,
                "template_used": question_format,
                "cefr_adjustment": cefr_adjustment
            }
            
        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            return self._get_fallback_prompt(question_format, target_cefr, topic)
    
    def _get_mcq_prompt_template(self) -> str:
        """Get MCQ prompt template"""
        
        return """You are an expert English language educator creating {complexity} multiple-choice questions for CEFR level {cefr_level}.

**Target Level**: {cefr_level} ({vocabulary} vocabulary, {grammar} grammar structures)

**Topic**: {topic}

**Context Information**:
{context}

**Error Focus Areas**:
{error_focus}

**Question Requirements**:
- Create a single multiple-choice question with 4 options (A, B, C, D)
- Use {complexity} language appropriate for {cefr_level} level
- Focus on the specified topic and error patterns
- Ensure only one correct answer
- Make distractors plausible but clearly incorrect
- Include a brief explanation for the correct answer

**Example Format**:
{examples}

**Additional Constraints**:
{constraints}

**Instructions**:
1. Analyze the context and error focus areas
2. Create a question that tests understanding of the topic
3. Ensure the question is appropriate for {cefr_level} level
4. Provide clear, educational explanations
5. Follow the exact JSON format specified

Please generate the question in the following JSON format:
{{
    "content": "Question text here?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "Option A",
    "explanation": "Explanation of why this is correct",
    "difficulty_level": 3,
    "topic": "{topic}",
    "cefr_level": "{cefr_level}",
    "error_patterns_addressed": ["pattern1", "pattern2"]
}}"""

    def _get_cloze_prompt_template(self) -> str:
        """Get cloze test prompt template"""
        
        return """You are an expert English language educator creating {complexity} cloze test questions for CEFR level {cefr_level}.

**Target Level**: {cefr_level} ({vocabulary} vocabulary, {grammar} grammar structures)

**Topic**: {topic}

**Context Information**:
{context}

**Error Focus Areas**:
{error_focus}

**Question Requirements**:
- Create a cloze test with 3-5 blanks
- Use {complexity} language appropriate for {cefr_level} level
- Focus on the specified topic and error patterns
- Provide 3-4 options for each blank
- Ensure clear context for each blank
- Include explanations for correct answers

**Example Format**:
{examples}

**Additional Constraints**:
{constraints}

**Instructions**:
1. Create a coherent passage with strategic blanks
2. Focus on grammar, vocabulary, or comprehension
3. Ensure appropriate difficulty for {cefr_level} level
4. Provide educational explanations

Please generate the cloze test in the following JSON format:
{{
    "content": "Passage with ___1___ and ___2___ blanks",
    "blanks": [
        {{
            "position": 1,
            "correct_answer": "correct_word",
            "options": ["correct_word", "wrong1", "wrong2", "wrong3"]
        }},
        {{
            "position": 2,
            "correct_answer": "correct_word",
            "options": ["correct_word", "wrong1", "wrong2", "wrong3"]
        }}
    ],
    "explanation": "Explanation of the cloze test",
    "difficulty_level": 3,
    "topic": "{topic}",
    "cefr_level": "{cefr_level}",
    "error_patterns_addressed": ["pattern1", "pattern2"]
}}"""

    def _get_error_correction_prompt_template(self) -> str:
        """Get error correction prompt template"""
        
        return """You are an expert English language educator creating {complexity} error correction questions for CEFR level {cefr_level}.

**Target Level**: {cefr_level} ({vocabulary} vocabulary, {grammar} grammar structures)

**Topic**: {topic}

**Context Information**:
{context}

**Error Focus Areas**:
{error_focus}

**Question Requirements**:
- Create sentences with intentional errors
- Use {complexity} language appropriate for {cefr_level} level
- Focus on the specified error patterns
- Provide clear corrections and explanations
- Ensure errors are realistic and educational

**Example Format**:
{examples}

**Additional Constraints**:
{constraints}

**Instructions**:
1. Create sentences with common errors from the focus areas
2. Ensure errors are appropriate for {cefr_level} level
3. Provide clear corrections and explanations
4. Focus on educational value

Please generate the error correction question in the following JSON format:
{{
    "content": "Sentence with error: I goes to school every day.",
    "corrected_content": "I go to school every day.",
    "error_type": "verb_tense",
    "explanation": "The verb 'go' should be in present simple form for first person singular",
    "difficulty_level": 3,
    "topic": "{topic}",
    "cefr_level": "{cefr_level}",
    "error_patterns_addressed": ["verb_tense_errors"]
}}"""

    def _get_fill_blank_prompt_template(self) -> str:
        """Get fill in the blank prompt template"""
        
        return """You are an expert English language educator creating {complexity} fill-in-the-blank questions for CEFR level {cefr_level}.

**Target Level**: {cefr_level} ({vocabulary} vocabulary, {grammar} grammar structures)

**Topic**: {topic}

**Context Information**:
{context}

**Error Focus Areas**:
{error_focus}

**Question Requirements**:
- Create sentences with blanks for vocabulary or grammar
- Use {complexity} language appropriate for {cefr_level} level
- Focus on the specified topic and error patterns
- Provide context clues for answers
- Include explanations

**Example Format**:
{examples}

**Additional Constraints**:
{constraints}

**Instructions**:
1. Create meaningful sentences with strategic blanks
2. Focus on vocabulary or grammar from the topic
3. Ensure appropriate difficulty for {cefr_level} level
4. Provide educational explanations

Please generate the fill-in-the-blank question in the following JSON format:
{{
    "content": "The weather is ___ today.",
    "correct_answer": "sunny",
    "hint": "Think about weather conditions",
    "explanation": "The word 'sunny' describes clear weather with sunshine",
    "difficulty_level": 3,
    "topic": "{topic}",
    "cefr_level": "{cefr_level}",
    "error_patterns_addressed": ["vocabulary_errors"]
}}"""

    def _get_true_false_prompt_template(self) -> str:
        """Get true/false prompt template"""
        
        return """You are an expert English language educator creating {complexity} true/false questions for CEFR level {cefr_level}.

**Target Level**: {cefr_level} ({vocabulary} vocabulary, {grammar} grammar structures)

**Topic**: {topic}

**Context Information**:
{context}

**Error Focus Areas**:
{error_focus}

**Question Requirements**:
- Create statements that are clearly true or false
- Use {complexity} language appropriate for {cefr_level} level
- Focus on the specified topic and error patterns
- Ensure statements are unambiguous
- Include explanations for both true and false answers

**Example Format**:
{examples}

**Additional Constraints**:
{constraints}

**Instructions**:
1. Create clear, unambiguous statements
2. Focus on factual information from the topic
3. Ensure appropriate difficulty for {cefr_level} level
4. Provide educational explanations

Please generate the true/false question in the following JSON format:
{{
    "content": "The Earth is the largest planet in our solar system.",
    "correct_answer": "False",
    "explanation": "Jupiter is the largest planet in our solar system, not Earth",
    "difficulty_level": 3,
    "topic": "{topic}",
    "cefr_level": "{cefr_level}",
    "error_patterns_addressed": ["factual_errors"]
}}"""

    def _build_context_section(self, context: str, error_focus: List[str], topic: str) -> str:
        """Build context section for prompt"""
        
        if not context:
            return f"Focus on the topic: {topic}"
        
        # Truncate context if too long
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "..."
        
        return f"""Based on the following context:
{context}

Topic focus: {topic}"""

    def _build_examples_section(self, question_format: str, target_cefr: str, topic: str) -> str:
        """Build examples section for prompt"""
        
        examples = {
            "mcq": f"""Question: What is the correct form of the verb in this sentence?
"She ___ to school every day."
A) go
B) goes  
C) going
D) gone

Correct Answer: B) goes
Explanation: For third person singular (she), we use 'goes' in present simple tense.""",
            
            "cloze": f"""Passage: The weather ___1___ beautiful today. I ___2___ to go for a walk.

Blanks:
1. A) is B) are C) was D) were
2. A) want B) wants C) wanting D) wanted

Answers: 1-A, 2-A
Explanation: "Weather" is singular, so we use "is". "I" takes "want" in present simple.""",
            
            "error_correction": f"""Sentence: I goes to school every day.
Correction: I go to school every day.
Explanation: For first person singular (I), we use 'go' in present simple tense, not 'goes'.""",
            
            "fill_blank": f"""Sentence: The weather is ___ today.
Answer: sunny
Explanation: "Sunny" describes clear weather with sunshine.""",
            
            "true_false": f"""Statement: The Earth is the largest planet in our solar system.
Answer: False
Explanation: Jupiter is the largest planet in our solar system, not Earth."""
        }
        
        return examples.get(question_format, examples["mcq"])

    def _build_constraints_section(self, constraints: Optional[Dict[str, Any]], user_level: int, learning_style: Optional[str]) -> str:
        """Build constraints section for prompt"""
        
        constraints_text = []
        
        if constraints:
            if constraints.get("avoid_topics"):
                constraints_text.append(f"Avoid topics: {', '.join(constraints['avoid_topics'])}")
            
            if constraints.get("prefer_topics"):
                constraints_text.append(f"Prefer topics: {', '.join(constraints['prefer_topics'])}")
            
            if constraints.get("max_length"):
                constraints_text.append(f"Maximum question length: {constraints['max_length']} words")
        
        if learning_style:
            constraints_text.append(f"Learning style: {learning_style}")
        
        constraints_text.append(f"User level: {user_level}/5")
        
        return "\n".join(constraints_text) if constraints_text else "No specific constraints"

    def _build_error_focus_section(self, error_focus: List[str], target_cefr: str) -> str:
        """Build error focus section for prompt"""
        
        if not error_focus:
            return "Focus on general language skills appropriate for the CEFR level."
        
        error_text = "Focus on these specific error patterns:\n"
        for i, error in enumerate(error_focus, 1):
            error_text += f"{i}. {error}\n"
        
        error_text += f"\nEnsure the question addresses these patterns while being appropriate for {target_cefr} level."
        
        return error_text

    def _get_fallback_prompt(self, question_format: str, target_cefr: str, topic: str) -> Dict[str, Any]:
        """Get fallback prompt when generation fails"""
        
        fallback_prompt = f"""Create a {question_format} question about {topic} for CEFR level {target_cefr}.

Requirements:
- Use appropriate language for {target_cefr} level
- Focus on the topic: {topic}
- Create an educational and engaging question
- Provide clear explanations

Please generate the question in JSON format."""

        return {
            "prompt": fallback_prompt,
            "metadata": {
                "format": question_format,
                "cefr_level": target_cefr,
                "topic": topic,
                "error_focus": [],
                "user_level": 3,
                "learning_style": None,
                "constraints": None,
                "context_length": 0,
                "prompt_length": len(fallback_prompt),
                "generated_at": datetime.now().isoformat(),
                "is_fallback": True
            },
            "template_used": "fallback",
            "cefr_adjustment": self.cefr_adjustments.get(target_cefr, self.cefr_adjustments["B1"])
        }

    async def get_prompt_statistics(self) -> Dict[str, Any]:
        """Get statistics about prompt engineering"""
        
        return {
            "available_formats": list(self.prompt_templates.keys()),
            "cefr_levels": list(self.cefr_adjustments.keys()),
            "max_context_length": self.max_context_length,
            "max_examples": self.max_examples,
            "cache_ttl": self.cache_ttl
        }

    async def validate_prompt(self, prompt: str, target_format: str, target_cefr: str) -> Dict[str, Any]:
        """Validate generated prompt"""
        
        validation_result = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "suggestions": []
        }
        
        # Check prompt length
        if len(prompt) < 100:
            validation_result["is_valid"] = False
            validation_result["issues"].append("Prompt too short")
        
        if len(prompt) > 5000:
            validation_result["warnings"].append("Prompt very long, may affect performance")
        
        # Check for required elements
        required_elements = ["format", "cefr_level", "topic", "context", "examples"]
        for element in required_elements:
            if element not in prompt.lower():
                validation_result["warnings"].append(f"Missing {element} in prompt")
        
        # Check format-specific requirements
        if target_format == "mcq" and "options" not in prompt.lower():
            validation_result["issues"].append("MCQ prompt missing options requirement")
        
        if target_format == "cloze" and "blanks" not in prompt.lower():
            validation_result["issues"].append("Cloze prompt missing blanks requirement")
        
        # Check CEFR level appropriateness
        if target_cefr not in self.cefr_adjustments:
            validation_result["warnings"].append(f"Unknown CEFR level: {target_cefr}")
        
        validation_result["is_valid"] = len(validation_result["issues"]) == 0
        
        return validation_result


# Global instance
prompt_engineering_service = PromptEngineeringService()
