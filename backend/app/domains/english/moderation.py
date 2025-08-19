from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc
import re
import logging
from datetime import datetime, timedelta

from app.models.question import Question, Subject
from app.models.student_attempt import StudentAttempt
from app.models.user import User
from app.services.cache_service import cache_service

logger = logging.getLogger(__name__)


class ContentModerator:
    """Content moderation service for English questions and answers"""
    
    def __init__(self):
        # Inappropriate content patterns
        self.inappropriate_patterns = [
            r'\b(bad|inappropriate|offensive)\b',
            r'\b(violence|hate|discrimination)\b',
            r'\b(drugs|alcohol|gambling)\b',
            r'\b(sex|porn|adult)\b'
        ]
        
        # Grammar and content quality patterns
        self.grammar_patterns = {
            'missing_article': r'\b([A-Z][a-z]+)\s+([a-z]+)\b',  # Capitalized word followed by lowercase
            'incomplete_sentence': r'[^.!?]+$',  # Sentence without proper ending
            'repeated_words': r'\b(\w+)\s+\1\b',  # Repeated consecutive words
            'too_short': r'^.{1,10}$',  # Very short content
            'too_long': r'^.{500,}$'   # Very long content
        }
        
        # Difficulty level validation
        self.difficulty_indicators = {
            1: ['basic', 'simple', 'easy', 'beginner'],
            2: ['elementary', 'fundamental', 'basic'],
            3: ['intermediate', 'moderate', 'standard'],
            4: ['advanced', 'complex', 'challenging'],
            5: ['expert', 'difficult', 'sophisticated']
        }
        
        # Cache settings
        self.cache_ttl = 1800  # 30 minutes
        
    async def moderate_question(
        self,
        question_content: str,
        options: List[str],
        correct_answer: str,
        difficulty_level: int,
        topic: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Moderate English question content for appropriateness and quality"""
        
        cache_key = f"moderation:question:{hash(question_content)}:{difficulty_level}:{topic}"
        
        # Try to get from cache first
        cached_result = await cache_service.get(cache_key)
        if cached_result:
            return cached_result
        
        moderation_result = {
            "is_appropriate": True,
            "is_quality_acceptable": True,
            "moderation_score": 1.0,
            "issues": [],
            "warnings": [],
            "suggestions": [],
            "difficulty_validation": True,
            "content_analysis": {}
        }
        
        # Check for inappropriate content
        inappropriate_check = self._check_inappropriate_content(question_content)
        if not inappropriate_check["is_appropriate"]:
            moderation_result["is_appropriate"] = False
            moderation_result["issues"].extend(inappropriate_check["issues"])
            moderation_result["moderation_score"] *= 0.3
        
        # Check grammar and content quality
        quality_check = self._check_content_quality(question_content)
        if not quality_check["is_quality_acceptable"]:
            moderation_result["is_quality_acceptable"] = False
            moderation_result["issues"].extend(quality_check["issues"])
            moderation_result["moderation_score"] *= 0.7
        
        # Validate difficulty level
        difficulty_check = self._validate_difficulty_level(
            question_content, difficulty_level, topic
        )
        if not difficulty_check["is_valid"]:
            moderation_result["difficulty_validation"] = False
            moderation_result["warnings"].extend(difficulty_check["warnings"])
            moderation_result["moderation_score"] *= 0.8
        
        # Check options quality
        options_check = self._check_options_quality(options, correct_answer)
        if not options_check["is_quality_acceptable"]:
            moderation_result["is_quality_acceptable"] = False
            moderation_result["issues"].extend(options_check["issues"])
            moderation_result["moderation_score"] *= 0.6
        
        # Generate suggestions
        moderation_result["suggestions"] = self._generate_suggestions(
            question_content, options, difficulty_level, topic
        )
        
        # Content analysis
        moderation_result["content_analysis"] = {
            "word_count": len(question_content.split()),
            "sentence_count": len(re.split(r'[.!?]+', question_content)),
            "average_word_length": sum(len(word) for word in question_content.split()) / len(question_content.split()) if question_content.split() else 0,
            "complexity_score": self._calculate_complexity_score(question_content),
            "topic_relevance": self._calculate_topic_relevance(question_content, topic)
        }
        
        # Cache result
        await cache_service.set(cache_key, moderation_result, self.cache_ttl)
        
        return moderation_result
    
    async def moderate_answer(
        self,
        student_answer: str,
        question_content: str,
        correct_answer: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Moderate student answer for appropriateness and quality"""
        
        moderation_result = {
            "is_appropriate": True,
            "is_quality_acceptable": True,
            "moderation_score": 1.0,
            "issues": [],
            "warnings": [],
            "suggestions": [],
            "content_analysis": {}
        }
        
        # Check for inappropriate content
        inappropriate_check = self._check_inappropriate_content(student_answer)
        if not inappropriate_check["is_appropriate"]:
            moderation_result["is_appropriate"] = False
            moderation_result["issues"].extend(inappropriate_check["issues"])
            moderation_result["moderation_score"] *= 0.3
        
        # Check answer quality
        quality_check = self._check_answer_quality(student_answer, question_content)
        if not quality_check["is_quality_acceptable"]:
            moderation_result["is_quality_acceptable"] = False
            moderation_result["warnings"].extend(quality_check["warnings"])
            moderation_result["moderation_score"] *= 0.8
        
        # Content analysis
        moderation_result["content_analysis"] = {
            "word_count": len(student_answer.split()),
            "sentence_count": len(re.split(r'[.!?]+', student_answer)),
            "average_word_length": sum(len(word) for word in student_answer.split()) / len(student_answer.split()) if student_answer.split() else 0,
            "complexity_score": self._calculate_complexity_score(student_answer),
            "relevance_score": self._calculate_answer_relevance(student_answer, question_content)
        }
        
        return moderation_result
    
    def _check_inappropriate_content(self, content: str) -> Dict[str, Any]:
        """Check for inappropriate content patterns"""
        
        result = {
            "is_appropriate": True,
            "issues": []
        }
        
        content_lower = content.lower()
        
        for pattern in self.inappropriate_patterns:
            if re.search(pattern, content_lower):
                result["is_appropriate"] = False
                result["issues"].append(f"Inappropriate content detected: {pattern}")
        
        return result
    
    def _check_content_quality(self, content: str) -> Dict[str, Any]:
        """Check content quality and grammar"""
        
        result = {
            "is_quality_acceptable": True,
            "issues": []
        }
        
        # Check for very short content
        if re.match(self.grammar_patterns['too_short'], content):
            result["is_quality_acceptable"] = False
            result["issues"].append("Content is too short")
        
        # Check for very long content
        if re.match(self.grammar_patterns['too_long'], content):
            result["is_quality_acceptable"] = False
            result["issues"].append("Content is too long")
        
        # Check for repeated words
        repeated_words = re.findall(self.grammar_patterns['repeated_words'], content)
        if repeated_words:
            result["issues"].append(f"Repeated words detected: {repeated_words}")
        
        # Check for incomplete sentences
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            if sentence.strip() and not re.match(r'^[A-Z]', sentence.strip()):
                result["issues"].append("Incomplete or improperly formatted sentence detected")
        
        return result
    
    def _validate_difficulty_level(
        self,
        content: str,
        difficulty_level: int,
        topic: str
    ) -> Dict[str, Any]:
        """Validate if content matches the specified difficulty level"""
        
        result = {
            "is_valid": True,
            "warnings": []
        }
        
        content_lower = content.lower()
        
        # Check for difficulty indicators
        expected_indicators = self.difficulty_indicators.get(difficulty_level, [])
        found_indicators = []
        
        for indicator in expected_indicators:
            if indicator in content_lower:
                found_indicators.append(indicator)
        
        # If no difficulty indicators found, add warning
        if not found_indicators and difficulty_level > 1:
            result["warnings"].append(f"No difficulty indicators found for level {difficulty_level}")
        
        # Check word complexity
        words = content.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        if difficulty_level == 1 and avg_word_length > 6:
            result["warnings"].append("Word complexity seems higher than level 1")
        elif difficulty_level == 5 and avg_word_length < 4:
            result["warnings"].append("Word complexity seems lower than level 5")
        
        return result
    
    def _check_options_quality(
        self,
        options: List[str],
        correct_answer: str
    ) -> Dict[str, Any]:
        """Check quality of multiple choice options"""
        
        result = {
            "is_quality_acceptable": True,
            "issues": []
        }
        
        if not options or len(options) < 2:
            result["is_quality_acceptable"] = False
            result["issues"].append("Insufficient options provided")
            return result
        
        # Check if correct answer is in options
        if correct_answer not in options:
            result["is_quality_acceptable"] = False
            result["issues"].append("Correct answer not found in options")
        
        # Check for duplicate options
        if len(options) != len(set(options)):
            result["is_quality_acceptable"] = False
            result["issues"].append("Duplicate options detected")
        
        # Check option lengths
        option_lengths = [len(option) for option in options]
        if max(option_lengths) - min(option_lengths) > 50:
            result["issues"].append("Options have significantly different lengths")
        
        return result
    
    def _check_answer_quality(
        self,
        student_answer: str,
        question_content: str
    ) -> Dict[str, Any]:
        """Check quality of student answer"""
        
        result = {
            "is_quality_acceptable": True,
            "warnings": []
        }
        
        # Check if answer is too short
        if len(student_answer.strip()) < 2:
            result["warnings"].append("Answer is very short")
        
        # Check if answer is too long (for multiple choice)
        if len(student_answer) > 200:
            result["warnings"].append("Answer is very long for the question type")
        
        return result
    
    def _generate_suggestions(
        self,
        question_content: str,
        options: List[str],
        difficulty_level: int,
        topic: str
    ) -> List[str]:
        """Generate improvement suggestions"""
        
        suggestions = []
        
        # Check question length
        if len(question_content) < 20:
            suggestions.append("Consider making the question more descriptive")
        
        if len(question_content) > 300:
            suggestions.append("Consider making the question more concise")
        
        # Check options
        if len(options) < 4:
            suggestions.append("Consider adding more options for better variety")
        
        # Check topic relevance
        if topic.lower() not in question_content.lower():
            suggestions.append("Consider making the question more relevant to the topic")
        
        return suggestions
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate text complexity score"""
        
        words = content.split()
        if not words:
            return 0.0
        
        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Calculate sentence complexity
        sentences = re.split(r'[.!?]+', content)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Normalize scores
        complexity_score = (avg_word_length * 0.6 + avg_sentence_length * 0.4) / 10
        
        return min(complexity_score, 1.0)
    
    def _calculate_topic_relevance(self, content: str, topic: str) -> float:
        """Calculate topic relevance score"""
        
        content_lower = content.lower()
        topic_lower = topic.lower()
        
        # Simple keyword matching
        topic_words = topic_lower.split()
        matches = sum(1 for word in topic_words if word in content_lower)
        
        relevance_score = matches / len(topic_words) if topic_words else 0.0
        
        return relevance_score
    
    def _calculate_answer_relevance(self, answer: str, question: str) -> float:
        """Calculate answer relevance to question"""
        
        answer_lower = answer.lower()
        question_lower = question.lower()
        
        # Extract key words from question
        question_words = set(re.findall(r'\b\w+\b', question_lower))
        answer_words = set(re.findall(r'\b\w+\b', answer_lower))
        
        if not question_words:
            return 0.0
        
        # Calculate overlap
        overlap = len(question_words.intersection(answer_words))
        relevance_score = overlap / len(question_words)
        
        return relevance_score


# Global instance
content_moderator = ContentModerator()
