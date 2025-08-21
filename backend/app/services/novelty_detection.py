from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import json
import hashlib
import re
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import asyncio

from app.core.cache import cache_service
from app.services.embedding_service import embedding_service
from app.database import database_manager

logger = logging.getLogger(__name__)


class NoveltyDetectionService:
    """Comprehensive novelty detection and content diversity analysis service"""
    
    def __init__(self):
        self.cache_ttl = 1800  # 30 minutes
        self.similarity_threshold = 0.85
        self.diversity_threshold = 0.7
        self.quality_threshold = 0.6
        
        # Content fingerprinting settings
        self.fingerprint_length = 64
        self.min_content_length = 10
        
        # Diversity metrics
        self.topic_diversity_weight = 0.3
        self.format_diversity_weight = 0.2
        self.difficulty_diversity_weight = 0.2
        self.error_pattern_diversity_weight = 0.3
    
    async def detect_novelty(
        self,
        content: str,
        question_type: str,
        topic: str,
        difficulty_level: int,
        error_patterns: List[str],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Comprehensive novelty detection for new content"""
        
        try:
            # Generate content fingerprint
            fingerprint = self._generate_content_fingerprint(content)
            
            # Check for exact duplicates
            exact_duplicates = await self._check_exact_duplicates(fingerprint)
            
            # Check for semantic duplicates
            semantic_duplicates = await self._check_semantic_duplicates(content, question_type, topic)
            
            # Analyze content diversity
            diversity_analysis = await self._analyze_content_diversity(
                content, question_type, topic, difficulty_level, error_patterns
            )
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(content, context)
            
            # Determine novelty score
            novelty_score = self._calculate_novelty_score(
                exact_duplicates, semantic_duplicates, diversity_analysis, quality_metrics
            )
            
            # Generate recommendations
            recommendations = self._generate_novelty_recommendations(
                exact_duplicates, semantic_duplicates, diversity_analysis, quality_metrics
            )
            
            return {
                "is_novel": novelty_score >= self.quality_threshold,
                "novelty_score": novelty_score,
                "exact_duplicates": exact_duplicates,
                "semantic_duplicates": semantic_duplicates,
                "diversity_analysis": diversity_analysis,
                "quality_metrics": quality_metrics,
                "recommendations": recommendations,
                "fingerprint": fingerprint,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in novelty detection: {e}")
            return {
                "is_novel": True,  # Default to novel if detection fails
                "novelty_score": 0.8,
                "exact_duplicates": [],
                "semantic_duplicates": [],
                "diversity_analysis": {},
                "quality_metrics": {},
                "recommendations": ["Novelty detection failed, manual review recommended"],
                "fingerprint": "",
                "analysis_timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _generate_content_fingerprint(self, content: str) -> str:
        """Generate content fingerprint for duplicate detection"""
        
        # Normalize content
        normalized = self._normalize_content(content)
        
        # Generate hash
        content_hash = hashlib.sha256(normalized.encode()).hexdigest()
        
        # Truncate to desired length
        return content_hash[:self.fingerprint_length]
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for fingerprinting"""
        
        # Convert to lowercase
        normalized = content.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove punctuation (keep basic structure)
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        words = normalized.split()
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        return ' '.join(filtered_words)
    
    async def _check_exact_duplicates(self, fingerprint: str) -> List[Dict[str, Any]]:
        """Check for exact duplicates using content fingerprint"""
        
        try:
            # Query database for exact fingerprint matches
            query = """
                SELECT id, content, topic, difficulty_level, question_type, created_at
                FROM questions
                WHERE content_fingerprint = %s
                AND is_active = true
                ORDER BY created_at DESC
                LIMIT 10
            """
            
            result = await database.fetch_all(query, [fingerprint])
            
            duplicates = []
            for row in result:
                duplicates.append({
                    "id": str(row["id"]),
                    "content": row["content"],
                    "topic": row["topic"],
                    "difficulty_level": row["difficulty_level"],
                    "question_type": row["question_type"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "similarity": 1.0,
                    "match_type": "exact"
                })
            
            return duplicates
            
        except Exception as e:
            logger.error(f"Error checking exact duplicates: {e}")
            return []
    
    async def _check_semantic_duplicates(
        self,
        content: str,
        question_type: str,
        topic: str
    ) -> List[Dict[str, Any]]:
        """Check for semantic duplicates using vector similarity"""
        
        try:
            # Generate embedding for the content
            content_embedding = await embedding_service.get_embedding(content)
            
            # Query for similar questions
            similar_questions = await self._find_similar_questions(
                content_embedding, question_type, topic
            )
            
            # Filter by similarity threshold
            semantic_duplicates = []
            for question in similar_questions:
                if question["similarity"] >= self.similarity_threshold:
                    question["match_type"] = "semantic"
                    semantic_duplicates.append(question)
            
            return semantic_duplicates
            
        except Exception as e:
            logger.error(f"Error checking semantic duplicates: {e}")
            return []
    
    async def _find_similar_questions(
        self,
        content_embedding: List[float],
        question_type: str,
        topic: str
    ) -> List[Dict[str, Any]]:
        """Find similar questions using vector similarity"""
        
        try:
            # Convert embedding to PostgreSQL array format
            embedding_array = f"[{','.join(map(str, content_embedding))}]"
            
            # Build similarity query
            query = f"""
                SELECT 
                    id, content, topic, difficulty_level, question_type, created_at,
                    content_embedding <-> '{embedding_array}'::vector as distance,
                    1 - (content_embedding <=> '{embedding_array}'::vector) as similarity
                FROM questions
                WHERE is_active = true
                AND question_type = %s
                AND topic ILIKE %s
                AND content_embedding IS NOT NULL
                ORDER BY distance ASC
                LIMIT 20
            """
            
            result = await database.fetch_all(query, [question_type, f"%{topic}%"])
            
            similar_questions = []
            for row in result:
                similar_questions.append({
                    "id": str(row["id"]),
                    "content": row["content"],
                    "topic": row["topic"],
                    "difficulty_level": row["difficulty_level"],
                    "question_type": row["question_type"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "similarity": float(row["similarity"]),
                    "distance": float(row["distance"])
                })
            
            return similar_questions
            
        except Exception as e:
            logger.error(f"Error finding similar questions: {e}")
            return []
    
    async def _analyze_content_diversity(
        self,
        content: str,
        question_type: str,
        topic: str,
        difficulty_level: int,
        error_patterns: List[str]
    ) -> Dict[str, Any]:
        """Analyze content diversity across multiple dimensions"""
        
        try:
            # Get recent questions for comparison
            recent_questions = await self._get_recent_questions(
                question_type, topic, days=30
            )
            
            # Analyze topic diversity
            topic_diversity = self._analyze_topic_diversity(recent_questions, topic)
            
            # Analyze format diversity
            format_diversity = self._analyze_format_diversity(recent_questions, question_type)
            
            # Analyze difficulty diversity
            difficulty_diversity = self._analyze_difficulty_diversity(recent_questions, difficulty_level)
            
            # Analyze error pattern diversity
            error_pattern_diversity = self._analyze_error_pattern_diversity(recent_questions, error_patterns)
            
            # Calculate overall diversity score
            overall_diversity = (
                topic_diversity["score"] * self.topic_diversity_weight +
                format_diversity["score"] * self.format_diversity_weight +
                difficulty_diversity["score"] * self.difficulty_diversity_weight +
                error_pattern_diversity["score"] * self.error_pattern_diversity_weight
            )
            
            return {
                "overall_diversity_score": overall_diversity,
                "topic_diversity": topic_diversity,
                "format_diversity": format_diversity,
                "difficulty_diversity": difficulty_diversity,
                "error_pattern_diversity": error_pattern_diversity,
                "is_diverse": overall_diversity >= self.diversity_threshold
            }
            
        except Exception as e:
            logger.error(f"Error analyzing content diversity: {e}")
            return {
                "overall_diversity_score": 0.5,
                "topic_diversity": {"score": 0.5, "details": {}},
                "format_diversity": {"score": 0.5, "details": {}},
                "difficulty_diversity": {"score": 0.5, "details": {}},
                "error_pattern_diversity": {"score": 0.5, "details": {}},
                "is_diverse": True
            }
    
    async def _get_recent_questions(
        self,
        question_type: str,
        topic: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get recent questions for diversity analysis"""
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            query = """
                SELECT id, content, topic, difficulty_level, question_type, 
                       error_patterns, created_at
                FROM questions
                WHERE is_active = true
                AND question_type = %s
                AND topic ILIKE %s
                AND created_at >= %s
                ORDER BY created_at DESC
                LIMIT 100
            """
            
            result = await database.fetch_all(query, [question_type, f"%{topic}%", cutoff_date])
            
            questions = []
            for row in result:
                questions.append({
                    "id": str(row["id"]),
                    "content": row["content"],
                    "topic": row["topic"],
                    "difficulty_level": row["difficulty_level"],
                    "question_type": row["question_type"],
                    "error_patterns": row["error_patterns"] or [],
                    "created_at": row["created_at"]
                })
            
            return questions
            
        except Exception as e:
            logger.error(f"Error getting recent questions: {e}")
            return []
    
    def _analyze_topic_diversity(
        self,
        recent_questions: List[Dict[str, Any]],
        current_topic: str
    ) -> Dict[str, Any]:
        """Analyze topic diversity"""
        
        if not recent_questions:
            return {"score": 1.0, "details": {"topic_count": 1, "current_topic_frequency": 0}}
        
        # Count topics
        topic_counts = Counter(q["topic"] for q in recent_questions)
        total_questions = len(recent_questions)
        
        # Calculate topic diversity
        topic_count = len(topic_counts)
        current_topic_frequency = topic_counts.get(current_topic, 0) / total_questions
        
        # Diversity score: higher for more topics, lower for overused current topic
        diversity_score = min(1.0, topic_count / 5) * (1.0 - current_topic_frequency)
        
        return {
            "score": diversity_score,
            "details": {
                "topic_count": topic_count,
                "current_topic_frequency": current_topic_frequency,
                "topic_distribution": dict(topic_counts)
            }
        }
    
    def _analyze_format_diversity(
        self,
        recent_questions: List[Dict[str, Any]],
        current_format: str
    ) -> Dict[str, Any]:
        """Analyze question format diversity"""
        
        if not recent_questions:
            return {"score": 1.0, "details": {"format_count": 1, "current_format_frequency": 0}}
        
        # Count formats
        format_counts = Counter(q["question_type"] for q in recent_questions)
        total_questions = len(recent_questions)
        
        # Calculate format diversity
        format_count = len(format_counts)
        current_format_frequency = format_counts.get(current_format, 0) / total_questions
        
        # Diversity score: higher for more formats, lower for overused current format
        diversity_score = min(1.0, format_count / 3) * (1.0 - current_format_frequency)
        
        return {
            "score": diversity_score,
            "details": {
                "format_count": format_count,
                "current_format_frequency": current_format_frequency,
                "format_distribution": dict(format_counts)
            }
        }
    
    def _analyze_difficulty_diversity(
        self,
        recent_questions: List[Dict[str, Any]],
        current_difficulty: int
    ) -> Dict[str, Any]:
        """Analyze difficulty level diversity"""
        
        if not recent_questions:
            return {"score": 1.0, "details": {"difficulty_range": 1, "current_difficulty_frequency": 0}}
        
        # Count difficulties
        difficulty_counts = Counter(q["difficulty_level"] for q in recent_questions)
        total_questions = len(recent_questions)
        
        # Calculate difficulty diversity
        difficulty_range = max(difficulty_counts.keys()) - min(difficulty_counts.keys()) + 1
        current_difficulty_frequency = difficulty_counts.get(current_difficulty, 0) / total_questions
        
        # Diversity score: higher for wider range, lower for overused current difficulty
        diversity_score = min(1.0, difficulty_range / 5) * (1.0 - current_difficulty_frequency)
        
        return {
            "score": diversity_score,
            "details": {
                "difficulty_range": difficulty_range,
                "current_difficulty_frequency": current_difficulty_frequency,
                "difficulty_distribution": dict(difficulty_counts)
            }
        }
    
    def _analyze_error_pattern_diversity(
        self,
        recent_questions: List[Dict[str, Any]],
        current_error_patterns: List[str]
    ) -> Dict[str, Any]:
        """Analyze error pattern diversity"""
        
        if not recent_questions:
            return {"score": 1.0, "details": {"pattern_count": 0, "current_pattern_overlap": 0}}
        
        # Collect all error patterns
        all_patterns = []
        for q in recent_questions:
            if q["error_patterns"]:
                all_patterns.extend(q["error_patterns"])
        
        pattern_counts = Counter(all_patterns)
        total_questions = len(recent_questions)
        
        # Calculate pattern diversity
        pattern_count = len(pattern_counts)
        current_pattern_overlap = 0
        
        for pattern in current_error_patterns:
            if pattern in pattern_counts:
                current_pattern_overlap += pattern_counts[pattern] / total_questions
        
        # Normalize overlap
        current_pattern_overlap = min(1.0, current_pattern_overlap / len(current_error_patterns)) if current_error_patterns else 0
        
        # Diversity score: higher for more patterns, lower for overused current patterns
        diversity_score = min(1.0, pattern_count / 10) * (1.0 - current_pattern_overlap)
        
        return {
            "score": diversity_score,
            "details": {
                "pattern_count": pattern_count,
                "current_pattern_overlap": current_pattern_overlap,
                "pattern_distribution": dict(pattern_counts)
            }
        }
    
    def _calculate_quality_metrics(self, content: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Calculate quality metrics for content"""
        
        # Content length
        content_length = len(content)
        length_score = min(1.0, content_length / 100)  # Normalize to 0-1
        
        # Content complexity
        words = content.split()
        unique_words = len(set(words))
        vocabulary_richness = unique_words / len(words) if words else 0
        
        # Grammar and structure
        sentence_count = len(re.split(r'[.!?]+', content))
        avg_sentence_length = len(words) / sentence_count if sentence_count > 0 else 0
        structure_score = 1.0 - abs(avg_sentence_length - 15) / 15  # Target 15 words per sentence
        
        # Context relevance (if context provided)
        context_relevance = 1.0
        if context:
            content_words = set(content.lower().split())
            context_words = set(context.lower().split())
            common_words = content_words & context_words
            context_relevance = len(common_words) / len(content_words) if content_words else 0
        
        # Overall quality score
        quality_score = (
            length_score * 0.2 +
            vocabulary_richness * 0.3 +
            structure_score * 0.3 +
            context_relevance * 0.2
        )
        
        return {
            "quality_score": quality_score,
            "content_length": content_length,
            "vocabulary_richness": vocabulary_richness,
            "structure_score": structure_score,
            "context_relevance": context_relevance,
            "is_high_quality": quality_score >= self.quality_threshold
        }
    
    def _calculate_novelty_score(
        self,
        exact_duplicates: List[Dict[str, Any]],
        semantic_duplicates: List[Dict[str, Any]],
        diversity_analysis: Dict[str, Any],
        quality_metrics: Dict[str, Any]
    ) -> float:
        """Calculate overall novelty score"""
        
        # Base score starts at 1.0
        novelty_score = 1.0
        
        # Penalty for exact duplicates
        if exact_duplicates:
            novelty_score *= 0.1  # Heavy penalty for exact duplicates
        
        # Penalty for semantic duplicates
        if semantic_duplicates:
            max_similarity = max(d["similarity"] for d in semantic_duplicates)
            semantic_penalty = max_similarity * 0.5
            novelty_score *= (1.0 - semantic_penalty)
        
        # Bonus for diversity
        diversity_score = diversity_analysis.get("overall_diversity_score", 0.5)
        novelty_score *= (0.8 + diversity_score * 0.2)
        
        # Bonus for quality
        quality_score = quality_metrics.get("quality_score", 0.5)
        novelty_score *= (0.8 + quality_score * 0.2)
        
        return max(0.0, min(1.0, novelty_score))
    
    def _generate_novelty_recommendations(
        self,
        exact_duplicates: List[Dict[str, Any]],
        semantic_duplicates: List[Dict[str, Any]],
        diversity_analysis: Dict[str, Any],
        quality_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on novelty analysis"""
        
        recommendations = []
        
        # Duplicate recommendations
        if exact_duplicates:
            recommendations.append("Content is an exact duplicate of existing questions")
            recommendations.append("Consider modifying content to make it unique")
        
        if semantic_duplicates:
            recommendations.append(f"Content is very similar to {len(semantic_duplicates)} existing questions")
            recommendations.append("Consider changing topic or approach to increase novelty")
        
        # Diversity recommendations
        diversity_score = diversity_analysis.get("overall_diversity_score", 0.5)
        if diversity_score < self.diversity_threshold:
            recommendations.append("Content lacks diversity compared to recent questions")
            recommendations.append("Consider using different topics, formats, or difficulty levels")
        
        # Quality recommendations
        quality_score = quality_metrics.get("quality_score", 0.5)
        if quality_score < self.quality_threshold:
            recommendations.append("Content quality is below threshold")
            recommendations.append("Consider improving length, vocabulary, or structure")
        
        # Positive feedback
        if not recommendations:
            recommendations.append("Content shows good novelty and quality")
            recommendations.append("No major issues detected")
        
        return recommendations
    
    async def get_novelty_statistics(self) -> Dict[str, Any]:
        """Get statistics about novelty detection"""
        
        return {
            "similarity_threshold": self.similarity_threshold,
            "diversity_threshold": self.diversity_threshold,
            "quality_threshold": self.quality_threshold,
            "fingerprint_length": self.fingerprint_length,
            "topic_diversity_weight": self.topic_diversity_weight,
            "format_diversity_weight": self.format_diversity_weight,
            "difficulty_diversity_weight": self.difficulty_diversity_weight,
            "error_pattern_diversity_weight": self.error_pattern_diversity_weight
        }


# Global instance
novelty_detection_service = NoveltyDetectionService()
