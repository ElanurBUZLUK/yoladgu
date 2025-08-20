from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, update, delete
from datetime import datetime, timedelta
import uuid

from app.models.user import User
from app.models.question import Question, Subject
from app.models.student_attempt import StudentAttempt
from app.models.error_pattern import ErrorPattern
from app.models.math_error_detail import MathErrorDetail
from app.schemas.answer import ErrorAnalysisResult, DetailedErrorAnalysis
from app.core.cache import cache_service


class ErrorPatternService:
    """Error pattern tracking and analytics service"""
    
    def __init__(self):
        pass
    
    async def track_error_pattern(
        self, 
        db: AsyncSession, 
        user_id: str,
        subject: Subject,
        error_type: str,
        topic_category: Optional[str] = None,
        difficulty_level: Optional[int] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> ErrorPattern:
        """Track and update error patterns for a user"""
        
        # Check if error pattern already exists
        result = await db.execute(
            select(ErrorPattern).where(
                and_(
                    ErrorPattern.user_id == user_id,
                    ErrorPattern.subject == subject,
                    ErrorPattern.error_type == error_type,
                    ErrorPattern.topic_category == topic_category
                )
            )
        )
        
        error_pattern = result.scalar_one_or_none()
        
        if error_pattern:
            # Update existing pattern
            error_pattern.error_count += 1
            error_pattern.last_occurrence = datetime.utcnow()
            if difficulty_level:
                error_pattern.difficulty_level = difficulty_level
        else:
            # Create new pattern
            error_pattern = ErrorPattern(
                id=uuid.uuid4(),
                user_id=user_id,
                subject=subject,
                error_type=error_type,
                error_count=1,
                topic_category=topic_category,
                difficulty_level=difficulty_level
            )
            db.add(error_pattern)
        
        await db.commit()
        await db.refresh(error_pattern)
        
        # Invalidate cache
        await self._invalidate_error_cache(user_id, subject)
        
        return error_pattern
    
    async def get_user_error_patterns(
        self, 
        db: AsyncSession, 
        user_id: str,
        subject: Optional[Subject] = None,
        limit: int = 20
    ) -> List[ErrorAnalysisResult]:
        """Get error patterns for a user with analysis"""
        
        # Try cache first
        cache_key = f"error_patterns:{user_id}:{subject.value if subject else 'all'}"
        cached_patterns = await cache_service.get(cache_key)
        
        if cached_patterns:
            return [ErrorAnalysisResult(**pattern) for pattern in cached_patterns]
        
        # Build query
        query = select(ErrorPattern).where(ErrorPattern.user_id == user_id)
        
        if subject:
            query = query.where(ErrorPattern.subject == subject)
        
        query = query.order_by(desc(ErrorPattern.error_count)).limit(limit)
        
        result = await db.execute(query)
        error_patterns = result.scalars().all()
        
        # Convert to analysis results
        analysis_results = []
        
        for pattern in error_patterns:
            # Generate practice recommendations
            recommendations = await self._generate_practice_recommendations(pattern)
            
            # Find related topics
            related_topics = await self._find_related_topics(db, pattern)
            
            analysis_result = ErrorAnalysisResult(
                error_type=pattern.error_type,
                description=self._get_error_description(pattern.error_type, pattern.subject),
                frequency=pattern.error_count,
                last_occurrence=pattern.last_occurrence.isoformat(),
                subject=pattern.subject,
                topic_category=pattern.topic_category,
                difficulty_level=pattern.difficulty_level,
                practice_recommendations=recommendations,
                related_topics=related_topics
            )
            
            analysis_results.append(analysis_result)
        
        # Cache results for 1 hour
        cache_data = [result.dict() for result in analysis_results]
        await cache_service.set(cache_key, cache_data, expire=3600)
        
        return analysis_results
    
    async def get_similar_students(
        self, 
        db: AsyncSession, 
        user_id: str,
        subject: Subject,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find students with similar error patterns"""
        
        # Get current user's error patterns
        user_patterns = await self.get_user_error_patterns(db, user_id, subject, limit=10)
        
        if not user_patterns:
            return []
        
        # Get top error types for the user
        top_errors = [pattern.error_type for pattern in user_patterns[:5]]
        
        # Find other users with similar error patterns
        similar_users_query = select(
            ErrorPattern.user_id,
            func.count(ErrorPattern.error_type).label('common_errors'),
            func.sum(ErrorPattern.error_count).label('total_errors')
        ).where(
            and_(
                ErrorPattern.user_id != user_id,
                ErrorPattern.subject == subject,
                ErrorPattern.error_type.in_(top_errors)
            )
        ).group_by(ErrorPattern.user_id).having(
            func.count(ErrorPattern.error_type) >= 2  # At least 2 common errors
        ).order_by(desc('common_errors')).limit(limit)
        
        result = await db.execute(similar_users_query)
        similar_users_data = result.fetchall()
        
        similar_students = []
        
        for user_data in similar_users_data:
            similar_user_id, common_errors, total_errors = user_data
            
            # Get user info
            user_result = await db.execute(
                select(User).where(User.id == similar_user_id)
            )
            user = user_result.scalar_one_or_none()
            
            if user:
                # Calculate similarity score
                similarity_score = min(common_errors / len(top_errors), 1.0)
                
                similar_students.append({
                    "user_id": str(similar_user_id),
                    "username": user.username,
                    "common_error_count": common_errors,
                    "similarity_score": similarity_score,
                    "total_errors": total_errors,
                    "learning_style": user.learning_style.value
                })
        
        return similar_students
    
    async def get_error_trend_analysis(
        self, 
        db: AsyncSession, 
        user_id: str,
        subject: Subject,
        days: int = 30
    ) -> Dict[str, Any]:
        """Analyze error trends over time"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get attempts with errors in the time period
        query = select(StudentAttempt, Question).join(Question).where(
            and_(
                StudentAttempt.user_id == user_id,
                StudentAttempt.is_correct == False,
                StudentAttempt.attempt_date >= cutoff_date,
                Question.subject == subject
            )
        ).order_by(StudentAttempt.attempt_date)
        
        result = await db.execute(query)
        error_attempts = result.fetchall()
        
        if not error_attempts:
            return {
                "total_errors": 0,
                "error_rate": 0.0,
                "trend": "stable",
                "weekly_breakdown": [],
                "most_frequent_errors": []
            }
        
        # Calculate weekly breakdown
        weekly_breakdown = {}
        error_counts = {}
        
        for attempt, question in error_attempts:
            # Week calculation
            week_start = attempt.attempt_date - timedelta(days=attempt.attempt_date.weekday())
            week_key = week_start.strftime("%Y-%m-%d")
            
            if week_key not in weekly_breakdown:
                weekly_breakdown[week_key] = {"errors": 0, "total_attempts": 0}
            
            weekly_breakdown[week_key]["errors"] += 1
            
            # Error type counting
            error_type = attempt.error_category or "unknown_error"
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Get total attempts for error rate calculation
        total_attempts_query = select(func.count(StudentAttempt.id)).join(Question).where(
            and_(
                StudentAttempt.user_id == user_id,
                StudentAttempt.attempt_date >= cutoff_date,
                Question.subject == subject
            )
        )
        
        total_result = await db.execute(total_attempts_query)
        total_attempts = total_result.scalar() or 0
        
        error_rate = (len(error_attempts) / total_attempts * 100) if total_attempts > 0 else 0
        
        # Calculate trend
        weeks = sorted(weekly_breakdown.keys())
        if len(weeks) >= 2:
            early_weeks = weeks[:len(weeks)//2]
            late_weeks = weeks[len(weeks)//2:]
            
            early_avg = sum(weekly_breakdown[week]["errors"] for week in early_weeks) / len(early_weeks)
            late_avg = sum(weekly_breakdown[week]["errors"] for week in late_weeks) / len(late_weeks)
            
            if late_avg < early_avg * 0.8:
                trend = "improving"
            elif late_avg > early_avg * 1.2:
                trend = "worsening"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        # Most frequent errors
        most_frequent = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_errors": len(error_attempts),
            "error_rate": error_rate,
            "trend": trend,
            "weekly_breakdown": [
                {
                    "week": week,
                    "errors": data["errors"],
                    "error_rate": (data["errors"] / data.get("total_attempts", 1)) * 100
                }
                for week, data in sorted(weekly_breakdown.items())
            ],
            "most_frequent_errors": [
                {"error_type": error_type, "count": count}
                for error_type, count in most_frequent
            ],
            "analysis_period": f"{days} days"
        }
    
    async def get_class_error_analytics(
        self, 
        db: AsyncSession, 
        subject: Subject,
        days: int = 30,
        min_students: int = 3
    ) -> Dict[str, Any]:
        """Get class-wide error analytics"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get all error patterns for the subject in the time period
        query = select(ErrorPattern).where(
            and_(
                ErrorPattern.subject == subject,
                ErrorPattern.last_occurrence >= cutoff_date
            )
        )
        
        result = await db.execute(query)
        error_patterns = result.scalars().all()
        
        if not error_patterns:
            return {
                "total_error_types": 0,
                "most_common_errors": [],
                "error_distribution": {},
                "affected_students": 0
            }
        
        # Analyze error distribution
        error_distribution = {}
        student_count = {}
        
        for pattern in error_patterns:
            error_type = pattern.error_type
            
            if error_type not in error_distribution:
                error_distribution[error_type] = {
                    "total_occurrences": 0,
                    "student_count": 0,
                    "avg_frequency": 0.0,
                    "difficulty_levels": []
                }
            
            error_distribution[error_type]["total_occurrences"] += pattern.error_count
            
            if pattern.user_id not in student_count:
                student_count[pattern.user_id] = set()
            student_count[pattern.user_id].add(error_type)
            
            if pattern.difficulty_level:
                error_distribution[error_type]["difficulty_levels"].append(pattern.difficulty_level)
        
        # Calculate student counts for each error type
        for error_type in error_distribution:
            students_with_error = sum(
                1 for user_errors in student_count.values() 
                if error_type in user_errors
            )
            error_distribution[error_type]["student_count"] = students_with_error
            
            if students_with_error > 0:
                error_distribution[error_type]["avg_frequency"] = (
                    error_distribution[error_type]["total_occurrences"] / students_with_error
                )
        
        # Filter errors that affect minimum number of students
        significant_errors = {
            error_type: data for error_type, data in error_distribution.items()
            if data["student_count"] >= min_students
        }
        
        # Most common errors (by student count)
        most_common = sorted(
            significant_errors.items(),
            key=lambda x: (x[1]["student_count"], x[1]["total_occurrences"]),
            reverse=True
        )[:10]
        
        return {
            "total_error_types": len(error_distribution),
            "significant_error_types": len(significant_errors),
            "most_common_errors": [
                {
                    "error_type": error_type,
                    "student_count": data["student_count"],
                    "total_occurrences": data["total_occurrences"],
                    "avg_frequency": data["avg_frequency"],
                    "description": self._get_error_description(error_type, subject)
                }
                for error_type, data in most_common
            ],
            "error_distribution": significant_errors,
            "affected_students": len(student_count),
            "analysis_period": f"{days} days",
            "subject": subject.value
        }
    
    async def recommend_interventions(
        self, 
        db: AsyncSession, 
        user_id: str,
        subject: Subject
    ) -> List[Dict[str, Any]]:
        """Recommend interventions based on error patterns"""
        
        # Get user's error patterns
        error_patterns = await self.get_user_error_patterns(db, user_id, subject, limit=5)
        
        if not error_patterns:
            return []
        
        interventions = []
        
        for pattern in error_patterns:
            intervention = await self._generate_intervention_recommendation(pattern, db)
            if intervention:
                interventions.append(intervention)
        
        return interventions
    
    async def update_error_pattern_metadata(
        self, 
        db: AsyncSession, 
        pattern_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Update error pattern with additional metadata"""
        
        result = await db.execute(
            select(ErrorPattern).where(ErrorPattern.id == pattern_id)
        )
        
        pattern = result.scalar_one_or_none()
        
        if not pattern:
            return False
        
        # Update metadata (this would require adding a metadata field to ErrorPattern model)
        # For now, we'll just update the last_occurrence
        pattern.last_occurrence = datetime.utcnow()
        
        await db.commit()
        return True
    
    # Private helper methods
    async def _invalidate_error_cache(self, user_id: str, subject: Subject):
        """Invalidate error pattern caches"""
        cache_keys = [
            f"error_patterns:{user_id}:{subject.value}",
            f"error_patterns:{user_id}:all"
        ]
        
        for key in cache_keys:
            await cache_service.delete(key)
    
    async def _generate_practice_recommendations(self, pattern: ErrorPattern) -> List[str]:
        """Generate practice recommendations based on error pattern"""
        
        recommendations = []
        
        if pattern.subject == Subject.MATH:
            math_recommendations = {
                "addition_error": [
                    "Practice basic addition with carrying",
                    "Use visual aids like number lines",
                    "Try mental math exercises"
                ],
                "subtraction_error": [
                    "Practice borrowing in subtraction",
                    "Use manipulatives for concrete understanding",
                    "Work on place value concepts"
                ],
                "multiplication_error": [
                    "Review multiplication tables",
                    "Practice with arrays and groups",
                    "Use the distributive property"
                ],
                "division_error": [
                    "Practice long division steps",
                    "Use estimation to check answers",
                    "Work with remainders"
                ],
                "fraction_error": [
                    "Practice finding common denominators",
                    "Use visual fraction models",
                    "Work on equivalent fractions"
                ],
                "algebra_error": [
                    "Review order of operations",
                    "Practice isolating variables",
                    "Work on combining like terms"
                ]
            }
            
            for error_key, recs in math_recommendations.items():
                if error_key in pattern.error_type.lower():
                    recommendations.extend(recs)
                    break
            
            if not recommendations:
                recommendations.append(f"Practice {pattern.topic_category or 'math'} problems at your level")
        
        elif pattern.subject == Subject.ENGLISH:
            english_recommendations = {
                "past_tense": [
                    "Practice regular and irregular past tense verbs",
                    "Create sentences using past tense",
                    "Read stories and identify past tense verbs"
                ],
                "present_perfect": [
                    "Learn the structure: have/has + past participle",
                    "Practice with time expressions (already, yet, just)",
                    "Distinguish between past simple and present perfect"
                ],
                "grammar": [
                    "Review basic grammar rules",
                    "Practice with grammar exercises",
                    "Read and analyze sentence structures"
                ],
                "vocabulary": [
                    "Build vocabulary with flashcards",
                    "Read more to encounter new words",
                    "Practice using new words in sentences"
                ],
                "preposition": [
                    "Study common preposition patterns",
                    "Practice with preposition exercises",
                    "Learn phrasal verbs"
                ]
            }
            
            for error_key, recs in english_recommendations.items():
                if error_key in pattern.error_type.lower():
                    recommendations.extend(recs)
                    break
            
            if not recommendations:
                recommendations.append(f"Practice {pattern.topic_category or 'English'} exercises")
        
        return recommendations[:3]  # Limit to 3 recommendations
    
    async def _find_related_topics(self, db: AsyncSession, pattern: ErrorPattern) -> List[str]:
        """Find topics related to the error pattern"""
        
        related_topics = []
        
        if pattern.topic_category:
            related_topics.append(pattern.topic_category)
        
        # Find other topics where this user has similar errors
        result = await db.execute(
            select(ErrorPattern.topic_category).where(
                and_(
                    ErrorPattern.user_id == pattern.user_id,
                    ErrorPattern.subject == pattern.subject,
                    ErrorPattern.topic_category.isnot(None),
                    ErrorPattern.topic_category != pattern.topic_category
                )
            ).distinct().limit(3)
        )
        
        other_topics = [row[0] for row in result.fetchall()]
        related_topics.extend(other_topics)
        
        return list(set(related_topics))  # Remove duplicates
    
    def _get_error_description(self, error_type: str, subject: Subject) -> str:
        """Get human-readable description of error type"""
        
        descriptions = {
            Subject.MATH: {
                "addition_error": "Difficulty with addition operations and carrying",
                "subtraction_error": "Problems with subtraction and borrowing",
                "multiplication_error": "Mistakes in multiplication calculations",
                "division_error": "Errors in division operations and remainders",
                "fraction_error": "Confusion with fraction operations and concepts",
                "decimal_error": "Mistakes with decimal calculations",
                "algebra_error": "Difficulty with algebraic expressions and equations",
                "geometry_error": "Problems with geometric concepts and calculations",
                "word_problem_error": "Difficulty interpreting and solving word problems",
                "calculation_error": "General computational mistakes"
            },
            Subject.ENGLISH: {
                "past_tense_error": "Difficulty with past tense verb forms",
                "present_perfect_error": "Confusion with present perfect tense",
                "grammar_error": "General grammar mistakes",
                "vocabulary_error": "Incorrect word usage or meaning",
                "preposition_error": "Mistakes with preposition usage",
                "article_error": "Confusion with articles (a, an, the)",
                "plural_error": "Mistakes with singular/plural forms",
                "sentence_structure_error": "Problems with sentence construction",
                "spelling_error": "Spelling mistakes",
                "punctuation_error": "Incorrect punctuation usage"
            }
        }
        
        subject_descriptions = descriptions.get(subject, {})
        return subject_descriptions.get(error_type, f"Error type: {error_type}")
    
    async def _generate_intervention_recommendation(
        self, 
        pattern: ErrorAnalysisResult, 
        db: AsyncSession
    ) -> Optional[Dict[str, Any]]:
        """Generate intervention recommendation for an error pattern"""
        
        if pattern.frequency < 3:
            return None  # Not frequent enough for intervention
        
        intervention = {
            "error_type": pattern.error_type,
            "priority": "high" if pattern.frequency >= 5 else "medium",
            "intervention_type": "practice",
            "description": f"Targeted practice for {pattern.description.lower()}",
            "recommended_actions": pattern.practice_recommendations,
            "estimated_time": "15-20 minutes daily",
            "success_criteria": "Reduce error frequency by 50% in 2 weeks"
        }
        
        # Add specific intervention based on error type
        if pattern.subject == Subject.MATH:
            if "calculation" in pattern.error_type.lower():
                intervention["intervention_type"] = "skill_building"
                intervention["recommended_actions"].append("Use calculator to verify answers")
            elif "word_problem" in pattern.error_type.lower():
                intervention["intervention_type"] = "strategy_training"
                intervention["recommended_actions"].append("Practice problem-solving strategies")
        
        elif pattern.subject == Subject.ENGLISH:
            if "grammar" in pattern.error_type.lower():
                intervention["intervention_type"] = "rule_learning"
                intervention["recommended_actions"].append("Study grammar rules with examples")
            elif "vocabulary" in pattern.error_type.lower():
                intervention["intervention_type"] = "vocabulary_building"
                intervention["recommended_actions"].append("Use spaced repetition for new words")
        
        return intervention


# Global error pattern service instance
error_pattern_service = ErrorPatternService()