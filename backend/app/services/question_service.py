from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc, asc, update
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status
from datetime import datetime, timedelta
import uuid
import random

from app.models.question import Question, Subject, QuestionType, SourceType
from app.models.user import User, LearningStyle
from app.models.student_attempt import StudentAttempt
from app.models.error_pattern import ErrorPattern
from app.schemas.question import (
    QuestionCreate, QuestionUpdate, QuestionRecommendationRequest,
    QuestionRecommendationResponse, QuestionSearchQuery, QuestionStats,
    QuestionDifficultyAdjustment, QuestionPool, QuestionValidationResult,
    BulkQuestionOperation, QuestionResponse
)
from app.core.cache import cache_service


class QuestionService:
    """Question management and recommendation service"""
    
    def __init__(self):
        pass
    
    async def create_question(self, db: AsyncSession, question_data: QuestionCreate) -> Question:
        """Create a new question"""
        
        # Validate question data
        validation_result = await self._validate_question_data(question_data)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Question validation failed: {', '.join(validation_result.errors)}"
            )
        
        # Create question
        db_question = Question(
            id=uuid.uuid4(),
            subject=question_data.subject,
            content=question_data.content,
            question_type=question_data.question_type,
            difficulty_level=question_data.difficulty_level,
            original_difficulty=question_data.difficulty_level,
            topic_category=question_data.topic_category,
            correct_answer=question_data.correct_answer,
            options=question_data.options,
            source_type=question_data.source_type,
            pdf_source_path=question_data.pdf_source_path,
            question_metadata=question_data.question_metadata or {}
        )
        
        try:
            db.add(db_question)
            await db.commit()
            await db.refresh(db_question)
            
            # Invalidate related caches
            await self._invalidate_question_caches(question_data.subject, question_data.difficulty_level)
            
            return db_question
        except IntegrityError:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question creation failed"
            )
    
    async def get_question_by_id(self, db: AsyncSession, question_id: str) -> Optional[Question]:
        """Get question by ID"""
        result = await db.execute(select(Question).where(Question.id == question_id))
        return result.scalar_one_or_none()
    
    async def update_question(self, db: AsyncSession, question_id: str, question_update: QuestionUpdate) -> Optional[Question]:
        """Update question"""
        
        question = await self.get_question_by_id(db, question_id)
        if not question:
            return None
        
        # Prepare update data
        update_data = {}
        
        if question_update.content is not None:
            update_data["content"] = question_update.content
        
        if question_update.question_type is not None:
            update_data["question_type"] = question_update.question_type
        
        if question_update.difficulty_level is not None:
            update_data["difficulty_level"] = question_update.difficulty_level
        
        if question_update.topic_category is not None:
            update_data["topic_category"] = question_update.topic_category
        
        if question_update.correct_answer is not None:
            update_data["correct_answer"] = question_update.correct_answer
        
        if question_update.options is not None:
            update_data["options"] = question_update.options
        
        if question_update.question_metadata is not None:
            update_data["question_metadata"] = question_update.question_metadata
        
        if not update_data:
            return question
        
        # Update question
        await db.execute(
            update(Question).where(Question.id == question_id).values(**update_data)
        )
        await db.commit()
        
        # Invalidate caches
        await self._invalidate_question_caches(question.subject, question.difficulty_level)
        
        return await self.get_question_by_id(db, question_id)
    
    async def delete_question(self, db: AsyncSession, question_id: str) -> bool:
        """Delete question (soft delete by archiving)"""
        
        question = await self.get_question_by_id(db, question_id)
        if not question:
            return False
        
        # Add archived flag to metadata
        metadata = question.question_metadata or {}
        metadata["archived"] = True
        metadata["archived_at"] = datetime.utcnow().isoformat()
        
        await db.execute(
            update(Question).where(Question.id == question_id).values(question_metadata=metadata)
        )
        await db.commit()
        
        # Invalidate caches
        await self._invalidate_question_caches(question.subject, question.difficulty_level)
        
        return True
    
    async def search_questions(
        self, 
        db: AsyncSession, 
        search_query: QuestionSearchQuery,
        skip: int = 0,
        limit: int = 100
    ) -> List[Question]:
        """Search questions with filters"""
        
        query = select(Question)
        conditions = []
        
        # Exclude archived questions
        conditions.append(
            or_(
                Question.question_metadata.is_(None),
                ~Question.question_metadata.op('?')('archived')
            )
        )
        
        if search_query.query:
            search_term = f"%{search_query.query}%"
            conditions.append(
                or_(
                    Question.content.ilike(search_term),
                    Question.topic_category.ilike(search_term),
                    Question.correct_answer.ilike(search_term)
                )
            )
        
        if search_query.subject:
            conditions.append(Question.subject == search_query.subject)
        
        if search_query.difficulty_level:
            conditions.append(Question.difficulty_level == search_query.difficulty_level)
        
        if search_query.question_type:
            conditions.append(Question.question_type == search_query.question_type)
        
        if search_query.topic_category:
            conditions.append(Question.topic_category.ilike(f"%{search_query.topic_category}%"))
        
        if search_query.source_type:
            conditions.append(Question.source_type == search_query.source_type)
        
        if search_query.min_difficulty:
            conditions.append(Question.difficulty_level >= search_query.min_difficulty)
        
        if search_query.max_difficulty:
            conditions.append(Question.difficulty_level <= search_query.max_difficulty)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        query = query.offset(skip).limit(limit)
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def recommend_questions(
        self, 
        db: AsyncSession, 
        user: User,
        request: QuestionRecommendationRequest
    ) -> QuestionRecommendationResponse:
        """Recommend questions based on user profile and preferences"""
        
        # Determine user level for the subject
        user_level = request.user_level or (
            user.current_math_level if request.subject == Subject.MATH 
            else user.current_english_level
        )
        
        # Get user's error patterns for this subject
        error_patterns = await self._get_user_error_patterns(db, user.id, request.subject)
        
        # Get recently attempted questions to exclude
        recent_questions = []
        if request.exclude_recent:
            recent_questions = await self._get_recent_question_ids(db, user.id, days=7)
        
        # Build recommendation query
        questions = await self._build_recommendation_query(
            db, request, user_level, error_patterns, recent_questions
        )
        
        # Apply learning style adaptations
        if request.learning_style or user.learning_style:
            learning_style = request.learning_style or user.learning_style.value
            questions = await self._apply_learning_style_filter(questions, learning_style)
        
        # Limit results
        questions = questions[:request.limit]
        
        # Generate recommendation reason
        reason = await self._generate_recommendation_reason(
            user_level, error_patterns, request.subject, len(questions)
        )
        
        # Get total available questions count
        total_available = await self._count_available_questions(db, request.subject, user_level)
        
        # Generate next recommendations
        next_recommendations = await self._generate_next_recommendations(
            db, user, request.subject, error_patterns
        )
        
        # Convert to response format
        question_responses = [
            QuestionResponse(
                id=str(q.id),
                subject=q.subject,
                content=q.content,
                question_type=q.question_type,
                difficulty_level=q.difficulty_level,
                original_difficulty=q.original_difficulty,
                topic_category=q.topic_category,
                correct_answer=q.correct_answer,
                options=q.options,
                source_type=q.source_type,
                pdf_source_path=q.pdf_source_path,
                question_metadata=q.question_metadata,
                created_at=q.created_at.isoformat()
            )
            for q in questions
        ]
        
        return QuestionRecommendationResponse(
            questions=question_responses,
            recommendation_reason=reason,
            difficulty_adjustment=None,  # Will be set if difficulty was adjusted
            total_available=total_available,
            user_level=user_level,
            next_recommendations=next_recommendations
        )
    
    async def adjust_question_difficulty(
        self, 
        db: AsyncSession, 
        question_id: str, 
        new_difficulty: int,
        reason: str,
        adjusted_by: str
    ) -> QuestionDifficultyAdjustment:
        """Adjust question difficulty level"""
        
        question = await self.get_question_by_id(db, question_id)
        if not question:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Question not found"
            )
        
        if not (1 <= new_difficulty <= 5):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Difficulty level must be between 1 and 5"
            )
        
        old_difficulty = question.difficulty_level
        
        # Update difficulty
        await db.execute(
            update(Question).where(Question.id == question_id).values(difficulty_level=new_difficulty)
        )
        await db.commit()
        
        # Create adjustment record
        adjustment = QuestionDifficultyAdjustment(
            question_id=question_id,
            old_difficulty=old_difficulty,
            new_difficulty=new_difficulty,
            reason=reason,
            adjusted_by=adjusted_by,
            adjustment_date=datetime.utcnow().isoformat()
        )
        
        # Invalidate caches
        await self._invalidate_question_caches(question.subject, old_difficulty)
        await self._invalidate_question_caches(question.subject, new_difficulty)
        
        return adjustment
    
    async def get_question_pool(
        self, 
        db: AsyncSession, 
        subject: Subject, 
        difficulty_level: int,
        topic_category: Optional[str] = None
    ) -> QuestionPool:
        """Get question pool for specific criteria"""
        
        cache_key = f"question_pool:{subject.value}:{difficulty_level}:{topic_category or 'all'}"
        
        # Try to get from cache
        cached_pool = await cache_service.get(cache_key)
        if cached_pool:
            return QuestionPool(**cached_pool)
        
        # Build query
        query = select(Question).where(
            and_(
                Question.subject == subject,
                Question.difficulty_level == difficulty_level,
                or_(
                    Question.question_metadata.is_(None),
                    ~Question.question_metadata.op('?')('archived')
                )
            )
        )
        
        if topic_category:
            query = query.where(Question.topic_category == topic_category)
        
        result = await db.execute(query)
        questions = result.scalars().all()
        
        # Convert to response format
        question_responses = [
            QuestionResponse(
                id=str(q.id),
                subject=q.subject,
                content=q.content,
                question_type=q.question_type,
                difficulty_level=q.difficulty_level,
                original_difficulty=q.original_difficulty,
                topic_category=q.topic_category,
                correct_answer=q.correct_answer,
                options=q.options,
                source_type=q.source_type,
                pdf_source_path=q.pdf_source_path,
                question_metadata=q.question_metadata,
                created_at=q.created_at.isoformat()
            )
            for q in questions
        ]
        
        pool = QuestionPool(
            subject=subject,
            difficulty_level=difficulty_level,
            topic_category=topic_category or "all",
            questions=question_responses,
            pool_size=len(questions),
            last_updated=datetime.utcnow().isoformat()
        )
        
        # Cache for 1 hour
        await cache_service.set(cache_key, pool.dict(), expire=3600)
        
        return pool
    
    async def get_question_stats(self, db: AsyncSession) -> QuestionStats:
        """Get question statistics"""
        
        # Total questions
        total_result = await db.execute(
            select(func.count(Question.id)).where(
                or_(
                    Question.question_metadata.is_(None),
                    ~Question.question_metadata.op('?')('archived')
                )
            )
        )
        total_questions = total_result.scalar()
        
        # By subject
        subject_stats = {}
        for subject in Subject:
            subject_result = await db.execute(
                select(func.count(Question.id)).where(
                    and_(
                        Question.subject == subject,
                        or_(
                            Question.question_metadata.is_(None),
                            ~Question.question_metadata.op('?')('archived')
                        )
                    )
                )
            )
            subject_stats[subject.value] = subject_result.scalar()
        
        # By difficulty
        difficulty_stats = {}
        for level in range(1, 6):
            diff_result = await db.execute(
                select(func.count(Question.id)).where(
                    and_(
                        Question.difficulty_level == level,
                        or_(
                            Question.question_metadata.is_(None),
                            ~Question.question_metadata.op('?')('archived')
                        )
                    )
                )
            )
            difficulty_stats[str(level)] = diff_result.scalar()
        
        # By type
        type_stats = {}
        for qtype in QuestionType:
            type_result = await db.execute(
                select(func.count(Question.id)).where(
                    and_(
                        Question.question_type == qtype,
                        or_(
                            Question.question_metadata.is_(None),
                            ~Question.question_metadata.op('?')('archived')
                        )
                    )
                )
            )
            type_stats[qtype.value] = type_result.scalar()
        
        # By source
        source_stats = {}
        for source in SourceType:
            source_result = await db.execute(
                select(func.count(Question.id)).where(
                    and_(
                        Question.source_type == source,
                        or_(
                            Question.question_metadata.is_(None),
                            ~Question.question_metadata.op('?')('archived')
                        )
                    )
                )
            )
            source_stats[source.value] = source_result.scalar()
        
        # Average difficulty
        avg_result = await db.execute(
            select(func.avg(Question.difficulty_level)).where(
                or_(
                    Question.question_metadata.is_(None),
                    ~Question.question_metadata.op('?')('archived')
                )
            )
        )
        average_difficulty = float(avg_result.scalar() or 0)
        
        # Most common topics
        topic_result = await db.execute(
            select(Question.topic_category, func.count(Question.id).label('count'))
            .where(
                or_(
                    Question.question_metadata.is_(None),
                    ~Question.question_metadata.op('?')('archived')
                )
            )
            .group_by(Question.topic_category)
            .order_by(desc('count'))
            .limit(10)
        )
        
        most_common_topics = [
            {"topic": row[0], "count": row[1]}
            for row in topic_result.fetchall()
        ]
        
        return QuestionStats(
            total_questions=total_questions,
            by_subject=subject_stats,
            by_difficulty=difficulty_stats,
            by_type=type_stats,
            by_source=source_stats,
            average_difficulty=average_difficulty,
            most_common_topics=most_common_topics
        )
    
    async def bulk_question_operation(
        self, 
        db: AsyncSession, 
        operation_data: BulkQuestionOperation
    ) -> Dict[str, Any]:
        """Perform bulk operations on questions"""
        
        if operation_data.operation not in ["adjust_difficulty", "change_category", "archive"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid operation"
            )
        
        success_count = 0
        failed_ids = []
        
        for question_id in operation_data.question_ids:
            try:
                if operation_data.operation == "adjust_difficulty":
                    new_difficulty = operation_data.parameters.get("new_difficulty")
                    if new_difficulty and 1 <= new_difficulty <= 5:
                        await db.execute(
                            update(Question).where(Question.id == question_id)
                            .values(difficulty_level=new_difficulty)
                        )
                
                elif operation_data.operation == "change_category":
                    new_category = operation_data.parameters.get("new_category")
                    if new_category:
                        await db.execute(
                            update(Question).where(Question.id == question_id)
                            .values(topic_category=new_category)
                        )
                
                elif operation_data.operation == "archive":
                    question = await self.get_question_by_id(db, question_id)
                    if question:
                        metadata = question.question_metadata or {}
                        metadata["archived"] = True
                        metadata["archived_at"] = datetime.utcnow().isoformat()
                        
                        await db.execute(
                            update(Question).where(Question.id == question_id)
                            .values(question_metadata=metadata)
                        )
                
                success_count += 1
            except Exception:
                failed_ids.append(question_id)
        
        await db.commit()
        
        # Invalidate caches
        await cache_service.delete("question_stats")
        
        return {
            "operation": operation_data.operation,
            "total_requested": len(operation_data.question_ids),
            "successful": success_count,
            "failed": len(failed_ids),
            "failed_ids": failed_ids
        }
    
    # Private helper methods
    async def _validate_question_data(self, question_data: QuestionCreate) -> QuestionValidationResult:
        """Validate question data"""
        
        errors = []
        warnings = []
        suggestions = []
        
        # Content validation
        if len(question_data.content.strip()) < 10:
            errors.append("Question content is too short")
        
        # Multiple choice validation
        if question_data.question_type == QuestionType.MULTIPLE_CHOICE:
            if not question_data.options or len(question_data.options) < 2:
                errors.append("Multiple choice questions must have at least 2 options")
            
            if not question_data.correct_answer:
                errors.append("Multiple choice questions must have a correct answer")
            
            if question_data.correct_answer and question_data.options:
                if question_data.correct_answer not in question_data.options:
                    warnings.append("Correct answer should be one of the provided options")
        
        # Fill blank validation
        if question_data.question_type == QuestionType.FILL_BLANK:
            if "_" not in question_data.content:
                warnings.append("Fill in the blank questions should contain underscores (_)")
        
        # Difficulty validation
        if question_data.difficulty_level < 1 or question_data.difficulty_level > 5:
            errors.append("Difficulty level must be between 1 and 5")
        
        # Quality suggestions
        if len(question_data.content) > 500:
            suggestions.append("Consider shortening the question for better readability")
        
        if question_data.question_type == QuestionType.MULTIPLE_CHOICE and question_data.options:
            if len(question_data.options) > 6:
                suggestions.append("Consider reducing the number of options for better usability")
        
        quality_score = 1.0
        if errors:
            quality_score -= len(errors) * 0.3
        if warnings:
            quality_score -= len(warnings) * 0.1
        
        quality_score = max(0.0, min(1.0, quality_score))
        
        return QuestionValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            quality_score=quality_score
        )
    
    async def _get_user_error_patterns(self, db: AsyncSession, user_id: str, subject: Subject) -> List[str]:
        """Get user's error patterns for a subject"""
        
        result = await db.execute(
            select(ErrorPattern.error_type).where(
                and_(
                    ErrorPattern.user_id == user_id,
                    ErrorPattern.subject == subject
                )
            ).order_by(desc(ErrorPattern.error_count)).limit(5)
        )
        
        return [row[0] for row in result.fetchall()]
    
    async def _get_recent_question_ids(self, db: AsyncSession, user_id: str, days: int = 7) -> List[str]:
        """Get recently attempted question IDs"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        result = await db.execute(
            select(StudentAttempt.question_id).where(
                and_(
                    StudentAttempt.user_id == user_id,
                    StudentAttempt.attempt_date >= cutoff_date
                )
            ).distinct()
        )
        
        return [str(row[0]) for row in result.fetchall()]
    
    async def _build_recommendation_query(
        self, 
        db: AsyncSession,
        request: QuestionRecommendationRequest,
        user_level: int,
        error_patterns: List[str],
        recent_questions: List[str]
    ) -> List[Question]:
        """Build question recommendation query"""
        
        query = select(Question).where(
            and_(
                Question.subject == request.subject,
                or_(
                    Question.question_metadata.is_(None),
                    ~Question.question_metadata.op('?')('archived')
                )
            )
        )
        
        # Exclude recent questions
        if recent_questions:
            query = query.where(~Question.id.in_(recent_questions))
        
        # Difficulty range based on user level
        preferred_difficulty = request.preferred_difficulty or user_level
        min_diff = max(1, preferred_difficulty - 1)
        max_diff = min(5, preferred_difficulty + 1)
        
        query = query.where(
            and_(
                Question.difficulty_level >= min_diff,
                Question.difficulty_level <= max_diff
            )
        )
        
        # Topic categories
        if request.topic_categories:
            query = query.where(Question.topic_category.in_(request.topic_categories))
        
        # Question types
        if request.question_types:
            query = query.where(Question.question_type.in_(request.question_types))
        
        # Order by relevance (prefer questions matching error patterns)
        if error_patterns:
            # This would be more sophisticated in a real implementation
            query = query.order_by(func.random())
        else:
            query = query.order_by(func.random())
        
        query = query.limit(request.limit * 2)  # Get more than needed for filtering
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def _apply_learning_style_filter(self, questions: List[Question], learning_style: str) -> List[Question]:
        """Apply learning style preferences to question selection"""
        
        # This is a placeholder implementation
        # In a real system, questions would have metadata indicating their suitability for different learning styles
        
        if learning_style == "visual":
            # Prefer questions with diagrams, charts, or visual elements
            pass
        elif learning_style == "auditory":
            # Prefer questions that can be read aloud or have audio components
            pass
        elif learning_style == "kinesthetic":
            # Prefer interactive questions or hands-on problems
            pass
        
        return questions
    
    async def _generate_recommendation_reason(
        self, 
        user_level: int, 
        error_patterns: List[str], 
        subject: Subject,
        question_count: int
    ) -> str:
        """Generate explanation for why these questions were recommended"""
        
        reasons = []
        
        reasons.append(f"Based on your current {subject.value} level ({user_level})")
        
        if error_patterns:
            reasons.append(f"Focusing on areas where you need practice: {', '.join(error_patterns[:2])}")
        
        if question_count > 0:
            reasons.append(f"Selected {question_count} questions to help you improve")
        else:
            reasons.append("No suitable questions found for your current criteria")
        
        return ". ".join(reasons)
    
    async def _count_available_questions(self, db: AsyncSession, subject: Subject, user_level: int) -> int:
        """Count total available questions for user level"""
        
        result = await db.execute(
            select(func.count(Question.id)).where(
                and_(
                    Question.subject == subject,
                    Question.difficulty_level.between(max(1, user_level - 1), min(5, user_level + 1)),
                    or_(
                        Question.question_metadata.is_(None),
                        ~Question.question_metadata.op('?')('archived')
                    )
                )
            )
        )
        
        return result.scalar() or 0
    
    async def _generate_next_recommendations(
        self, 
        db: AsyncSession, 
        user: User, 
        subject: Subject,
        error_patterns: List[str]
    ) -> List[str]:
        """Generate suggestions for next study sessions"""
        
        recommendations = []
        
        if error_patterns:
            recommendations.append(f"Continue practicing {error_patterns[0]} problems")
        
        user_level = user.current_math_level if subject == Subject.MATH else user.current_english_level
        
        if user_level < 5:
            recommendations.append(f"Try some level {user_level + 1} questions when ready")
        
        recommendations.append("Review your mistakes to learn from them")
        
        return recommendations[:3]
    
    async def _invalidate_question_caches(self, subject: Subject, difficulty_level: int):
        """Invalidate related caches"""
        
        cache_keys = [
            f"question_pool:{subject.value}:{difficulty_level}:all",
            "question_stats"
        ]
        
        for key in cache_keys:
            await cache_service.delete(key)


# Global question service instance
question_service = QuestionService()