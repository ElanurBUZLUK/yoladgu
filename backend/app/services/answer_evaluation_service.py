from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc, update
from datetime import datetime, timedelta
import uuid
import re
import json

from app.models.user import User
from app.models.question import Question, Subject, QuestionType
from app.models.student_attempt import StudentAttempt
from app.models.error_pattern import ErrorPattern
from app.models.math_error_detail import MathErrorDetail
from app.schemas.answer import (
    AnswerSubmission, AnswerEvaluation, AnswerEvaluationRequest,
    StudentAttemptResponse, ErrorAnalysisResult, PerformanceMetrics,
    DetailedErrorAnalysis, LevelAdjustmentRecommendation,
    FeedbackGeneration
)
from pydantic import BaseModel
from typing import Optional, List
from app.services.llm_gateway import llm_gateway
from app.services.mcp_service import mcp_service
from app.core.cache import cache_service


class AnswerValidation(BaseModel):
    """Validation result for answer format"""
    is_valid_format: bool
    validation_errors: List[str] = []
    normalized_answer: Optional[str] = None
    confidence_score: float = 0.0


class AnswerEvaluationService:
    """Answer evaluation and analysis service"""
    
    def __init__(self):
        pass
    
    async def evaluate_answer(
        self, 
        db: AsyncSession, 
        user: User,
        evaluation_request: AnswerEvaluationRequest
    ) -> AnswerEvaluation:
        """Evaluate a student's answer"""
        
        # Get question details if not provided
        question = await self._get_question_details(db, evaluation_request.question_id)
        if not question:
            raise ValueError(f"Question not found: {evaluation_request.question_id}")
        
        # Fill in missing details from question
        if not evaluation_request.question_content:
            evaluation_request.question_content = question.content
        if not evaluation_request.correct_answer:
            evaluation_request.correct_answer = question.correct_answer
        if not evaluation_request.question_type:
            evaluation_request.question_type = question.question_type
        if not evaluation_request.subject:
            evaluation_request.subject = question.subject
        if not evaluation_request.difficulty_level:
            evaluation_request.difficulty_level = question.difficulty_level
        
        # Validate answer format
        validation = await self._validate_answer_format(
            evaluation_request.student_answer, 
            evaluation_request.question_type
        )
        
        if not validation.is_valid_format:
            return AnswerEvaluation(
                is_correct=False,
                score=0.0,
                feedback="Invalid answer format. Please check your response.",
                error_analysis={"format_errors": validation.validation_errors}
            )
        
        # Use normalized answer if available
        student_answer = validation.normalized_answer or evaluation_request.student_answer
        
        # Evaluate using appropriate method
        if evaluation_request.use_llm and evaluation_request.subject == Subject.ENGLISH:
            # Use LLM for English evaluation
            evaluation = await self._evaluate_with_llm(evaluation_request, student_answer)
        else:
            # Use rule-based evaluation
            evaluation = await self._evaluate_with_rules(evaluation_request, student_answer)
        
        # Enhance evaluation with additional analysis
        evaluation = await self._enhance_evaluation(db, user, question, evaluation, student_answer)
        
        # Save attempt to database
        await self._save_student_attempt(db, user, question, evaluation, student_answer, evaluation_request.time_spent)
        
        # Update error patterns
        await self._update_error_patterns(db, user, question, evaluation)
        
        return evaluation
    
    async def submit_answer(
        self, 
        db: AsyncSession, 
        user: User,
        submission: AnswerSubmission
    ) -> Tuple[AnswerEvaluation, StudentAttemptResponse]:
        """Submit and evaluate an answer"""
        
        # Create evaluation request
        evaluation_request = AnswerEvaluationRequest(
            question_id=submission.question_id,
            student_answer=submission.student_answer,
            time_spent=submission.time_spent,
            use_llm=True
        )
        
        # Evaluate answer
        evaluation = await self.evaluate_answer(db, user, evaluation_request)
        
        # Get the saved attempt
        attempt = await self._get_latest_attempt(db, user.id, submission.question_id)
        
        attempt_response = StudentAttemptResponse(
            id=str(attempt.id),
            question_id=str(attempt.question_id),
            student_answer=attempt.student_answer,
            is_correct=attempt.is_correct,
            time_spent=attempt.time_spent,
            attempt_date=attempt.attempt_date.isoformat(),
            error_category=attempt.error_category,
            score=evaluation.score,
            feedback=evaluation.feedback
        )
        
        return evaluation, attempt_response
    
    async def get_user_performance_metrics(
        self, 
        db: AsyncSession, 
        user_id: str,
        subject: Optional[Subject] = None,
        days: int = 30
    ) -> PerformanceMetrics:
        """Get comprehensive performance metrics for a user"""
        
        # Date range
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Base query
        query = select(StudentAttempt).where(
            and_(
                StudentAttempt.user_id == user_id,
                StudentAttempt.attempt_date >= cutoff_date
            )
        )
        
        if subject:
            # Join with Question to filter by subject
            query = query.join(Question).where(Question.subject == subject)
        
        result = await db.execute(query)
        attempts = result.scalars().all()
        
        if not attempts:
            return PerformanceMetrics(
                total_attempts=0,
                correct_attempts=0,
                accuracy_rate=0.0,
                average_score=0.0,
                average_time_per_question=0.0
            )
        
        # Calculate basic metrics
        total_attempts = len(attempts)
        correct_attempts = sum(1 for a in attempts if a.is_correct)
        accuracy_rate = (correct_attempts / total_attempts) * 100
        
        # Calculate average time
        time_attempts = [a for a in attempts if a.time_spent is not None]
        average_time = sum(a.time_spent for a in time_attempts) / len(time_attempts) if time_attempts else 0
        
        # Calculate average score (placeholder - would need score field in StudentAttempt)
        average_score = accuracy_rate  # Simplified for now
        
        # Subject-specific performance
        subject_performance = await self._calculate_subject_performance(db, user_id, attempts)
        
        # Difficulty performance
        difficulty_performance = await self._calculate_difficulty_performance(db, user_id, attempts)
        
        # Recent performance trend
        recent_accuracy = await self._calculate_recent_accuracy(attempts, days=7)
        improvement_trend = await self._calculate_improvement_trend(attempts)
        
        # Streaks
        current_streak, best_streak = await self._calculate_streaks(attempts)
        
        return PerformanceMetrics(
            total_attempts=total_attempts,
            correct_attempts=correct_attempts,
            accuracy_rate=accuracy_rate,
            average_score=average_score,
            average_time_per_question=average_time,
            subject_performance=subject_performance,
            difficulty_performance=difficulty_performance,
            recent_accuracy=recent_accuracy,
            improvement_trend=improvement_trend,
            current_streak=current_streak,
            best_streak=best_streak
        )
    
    async def get_error_analysis(
        self, 
        db: AsyncSession, 
        user_id: str,
        subject: Optional[Subject] = None
    ) -> List[ErrorAnalysisResult]:
        """Get detailed error analysis for a user"""
        
        query = select(ErrorPattern).where(ErrorPattern.user_id == user_id)
        
        if subject:
            query = query.where(ErrorPattern.subject == subject)
        
        query = query.order_by(desc(ErrorPattern.error_count))
        
        result = await db.execute(query)
        error_patterns = result.scalars().all()
        
        analysis_results = []
        
        for pattern in error_patterns:
            # Generate practice recommendations
            recommendations = await self._generate_practice_recommendations(pattern)
            
            # Find related topics
            related_topics = await self._find_related_topics(db, pattern)
            
            analysis_results.append(ErrorAnalysisResult(
                error_type=pattern.error_type,
                description=self._get_error_description(pattern.error_type),
                frequency=pattern.error_count,
                last_occurrence=pattern.last_occurrence.isoformat(),
                subject=pattern.subject,
                topic_category=pattern.topic_category,
                difficulty_level=pattern.difficulty_level,
                practice_recommendations=recommendations,
                related_topics=related_topics
            ))
        
        return analysis_results
    
    async def get_detailed_error_analysis(
        self, 
        db: AsyncSession, 
        user_id: str,
        subject: Subject
    ) -> DetailedErrorAnalysis:
        """Get detailed error analysis with subject-specific breakdowns"""
        
        if subject == Subject.MATH:
            return await self._get_math_error_analysis(db, user_id)
        elif subject == Subject.ENGLISH:
            return await self._get_english_error_analysis(db, user_id)
        else:
            return DetailedErrorAnalysis()
    
    async def recommend_level_adjustment(
        self, 
        db: AsyncSession, 
        user: User,
        subject: Subject
    ) -> Optional[LevelAdjustmentRecommendation]:
        """Recommend level adjustment based on performance"""
        
        current_level = user.current_math_level if subject == Subject.MATH else user.current_english_level
        
        # Get recent performance
        metrics = await self.get_user_performance_metrics(db, str(user.id), subject, days=14)
        
        # Decision thresholds
        PROMOTION_ACCURACY = 85.0
        DEMOTION_ACCURACY = 60.0
        CONSISTENCY_THRESHOLD = 0.8
        
        if metrics.total_attempts < 5:
            return None  # Not enough data
        
        # Calculate consistency (how stable the performance is)
        consistency = await self._calculate_performance_consistency(db, str(user.id), subject)
        
        recommendation = None
        
        if (metrics.accuracy_rate >= PROMOTION_ACCURACY and 
            consistency >= CONSISTENCY_THRESHOLD and 
            current_level < 5):
            
            recommendation = LevelAdjustmentRecommendation(
                current_level=current_level,
                recommended_level=current_level + 1,
                reason=f"Excellent performance ({metrics.accuracy_rate:.1f}% accuracy) with consistent results",
                confidence=min(0.9, consistency),
                supporting_evidence=[
                    f"Accuracy rate: {metrics.accuracy_rate:.1f}%",
                    f"Consistency score: {consistency:.2f}",
                    f"Total attempts: {metrics.total_attempts}"
                ],
                accuracy_threshold=PROMOTION_ACCURACY,
                consistency_threshold=CONSISTENCY_THRESHOLD
            )
        
        elif (metrics.accuracy_rate <= DEMOTION_ACCURACY and 
              current_level > 1):
            
            recommendation = LevelAdjustmentRecommendation(
                current_level=current_level,
                recommended_level=current_level - 1,
                reason=f"Performance below threshold ({metrics.accuracy_rate:.1f}% accuracy)",
                confidence=0.7,
                supporting_evidence=[
                    f"Accuracy rate: {metrics.accuracy_rate:.1f}%",
                    f"Below threshold of {DEMOTION_ACCURACY}%",
                    f"Total attempts: {metrics.total_attempts}"
                ],
                accuracy_threshold=DEMOTION_ACCURACY,
                consistency_threshold=CONSISTENCY_THRESHOLD
            )
        
        return recommendation
    
    async def generate_personalized_feedback(
        self, 
        db: AsyncSession, 
        user: User,
        evaluation: AnswerEvaluation,
        question: Question
    ) -> FeedbackGeneration:
        """Generate personalized feedback based on user profile and performance"""
        
        feedback = FeedbackGeneration()
        
        # Positive feedback
        if evaluation.is_correct:
            feedback.positive_feedback.extend([
                "Excellent work! You got it right.",
                "Great job understanding the concept.",
                "Your answer shows good comprehension."
            ])
        elif evaluation.score > 70:
            feedback.positive_feedback.extend([
                "Good effort! You're on the right track.",
                "You show understanding of the main concept."
            ])
        
        # Constructive feedback
        if not evaluation.is_correct:
            feedback.constructive_feedback.extend([
                "Let's review this concept together.",
                "This is a common area where students need practice."
            ])
        
        # Learning style adaptations
        if user.learning_style.value == "visual":
            feedback.learning_style_adaptations.extend([
                "Try drawing a diagram to visualize the problem.",
                "Use colors to highlight key information."
            ])
        elif user.learning_style.value == "auditory":
            feedback.learning_style_adaptations.extend([
                "Try reading the question aloud.",
                "Explain your thinking process out loud."
            ])
        elif user.learning_style.value == "kinesthetic":
            feedback.learning_style_adaptations.extend([
                "Try working through this step by step.",
                "Use physical objects to represent the problem."
            ])
        
        # Encouragement
        feedback.encouragement.extend([
            "Keep practicing - you're improving!",
            "Every mistake is a learning opportunity.",
            "You're building important skills."
        ])
        
        # Motivational elements
        recent_performance = await self.get_user_performance_metrics(db, str(user.id), question.subject, days=7)
        if recent_performance.improvement_trend == "improving":
            feedback.motivational_elements.append("Your recent performance shows great improvement!")
        
        return feedback
    
    # Private helper methods
    async def _get_question_details(self, db: AsyncSession, question_id: str) -> Optional[Question]:
        """Get question details from database"""
        result = await db.execute(select(Question).where(Question.id == question_id))
        return result.scalar_one_or_none()
    
    async def _validate_answer_format(self, answer: str, question_type: QuestionType) -> AnswerValidation:
        """Validate answer format based on question type"""
        
        errors = []
        normalized_answer = answer.strip()
        
        if question_type == QuestionType.MULTIPLE_CHOICE:
            # Check if answer is a valid option (A, B, C, D or full text)
            if not re.match(r'^[A-Da-d]$', normalized_answer) and len(normalized_answer) > 100:
                errors.append("Multiple choice answer should be A, B, C, or D")
            else:
                normalized_answer = normalized_answer.upper()
        
        elif question_type == QuestionType.TRUE_FALSE:
            # Normalize true/false answers
            lower_answer = normalized_answer.lower()
            if lower_answer in ['true', 't', 'yes', 'y', '1']:
                normalized_answer = 'True'
            elif lower_answer in ['false', 'f', 'no', 'n', '0']:
                normalized_answer = 'False'
            else:
                errors.append("True/False answer should be True or False")
        
        elif question_type == QuestionType.FILL_BLANK:
            # Basic validation for fill in the blank
            if len(normalized_answer) == 0:
                errors.append("Answer cannot be empty")
            elif len(normalized_answer) > 200:
                errors.append("Answer is too long for fill in the blank")
        
        elif question_type == QuestionType.OPEN_ENDED:
            # Basic validation for open ended
            if len(normalized_answer) < 5:
                errors.append("Answer is too short for an open-ended question")
            elif len(normalized_answer) > 1000:
                errors.append("Answer is too long")
        
        return AnswerValidation(
            is_valid_format=len(errors) == 0,
            validation_errors=errors,
            normalized_answer=normalized_answer if len(errors) == 0 else None,
            confidence_score=1.0 if len(errors) == 0 else 0.0
        )
    
    async def _evaluate_with_llm(self, request: AnswerEvaluationRequest, student_answer: str) -> AnswerEvaluation:
        """Evaluate answer using MCP service with fallbacks"""

        evaluation_data = None
        try:
            evaluation_data = await mcp_service.evaluate_student_answer(
                question_content=request.question_content,
                correct_answer=request.correct_answer,
                student_answer=student_answer,
                subject=request.subject.value,
                question_type=request.question_type.value,
                difficulty_level=request.difficulty_level or 3,
            )
        except Exception as mcp_error:
            print(f"MCP evaluation error: {mcp_error}")
            try:
                llm_result = await llm_gateway.evaluate_student_answer(
                    question_content=request.question_content,
                    correct_answer=request.correct_answer,
                    student_answer=student_answer,
                    subject=request.subject.value,
                    question_type=request.question_type.value,
                )
                
                if llm_result.get("success"):
                    evaluation_data = llm_result.get("content", {})
                else:
                    return await self._evaluate_with_rules(request, student_answer)
            except Exception as e:
                print(f"LLM evaluation error: {e}")
                return await self._evaluate_with_rules(request, student_answer)
                
        except Exception as e:
            print(f"LLM evaluation error: {e}")
            # Fallback to rule-based evaluation
            return await self._evaluate_with_rules(request, student_answer)

        return AnswerEvaluation(
            is_correct=evaluation_data.get("is_correct", False),
            score=evaluation_data.get("score", 0.0),
            feedback=evaluation_data.get("feedback", "No feedback available"),
            detailed_feedback=evaluation_data.get("detailed_feedback"),
            explanation=evaluation_data.get("explanation"),
            error_analysis=evaluation_data.get("error_analysis", {}),
            recommendations=evaluation_data.get("recommendations", []),
            next_difficulty=evaluation_data.get("next_difficulty"),
        )
    
    async def _evaluate_with_rules(self, request: AnswerEvaluationRequest, student_answer: str) -> AnswerEvaluation:
        """Evaluate answer using rule-based logic"""
        
        correct_answer = request.correct_answer or ""
        
        # Basic correctness check
        is_correct = self._check_answer_correctness(student_answer, correct_answer, request.question_type)
        
        # Calculate score
        score = 100.0 if is_correct else 0.0
        
        # Generate basic feedback
        if is_correct:
            feedback = "Correct! Well done."
        else:
            feedback = f"Incorrect. The correct answer is: {correct_answer}"
        
        # Basic error analysis
        error_analysis = {}
        if not is_correct:
            if request.subject == Subject.MATH:
                error_analysis["calculation_errors"] = ["Incorrect calculation or method"]
            elif request.subject == Subject.ENGLISH:
                error_analysis["grammar_errors"] = ["Grammar or vocabulary error"]
        
        return AnswerEvaluation(
            is_correct=is_correct,
            score=score,
            feedback=feedback,
            error_analysis=error_analysis,
            recommendations=["Review the concept and try similar problems"] if not is_correct else []
        )
    
    def _check_answer_correctness(self, student_answer: str, correct_answer: str, question_type: QuestionType) -> bool:
        """Check if student answer matches correct answer"""
        
        student_clean = student_answer.strip().lower()
        correct_clean = correct_answer.strip().lower()
        
        if question_type == QuestionType.MULTIPLE_CHOICE:
            # For multiple choice, check both letter and full text
            if len(student_clean) == 1:
                # Letter answer (A, B, C, D)
                return student_clean == correct_clean
            else:
                # Full text answer
                return student_clean in correct_clean or correct_clean in student_clean
        
        elif question_type == QuestionType.TRUE_FALSE:
            return student_clean == correct_clean
        
        elif question_type in [QuestionType.FILL_BLANK, QuestionType.OPEN_ENDED]:
            # More flexible matching for text answers
            return (student_clean == correct_clean or 
                    student_clean in correct_clean or 
                    correct_clean in student_clean)
        
        return student_clean == correct_clean
    
    async def _enhance_evaluation(
        self, 
        db: AsyncSession, 
        user: User, 
        question: Question, 
        evaluation: AnswerEvaluation,
        student_answer: str
    ) -> AnswerEvaluation:
        """Enhance evaluation with additional analysis"""
        
        # Add performance context
        recent_performance = await self.get_user_performance_metrics(db, str(user.id), question.subject, days=7)
        evaluation.accuracy_rate = recent_performance.accuracy_rate
        
        # Add improvement suggestions based on error patterns
        if not evaluation.is_correct:
            error_patterns = await self.get_error_analysis(db, str(user.id), question.subject)
            if error_patterns:
                evaluation.improvement_suggestions.extend([
                    f"Focus on {pattern.error_type}" for pattern in error_patterns[:2]
                ])
        
        return evaluation
    
    async def _save_student_attempt(
        self, 
        db: AsyncSession, 
        user: User, 
        question: Question, 
        evaluation: AnswerEvaluation,
        student_answer: str,
        time_spent: Optional[int]
    ) -> StudentAttempt:
        """Save student attempt to database"""
        
        attempt = StudentAttempt(
            id=uuid.uuid4(),
            user_id=user.id,
            question_id=question.id,
            student_answer=student_answer,
            is_correct=evaluation.is_correct,
            time_spent=time_spent,
            error_category=evaluation.error_category,
            grammar_errors=evaluation.error_analysis.get("grammar_errors"),
            vocabulary_errors=evaluation.error_analysis.get("vocabulary_errors")
        )
        
        db.add(attempt)
        await db.commit()
        await db.refresh(attempt)
        
        # Save math error details if applicable
        if question.subject == Subject.MATH and not evaluation.is_correct:
            await self._save_math_error_details(db, attempt, evaluation)
        
        return attempt
    
    async def _save_math_error_details(self, db: AsyncSession, attempt: StudentAttempt, evaluation: AnswerEvaluation):
        """Save detailed math error analysis"""
        
        math_errors = evaluation.error_analysis.get("math_errors", {})
        
        if math_errors:
            math_error_detail = MathErrorDetail(
                id=uuid.uuid4(),
                attempt_id=attempt.id,
                operation=math_errors.get("operation"),
                math_concept=math_errors.get("concept"),
                error_step=math_errors.get("step")
            )
            
            db.add(math_error_detail)
            await db.commit()
    
    async def _update_error_patterns(self, db: AsyncSession, user: User, question: Question, evaluation: AnswerEvaluation):
        """Update error patterns based on evaluation"""
        
        if evaluation.is_correct:
            return  # No errors to track
        
        error_type = evaluation.error_category or "general_error"
        
        # Check if error pattern exists
        result = await db.execute(
            select(ErrorPattern).where(
                and_(
                    ErrorPattern.user_id == user.id,
                    ErrorPattern.subject == question.subject,
                    ErrorPattern.error_type == error_type
                )
            )
        )
        
        error_pattern = result.scalar_one_or_none()
        
        if error_pattern:
            # Update existing pattern
            error_pattern.error_count += 1
            error_pattern.last_occurrence = datetime.utcnow()
        else:
            # Create new pattern
            error_pattern = ErrorPattern(
                id=uuid.uuid4(),
                user_id=user.id,
                subject=question.subject,
                error_type=error_type,
                error_count=1,
                topic_category=question.topic_category,
                difficulty_level=question.difficulty_level
            )
            db.add(error_pattern)
        
        await db.commit()
    
    async def _get_latest_attempt(self, db: AsyncSession, user_id: str, question_id: str) -> StudentAttempt:
        """Get the latest attempt for a user and question"""
        
        result = await db.execute(
            select(StudentAttempt).where(
                and_(
                    StudentAttempt.user_id == user_id,
                    StudentAttempt.question_id == question_id
                )
            ).order_by(desc(StudentAttempt.attempt_date)).limit(1)
        )
        
        return result.scalar_one()
    
    # Additional helper methods for metrics calculation
    async def _calculate_subject_performance(self, db: AsyncSession, user_id: str, attempts: List[StudentAttempt]) -> Dict[str, Dict[str, Any]]:
        """Calculate performance by subject"""
        # Implementation would group attempts by subject and calculate metrics
        return {}
    
    async def _calculate_difficulty_performance(self, db: AsyncSession, user_id: str, attempts: List[StudentAttempt]) -> Dict[str, Dict[str, Any]]:
        """Calculate performance by difficulty level"""
        # Implementation would group attempts by difficulty and calculate metrics
        return {}
    
    async def _calculate_recent_accuracy(self, attempts: List[StudentAttempt], days: int = 7) -> Optional[float]:
        """Calculate accuracy for recent attempts"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        recent_attempts = [a for a in attempts if a.attempt_date >= cutoff]
        
        if not recent_attempts:
            return None
        
        correct = sum(1 for a in recent_attempts if a.is_correct)
        return (correct / len(recent_attempts)) * 100
    
    async def _calculate_improvement_trend(self, attempts: List[StudentAttempt]) -> Optional[str]:
        """Calculate if performance is improving, declining, or stable"""
        if len(attempts) < 10:
            return None
        
        # Simple trend calculation based on recent vs older performance
        mid_point = len(attempts) // 2
        older_attempts = attempts[:mid_point]
        recent_attempts = attempts[mid_point:]
        
        older_accuracy = sum(1 for a in older_attempts if a.is_correct) / len(older_attempts)
        recent_accuracy = sum(1 for a in recent_attempts if a.is_correct) / len(recent_attempts)
        
        diff = recent_accuracy - older_accuracy
        
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        else:
            return "stable"
    
    async def _calculate_streaks(self, attempts: List[StudentAttempt]) -> Tuple[int, int]:
        """Calculate current and best streaks"""
        if not attempts:
            return 0, 0
        
        # Sort by date
        sorted_attempts = sorted(attempts, key=lambda x: x.attempt_date)
        
        current_streak = 0
        best_streak = 0
        temp_streak = 0
        
        for attempt in reversed(sorted_attempts):  # Start from most recent
            if attempt.is_correct:
                temp_streak += 1
                if temp_streak > best_streak:
                    best_streak = temp_streak
            else:
                if current_streak == 0:  # First incorrect from the end
                    current_streak = temp_streak
                temp_streak = 0
        
        # If all recent attempts are correct
        if current_streak == 0:
            current_streak = temp_streak
        
        return current_streak, best_streak
    
    async def _calculate_performance_consistency(self, db: AsyncSession, user_id: str, subject: Subject) -> float:
        """Calculate how consistent the user's performance is"""
        # Simplified consistency calculation
        # In a real implementation, this would analyze variance in performance
        return 0.8  # Placeholder
    
    async def _generate_practice_recommendations(self, error_pattern: ErrorPattern) -> List[str]:
        """Generate practice recommendations based on error pattern"""
        
        recommendations = []
        
        if error_pattern.subject == Subject.MATH:
            if "addition" in error_pattern.error_type:
                recommendations.append("Practice basic addition problems")
            elif "multiplication" in error_pattern.error_type:
                recommendations.append("Review multiplication tables")
            else:
                recommendations.append(f"Focus on {error_pattern.error_type} problems")
        
        elif error_pattern.subject == Subject.ENGLISH:
            if "past_tense" in error_pattern.error_type:
                recommendations.append("Practice past tense verb forms")
            elif "grammar" in error_pattern.error_type:
                recommendations.append("Review grammar rules")
            else:
                recommendations.append(f"Study {error_pattern.error_type}")
        
        return recommendations
    
    async def _find_related_topics(self, db: AsyncSession, error_pattern: ErrorPattern) -> List[str]:
        """Find topics related to the error pattern"""
        # Simplified implementation
        return [error_pattern.topic_category] if error_pattern.topic_category else []
    
    def _get_error_description(self, error_type: str) -> str:
        """Get human-readable description of error type"""
        
        descriptions = {
            "past_tense_error": "Difficulty with past tense verb forms",
            "grammar_error": "General grammar mistakes",
            "vocabulary_error": "Incorrect word usage or meaning",
            "calculation_error": "Mathematical calculation mistakes",
            "concept_error": "Misunderstanding of core concepts"
        }
        
        return descriptions.get(error_type, f"Error type: {error_type}")
    
    async def _get_math_error_analysis(self, db: AsyncSession, user_id: str) -> DetailedErrorAnalysis:
        """Get detailed math error analysis"""
        
        # Get math error details
        result = await db.execute(
            select(MathErrorDetail).join(StudentAttempt).where(
                StudentAttempt.user_id == user_id
            )
        )
        
        math_error_details = result.scalars().all()
        
        # Analyze math errors
        math_errors = {}
        for detail in math_error_details:
            operation = detail.operation or "unknown"
            if operation not in math_errors:
                math_errors[operation] = 0
            math_errors[operation] += 1
        
        return DetailedErrorAnalysis(
            math_errors=math_errors,
            error_frequency=math_errors,
            most_common_errors=sorted(math_errors.keys(), key=lambda x: math_errors[x], reverse=True)[:3]
        )
    
    async def _get_english_error_analysis(self, db: AsyncSession, user_id: str) -> DetailedErrorAnalysis:
        """Get detailed English error analysis"""
        
        # Get English attempts with error details
        result = await db.execute(
            select(StudentAttempt).where(
                and_(
                    StudentAttempt.user_id == user_id,
                    StudentAttempt.grammar_errors.isnot(None)
                )
            )
        )
        
        attempts = result.scalars().all()
        
        grammar_errors = []
        vocabulary_errors = []
        
        for attempt in attempts:
            if attempt.grammar_errors:
                grammar_errors.extend(attempt.grammar_errors)
            if attempt.vocabulary_errors:
                vocabulary_errors.extend(attempt.vocabulary_errors)
        
        return DetailedErrorAnalysis(
            grammar_errors=grammar_errors,
            vocabulary_errors=vocabulary_errors
        )


# Global answer evaluation service instance
answer_evaluation_service = AnswerEvaluationService()