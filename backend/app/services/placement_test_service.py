import asyncio
import logging
import random
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from datetime import datetime
import json

from app.models.question import Question, Subject, QuestionType, SourceType
from app.models.student_attempt import StudentAttempt
from app.models.math_profile import MathProfile
from app.services.question_generator import question_generator
from app.services.embedding_service import embedding_service
from app.services.metadata_schema_service import metadata_schema_service, ContentType, Domain
from app.core.config import settings

logger = logging.getLogger(__name__)


class PlacementTestService:
    """Dynamic placement test service with adaptive question generation"""
    
    def __init__(self):
        self.test_configs = {
            Subject.MATH: {
                "initial_difficulty": 2.5,
                "difficulty_step": 0.5,
                "max_questions": 15,
                "min_questions": 8,
                "confidence_threshold": 0.8,
                "topics": ["arithmetic", "algebra", "geometry", "basic_calculus"],
                "question_types": ["multiple_choice", "fill_blank", "open_ended"]
            },
            Subject.ENGLISH: {
                "initial_difficulty": 2.5,
                "difficulty_step": 0.5,
                "max_questions": 12,
                "min_questions": 6,
                "confidence_threshold": 0.8,
                "topics": ["grammar", "vocabulary", "reading_comprehension", "writing"],
                "question_types": ["multiple_choice", "cloze", "open_ended"]
            }
        }
        
        # Adaptive test parameters
        self.adaptive_params = {
            "correct_answer_bonus": 0.3,
            "incorrect_answer_penalty": 0.2,
            "time_bonus_threshold": 0.7,  # %70'den hızlı cevaplar
            "time_penalty_threshold": 1.5,  # %150'den yavaş cevaplar
            "confidence_increase": 0.1,
            "confidence_decrease": 0.15
        }
    
    async def create_placement_test(
        self,
        session: AsyncSession,
        user_id: str,
        subject: Subject,
        test_type: str = "adaptive"
    ) -> Dict[str, Any]:
        """Create a new placement test for user"""
        
        try:
            logger.info(f"🚀 Creating placement test for user {user_id}, subject: {subject.value}")
            
            # Test configuration
            config = self.test_configs.get(subject, self.test_configs[Subject.MATH])
            
            # Generate initial questions
            initial_questions = await self._generate_initial_questions(
                session, subject, config["initial_difficulty"], 3
            )
            
            # Create test session
            test_session = {
                "test_id": f"placement_{subject.value}_{user_id}_{datetime.utcnow().timestamp()}",
                "user_id": user_id,
                "subject": subject.value,
                "test_type": test_type,
                "status": "active",
                "created_at": datetime.utcnow().isoformat(),
                "current_question_index": 0,
                "total_questions": len(initial_questions),
                "questions": initial_questions,
                "answers": [],
                "current_difficulty": config["initial_difficulty"],
                "confidence_score": 0.0,
                "estimated_level": None,
                "config": config
            }
            
            # Store test session in cache/database
            await self._store_test_session(test_session)
            
            logger.info(f"✅ Placement test created: {test_session['test_id']}")
            
            return {
                "success": True,
                "test_id": test_session["test_id"],
                "questions": initial_questions[:3],  # İlk 3 soru
                "total_questions": test_session["total_questions"],
                "current_difficulty": test_session["current_difficulty"],
                "estimated_level": test_session["estimated_level"]
            }
            
        except Exception as e:
            logger.error(f"❌ Error creating placement test: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _generate_initial_questions(
        self,
        session: AsyncSession,
        subject: Subject,
        difficulty: float,
        count: int
    ) -> List[Dict[str, Any]]:
        """Generate initial questions for placement test"""
        
        try:
            questions = []
            
            # Try to get existing questions from database
            existing_questions = await self._get_questions_by_difficulty_and_topic(
                session, subject, difficulty, count * 2
            )
            
            if existing_questions:
                # Select random questions from existing pool
                selected_questions = random.sample(existing_questions, min(count, len(existing_questions)))
                
                for question in selected_questions:
                    questions.append({
                        "id": str(question.id),
                        "content": question.content,
                        "question_type": question.question_type.value,
                        "difficulty_level": question.difficulty_level,
                        "topic_category": question.topic_category,
                        "options": question.options if question.options else [],
                        "correct_answer": question.correct_answer,
                        "source": "database"
                    })
            
            # If not enough questions, generate new ones
            if len(questions) < count:
                remaining_count = count - len(questions)
                
                # Generate questions using question generator
                generated_questions = await self._generate_questions_with_ai(
                    subject, difficulty, remaining_count
                )
                
                questions.extend(generated_questions)
            
            return questions[:count]
            
        except Exception as e:
            logger.error(f"❌ Error generating initial questions: {e}")
            return []
    
    async def _get_questions_by_difficulty_and_topic(
        self,
        session: AsyncSession,
        subject: Subject,
        difficulty: float,
        limit: int
    ) -> List[Question]:
        """Get questions from database by difficulty and topic"""
        
        try:
            # Difficulty range
            difficulty_low = max(0.0, difficulty - 0.5)
            difficulty_high = min(5.0, difficulty + 0.5)
            
            # Query questions
            stmt = select(Question).where(
                and_(
                    Question.subject == subject,
                    Question.difficulty_level >= difficulty_low,
                    Question.difficulty_level <= difficulty_high,
                    Question.is_active == True
                )
            ).limit(limit)
            
            result = await session.execute(stmt)
            questions = result.scalars().all()
            
            return questions
            
        except Exception as e:
            logger.error(f"❌ Error getting questions from database: {e}")
            return []
    
    async def _generate_questions_with_ai(
        self,
        subject: Subject,
        difficulty: float,
        count: int
    ) -> List[Dict[str, Any]]:
        """Generate questions using AI question generator"""
        
        try:
            questions = []
            
            for i in range(count):
                # Generate question using question generator
                generated_question = await question_generator.generate_question(
                    subject=subject.value,
                    difficulty_level=difficulty,
                    question_type="multiple_choice",
                    topic_category="placement_test"
                )
                
                if generated_question:
                    questions.append({
                        "id": f"generated_{subject.value}_{i}_{datetime.utcnow().timestamp()}",
                        "content": generated_question.get("content", ""),
                        "question_type": generated_question.get("question_type", "multiple_choice"),
                        "difficulty_level": generated_question.get("difficulty_level", difficulty),
                        "topic_category": generated_question.get("topic_category", "placement_test"),
                        "options": generated_question.get("options", []),
                        "correct_answer": generated_question.get("correct_answer", ""),
                        "source": "ai_generated"
                    })
            
            return questions
            
        except Exception as e:
            logger.error(f"❌ Error generating questions with AI: {e}")
            return []
    
    async def submit_answer(
        self,
        session: AsyncSession,
        test_id: str,
        question_id: str,
        user_answer: str,
        time_taken: float,
        confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """Submit answer for placement test question"""
        
        try:
            logger.info(f"📝 Submitting answer for test {test_id}, question {question_id}")
            
            # Get test session
            test_session = await self._get_test_session(test_id)
            if not test_session:
                raise ValueError(f"Test session {test_id} not found")
            
            # Get question details
            question = self._find_question_in_test(test_session, question_id)
            if not question:
                raise ValueError(f"Question {question_id} not found in test {test_id}")
            
            # Evaluate answer
            answer_evaluation = self._evaluate_answer(question, user_answer, time_taken)
            
            # Store answer
            answer_data = {
                "question_id": question_id,
                "user_answer": user_answer,
                "correct_answer": question.get("correct_answer", ""),
                "is_correct": answer_evaluation["is_correct"],
                "time_taken": time_taken,
                "confidence": confidence,
                "score": answer_evaluation["score"],
                "submitted_at": datetime.utcnow().isoformat()
            }
            
            test_session["answers"].append(answer_data)
            
            # Update test session
            await self._update_test_session(test_session)
            
            # Store in database
            await self._store_student_answer(session, test_session, answer_data)
            
            # Check if test should continue
            should_continue = await self._should_continue_test(test_session)
            
            if should_continue:
                # Generate next question
                next_question = await self._generate_next_question(test_session)
                if next_question:
                    test_session["questions"].append(next_question)
                    test_session["total_questions"] += 1
                    await self._update_test_session(test_session)
            
            return {
                "success": True,
                "answer_evaluation": answer_evaluation,
                "should_continue": should_continue,
                "next_question": next_question if should_continue else None,
                "current_difficulty": test_session["current_difficulty"],
                "confidence_score": test_session["confidence_score"]
            }
            
        except Exception as e:
            logger.error(f"❌ Error submitting answer: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def _evaluate_answer(
        self,
        question: Dict[str, Any],
        user_answer: str,
        time_taken: float
    ) -> Dict[str, Any]:
        """Evaluate user answer and calculate score"""
        
        try:
            correct_answer = question.get("correct_answer", "")
            is_correct = user_answer.strip().lower() == correct_answer.strip().lower()
            
            # Base score
            base_score = 1.0 if is_correct else 0.0
            
            # Time bonus/penalty
            time_score = 0.0
            if is_correct:
                if time_taken < 30:  # Hızlı doğru cevap
                    time_score = 0.1
                elif time_taken > 120:  # Yavaş doğru cevap
                    time_score = -0.05
            else:
                if time_taken < 15:  # Çok hızlı yanlış cevap
                    time_score = -0.1
                elif time_taken > 180:  # Çok yavaş yanlış cevap
                    time_score = -0.05
            
            # Total score
            total_score = max(0.0, min(1.0, base_score + time_score))
            
            return {
                "is_correct": is_correct,
                "score": total_score,
                "time_score": time_score,
                "feedback": self._generate_feedback(is_correct, time_taken)
            }
            
        except Exception as e:
            logger.error(f"❌ Error evaluating answer: {e}")
            return {
                "is_correct": False,
                "score": 0.0,
                "time_score": 0.0,
                "feedback": "Error evaluating answer"
            }
    
    def _generate_feedback(self, is_correct: bool, time_taken: float) -> str:
        """Generate feedback based on answer correctness and time"""
        
        if is_correct:
            if time_taken < 30:
                return "Excellent! Quick and correct answer."
            elif time_taken < 60:
                return "Great job! Correct answer."
            else:
                return "Good! Correct answer, but try to be faster next time."
        else:
            if time_taken < 15:
                return "Take your time to read the question carefully."
            elif time_taken > 180:
                return "Don't spend too long on one question. Move on if you're stuck."
            else:
                return "Keep trying! Review the topic and try again."
    
    async def _should_continue_test(self, test_session: Dict[str, Any]) -> bool:
        """Determine if test should continue based on confidence and question count"""
        
        try:
            config = test_session["config"]
            answers = test_session["answers"]
            
            # Check minimum questions
            if len(answers) < config["min_questions"]:
                return True
            
            # Check maximum questions
            if len(answers) >= config["max_questions"]:
                return False
            
            # Check confidence threshold
            if test_session["confidence_score"] >= config["confidence_threshold"]:
                return False
            
            # Check if we have enough data for reliable estimation
            if len(answers) >= 5:
                recent_answers = answers[-5:]
                recent_accuracy = sum(1 for a in recent_answers if a["is_correct"]) / len(recent_answers)
                
                if recent_accuracy > 0.8:
                    return False  # High recent accuracy, can stop
                elif recent_accuracy < 0.2:
                    return False  # Very low recent accuracy, can stop
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error determining if test should continue: {e}")
            return False
    
    async def _generate_next_question(self, test_session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate next question based on adaptive algorithm"""
        
        try:
            answers = test_session["answers"]
            current_difficulty = test_session["current_difficulty"]
            config = test_session["config"]
            
            # Calculate new difficulty based on recent performance
            if len(answers) >= 2:
                recent_answers = answers[-2:]
                recent_accuracy = sum(1 for a in recent_answers if a["is_correct"]) / len(recent_answers)
                
                if recent_accuracy > 0.7:
                    # Increase difficulty
                    new_difficulty = min(5.0, current_difficulty + config["difficulty_step"])
                elif recent_accuracy < 0.3:
                    # Decrease difficulty
                    new_difficulty = max(0.0, current_difficulty - config["difficulty_step"])
                else:
                    # Keep current difficulty
                    new_difficulty = current_difficulty
                
                test_session["current_difficulty"] = new_difficulty
            
            # Generate question with new difficulty
            next_question = await self._generate_questions_with_ai(
                Subject(test_session["subject"]),
                test_session["current_difficulty"],
                1
            )
            
            return next_question[0] if next_question else None
            
        except Exception as e:
            logger.error(f"❌ Error generating next question: {e}")
            return None
    
    async def complete_test(
        self,
        session: AsyncSession,
        test_id: str
    ) -> Dict[str, Any]:
        """Complete placement test and calculate final results"""
        
        try:
            logger.info(f"🏁 Completing placement test {test_id}")
            
            # Get test session
            test_session = await self._get_test_session(test_id)
            if not test_session:
                raise ValueError(f"Test session {test_id} not found")
            
            # Calculate final results
            final_results = self._calculate_final_results(test_session)
            
            # Update test session
            test_session["status"] = "completed"
            test_session["completed_at"] = datetime.utcnow().isoformat()
            test_session["final_results"] = final_results
            await self._update_test_session(test_session)
            
            # Store results in database
            await self._store_test_results(session, test_session, final_results)
            
            logger.info(f"✅ Placement test completed: {test_id}")
            
            return {
                "success": True,
                "test_id": test_id,
                "final_results": final_results,
                "recommendations": self._generate_recommendations(final_results)
            }
            
        except Exception as e:
            logger.error(f"❌ Error completing test: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def _calculate_final_results(self, test_session: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate final test results and estimated level"""
        
        try:
            answers = test_session["answers"]
            questions = test_session["questions"]
            
            if not answers:
                return {
                    "total_questions": 0,
                    "correct_answers": 0,
                    "accuracy": 0.0,
                    "average_time": 0.0,
                    "estimated_level": 1.0,
                    "confidence_score": 0.0
                }
            
            # Basic statistics
            total_questions = len(answers)
            correct_answers = sum(1 for a in answers if a["is_correct"])
            accuracy = correct_answers / total_questions
            average_time = sum(a["time_taken"] for a in answers) / total_questions
            
            # Calculate difficulty-weighted score
            difficulty_scores = []
            for answer in answers:
                question = self._find_question_in_test(test_session, answer["question_id"])
                if question:
                    difficulty = question.get("difficulty_level", 2.5)
                    score = answer["score"] * difficulty
                    difficulty_scores.append(score)
            
            weighted_score = sum(difficulty_scores) / len(difficulty_scores) if difficulty_scores else 0.0
            
            # Estimate level based on performance
            estimated_level = self._estimate_level_from_performance(accuracy, weighted_score, test_session["current_difficulty"])
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(accuracy, total_questions, weighted_score)
            
            return {
                "total_questions": total_questions,
                "correct_answers": correct_answers,
                "accuracy": accuracy,
                "average_time": average_time,
                "weighted_score": weighted_score,
                "estimated_level": estimated_level,
                "confidence_score": confidence_score,
                "difficulty_progression": [a.get("difficulty", 2.5) for a in answers],
                "topic_breakdown": self._calculate_topic_breakdown(questions, answers)
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating final results: {e}")
            return {}
    
    def _estimate_level_from_performance(
        self,
        accuracy: float,
        weighted_score: float,
        final_difficulty: float
    ) -> float:
        """Estimate student level from test performance"""
        
        try:
            # Base level from final difficulty
            base_level = final_difficulty
            
            # Adjust based on accuracy
            if accuracy > 0.8:
                level_adjustment = 0.5
            elif accuracy > 0.6:
                level_adjustment = 0.2
            elif accuracy < 0.4:
                level_adjustment = -0.3
            else:
                level_adjustment = 0.0
            
            # Adjust based on weighted score
            if weighted_score > 3.0:
                score_adjustment = 0.3
            elif weighted_score < 1.5:
                score_adjustment = -0.2
            else:
                score_adjustment = 0.0
            
            # Calculate final level
            estimated_level = base_level + level_adjustment + score_adjustment
            
            # Ensure level is within bounds
            return max(1.0, min(5.0, estimated_level))
            
        except Exception as e:
            logger.error(f"❌ Error estimating level: {e}")
            return 2.5
    
    def _calculate_confidence_score(
        self,
        accuracy: float,
        total_questions: int,
        weighted_score: float
    ) -> float:
        """Calculate confidence score for the test results"""
        
        try:
            # Base confidence from accuracy
            accuracy_confidence = accuracy
            
            # Question count confidence (more questions = higher confidence)
            count_confidence = min(1.0, total_questions / 10.0)
            
            # Performance confidence
            performance_confidence = min(1.0, weighted_score / 4.0)
            
            # Weighted average
            confidence = (accuracy_confidence * 0.5 + count_confidence * 0.3 + performance_confidence * 0.2)
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"❌ Error calculating confidence score: {e}")
            return 0.5
    
    def _calculate_topic_breakdown(
        self,
        questions: List[Dict[str, Any]],
        answers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate performance breakdown by topic"""
        
        try:
            topic_performance = {}
            
            for answer in answers:
                question = self._find_question_in_test({"questions": questions}, answer["question_id"])
                if question:
                    topic = question.get("topic_category", "unknown")
                    
                    if topic not in topic_performance:
                        topic_performance[topic] = {
                            "total": 0,
                            "correct": 0,
                            "accuracy": 0.0
                        }
                    
                    topic_performance[topic]["total"] += 1
                    if answer["is_correct"]:
                        topic_performance[topic]["correct"] += 1
            
            # Calculate accuracy for each topic
            for topic, stats in topic_performance.items():
                stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            
            return topic_performance
            
        except Exception as e:
            logger.error(f"❌ Error calculating topic breakdown: {e}")
            return {}
    
    def _generate_recommendations(self, final_results: Dict[str, Any]) -> List[str]:
        """Generate learning recommendations based on test results"""
        
        try:
            recommendations = []
            accuracy = final_results.get("accuracy", 0.0)
            estimated_level = final_results.get("estimated_level", 2.5)
            
            if accuracy < 0.4:
                recommendations.append("Focus on fundamental concepts and basic skills")
                recommendations.append("Practice with easier questions to build confidence")
            elif accuracy < 0.6:
                recommendations.append("Review core concepts and practice medium difficulty questions")
                recommendations.append("Work on time management and problem-solving strategies")
            elif accuracy < 0.8:
                recommendations.append("Good foundation, focus on challenging topics")
                recommendations.append("Practice advanced problems to improve skills")
            else:
                recommendations.append("Excellent performance! Ready for advanced topics")
                recommendations.append("Consider taking more challenging assessments")
            
            # Level-specific recommendations
            if estimated_level < 2.0:
                recommendations.append("Start with beginner-level materials and gradually progress")
            elif estimated_level > 4.0:
                recommendations.append("Ready for advanced coursework and complex problem-solving")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {e}")
            return ["Review test results and consult with instructor"]
    
    # Helper methods for test session management
    async def _store_test_session(self, test_session: Dict[str, Any]):
        """Store test session in cache/database"""
        # Implementation depends on your storage solution
        pass
    
    async def _get_test_session(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get test session from cache/database"""
        # Implementation depends on your storage solution
        pass
    
    async def _update_test_session(self, test_session: Dict[str, Any]):
        """Update test session in cache/database"""
        # Implementation depends on your storage solution
        pass
    
    def _find_question_in_test(self, test_session: Dict[str, Any], question_id: str) -> Optional[Dict[str, Any]]:
        """Find question in test session"""
        for question in test_session.get("questions", []):
            if question.get("id") == question_id:
                return question
        return None
    
    async def _store_student_answer(self, session: AsyncSession, test_session: Dict[str, Any], answer_data: Dict[str, Any]):
        """Store student answer in database"""
        # Implementation for storing in database
        pass
    
    async def _store_test_results(self, session: AsyncSession, test_session: Dict[str, Any], final_results: Dict[str, Any]):
        """Store final test results in database"""
        # Implementation for storing in database
        pass


# Global instance
placement_test_service = PlacementTestService()
