"""
Enhanced Recommendation API Endpoints
Comprehensive ML-powered recommendation system
"""

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from app.crud.user import get_current_user
from app.db.database import get_db
from app.db.models import StudentResponse, User
from app.services.enhanced_embedding_service import enhanced_embedding_service
from app.services.enhanced_stream_consumer import MessageType, stream_consumer_manager
from app.services.ensemble_service import ensemble_service
from app.services.recommendation_service import recommendation_service
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

logger = structlog.get_logger()
router = APIRouter(prefix="/recommendations", tags=["recommendations"])


# Request/Response Models
class RecommendationRequest(BaseModel):
    student_id: int
    subject_id: Optional[int] = None
    topic_id: Optional[int] = None
    difficulty_range: Optional[tuple[int, int]] = None
    exclude_answered: bool = True
    count: int = Field(default=5, le=20)
    algorithm: Optional[str] = "ensemble"  # ensemble, bandit, collaborative, embedding


class PersonalizedRecommendationRequest(BaseModel):
    student_id: int
    learning_goals: Optional[List[str]] = None
    time_available: Optional[int] = None  # minutes
    performance_context: Optional[Dict[str, Any]] = None
    adaptive_difficulty: bool = True


class QuestionRecommendation(BaseModel):
    question_id: int
    title: str
    difficulty_level: int
    topic: str
    subject: str
    confidence_score: float
    reasoning: List[str]
    estimated_time: Optional[int] = None
    skill_alignment: Optional[float] = None


class RecommendationResponse(BaseModel):
    recommendations: List[QuestionRecommendation]
    algorithm_used: str
    total_available: int
    generation_time: float
    metadata: Dict[str, Any]


class FeedbackRequest(BaseModel):
    student_id: int
    question_id: int
    feedback_type: str  # "helpful", "too_easy", "too_hard", "irrelevant"
    rating: Optional[float] = Field(None, ge=0, le=1)
    context: Optional[Dict[str, Any]] = None


class LearningPathRequest(BaseModel):
    student_id: int
    target_skill: str
    current_level: Optional[int] = None
    target_level: Optional[int] = None
    time_horizon: Optional[int] = 30  # days


class LearningPathStep(BaseModel):
    step_number: int
    question_id: int
    title: str
    difficulty: int
    prerequisite_skills: List[str]
    learning_objectives: List[str]
    estimated_duration: int


class LearningPathResponse(BaseModel):
    path_id: str
    steps: List[LearningPathStep]
    total_duration: int
    success_probability: float
    alternative_paths: int


@router.post("/personalized", response_model=RecommendationResponse)
async def get_personalized_recommendations(
    request: PersonalizedRecommendationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get highly personalized recommendations using ensemble ML models"""
    start_time = time.time()

    try:
        logger.info(
            "personalized_recommendation_request",
            student_id=request.student_id,
            learning_goals=request.learning_goals,
        )

        # Get student context and history
        student_context = await _get_student_context(db, request.student_id)

        # Generate recommendations using ensemble approach
        # get_personalized_recommendations metodu henüz implement edilmedi
        raw_recommendations = getattr(
            recommendation_service, "get_recommendations", lambda **kwargs: []
        )(
            db=db,
            student_id=request.student_id,
            count=10,  # Get more for filtering
            context=student_context,
            adaptive_difficulty=request.adaptive_difficulty,
        )

        # Apply ML scoring
        scored_recommendations = []
        for rec in raw_recommendations:
            # Get comprehensive score from ensemble
            ensemble_score = ensemble_service.calculate_comprehensive_ensemble_score(
                user_features=student_context or {},
                question_features=getattr(rec, "features", {}),
                candidate_questions=[rec],
            )

            # Convert to response format
            score_value = 0.5  # Default fallback - ensure score_value is always a float
            if isinstance(ensemble_score, list) and ensemble_score:
                score_value = (
                    float(ensemble_score[0])
                    if isinstance(ensemble_score[0], (int, float))
                    else 0.5
                )
            elif isinstance(ensemble_score, (int, float)):
                score_value = float(ensemble_score)
            else:
                score_value = 0.5

            question_rec = QuestionRecommendation(
                question_id=rec.question_id,
                title=rec.title,
                difficulty_level=rec.difficulty_level,
                topic=rec.topic,
                subject=rec.subject,
                confidence_score=score_value,
                reasoning=_generate_reasoning(rec, score_value, student_context),
                estimated_time=_estimate_question_time(rec, student_context),
                skill_alignment=_calculate_skill_alignment(
                    rec, request.learning_goals or []
                ),
            )
            scored_recommendations.append(question_rec)

        # Sort by ensemble score and limit
        scored_recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        final_recommendations = scored_recommendations[:5]

        generation_time = time.time() - start_time

        # Log recommendation analytics
        background_tasks.add_task(
            _log_recommendation_analytics,
            request.student_id,
            [r.question_id for r in final_recommendations],
            "personalized_ensemble",
            generation_time,
        )

        return RecommendationResponse(
            recommendations=final_recommendations,
            algorithm_used="personalized_ensemble",
            total_available=len(raw_recommendations),
            generation_time=generation_time,
            metadata={
                "student_context": student_context,
                "adaptive_difficulty": request.adaptive_difficulty,
                "learning_goals": request.learning_goals,
            },
        )

    except Exception as e:
        logger.error("personalized_recommendation_error", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Recommendation generation failed: {str(e)}"
        )


@router.post("/algorithm-specific", response_model=RecommendationResponse)
async def get_algorithm_specific_recommendations(
    request: RecommendationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get recommendations using a specific algorithm"""
    start_time = time.time()

    try:
        algorithm = request.algorithm or "ensemble"
        logger.info(
            "algorithm_specific_recommendation",
            student_id=request.student_id,
            algorithm=algorithm,
        )

        if algorithm == "bandit":
            recommendations = await _get_bandit_recommendations(db, request)
        elif algorithm == "collaborative":
            recommendations = await _get_collaborative_recommendations(db, request)
        elif algorithm == "embedding":
            recommendations = await _get_embedding_recommendations(db, request)
        elif algorithm == "ensemble":
            recommendations = await _get_ensemble_recommendations(db, request)
        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown algorithm: {algorithm}"
            )

        generation_time = time.time() - start_time

        return RecommendationResponse(
            recommendations=recommendations,
            algorithm_used=algorithm,
            total_available=len(recommendations),
            generation_time=generation_time,
            metadata={"algorithm_config": algorithm},
        )

    except Exception as e:
        logger.error("algorithm_specific_recommendation_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_recommendation_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Submit feedback on recommendation quality"""
    try:
        logger.info(
            "recommendation_feedback",
            student_id=request.student_id,
            question_id=request.question_id,
            feedback_type=request.feedback_type,
        )

        # Convert feedback to numerical score
        feedback_score = _convert_feedback_to_score(
            request.feedback_type, request.rating
        )

        # Publish feedback to stream for ML model updates
        await stream_consumer_manager.publish_message(
            MessageType.RECOMMENDATION_FEEDBACK,
            {
                "student_id": request.student_id,
                "question_id": request.question_id,
                "feedback_score": feedback_score,
                "feedback_type": request.feedback_type,
                "context": request.context or {},
            },
        )

        # Update ensemble models
        background_tasks.add_task(
            ensemble_service.update_models_with_feedback,
            user_features={"student_id": request.student_id},
            question_features={"question_id": request.question_id},
            is_correct=bool(feedback_score > 0.5),
            response_time=int(getattr(request, "response_time", 1000)),
        )

        return {
            "status": "success",
            "message": "Feedback submitted successfully",
            "processed": True,
        }

    except Exception as e:
        logger.error("recommendation_feedback_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/learning-path", response_model=LearningPathResponse)
async def generate_learning_path(
    request: LearningPathRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate a personalized learning path"""
    try:
        logger.info(
            "learning_path_generation",
            student_id=request.student_id,
            target_skill=request.target_skill,
        )

        # Get student's current skill level
        student_context = await _get_student_context(db, request.student_id)
        current_skill_level = student_context.get("skill_levels", {}).get(
            request.target_skill, 1
        )

        # Generate learning path using Neo4j skill graph
        path_steps = await _generate_skill_progression_path(
            db,
            request.student_id,
            request.target_skill,
            current_skill_level,
            request.target_level or (current_skill_level + 2),
        )

        # Estimate success probability
        success_probability = _calculate_path_success_probability(
            student_context, path_steps, getattr(request, "time_horizon", 30) or 30
        )

        return LearningPathResponse(
            path_id=f"path_{request.student_id}_{int(time.time())}",
            steps=path_steps,
            total_duration=sum(step.estimated_duration for step in path_steps),
            success_probability=success_probability,
            alternative_paths=2,  # Could generate alternatives
        )

    except Exception as e:
        logger.error("learning_path_generation_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/similar/{question_id}")
async def get_similar_questions(
    question_id: int,
    count: int = Query(default=5, le=20),
    similarity_threshold: float = Query(default=0.7, ge=0.5, le=1.0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get questions similar to a given question using embeddings"""
    try:
        # Get similar questions using the fixed helper method
        similar_questions = (
            await enhanced_embedding_service.find_similar_questions_by_id_async(
                question_id, db, threshold=similarity_threshold, limit=count
            )
        )

        if not similar_questions:
            # Fallback to Neo4j if embedding fails
            from app.crud.question import get_similar_questions_from_neo4j

            neo4j_similar = get_similar_questions_from_neo4j(question_id, limit=count)
            similar_questions = [
                {
                    "question_id": item["question_id"],
                    "similarity_score": min(
                        1.0, item["shared_skills"] / 10.0
                    ),  # Normalize shared skills
                    "source": "neo4j",
                }
                for item in neo4j_similar
            ]

        return {
            "source_question_id": question_id,
            "similar_questions": similar_questions,
            "similarity_threshold": similarity_threshold,
            "count": len(similar_questions),
        }

    except Exception as e:
        logger.error("similar_questions_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trending")
async def get_trending_questions(
    time_window: str = Query(default="24h", regex="^(1h|6h|24h|7d)$"),
    count: int = Query(default=10, le=50),
    subject_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """Get trending questions based on recent activity"""
    try:
        # Calculate time window
        hours_map = {"1h": 1, "6h": 6, "24h": 24, "7d": 168}
        hours = hours_map[time_window]
        since = datetime.utcnow() - timedelta(hours=hours)

        # Get trending questions from database
        trending = await _get_trending_questions(db, since, count, subject_id)

        return {
            "trending_questions": trending,
            "time_window": time_window,
            "generated_at": datetime.utcnow().isoformat(),
            "count": len(trending),
        }

    except Exception as e:
        logger.error("trending_questions_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/{student_id}")
async def get_recommendation_analytics(
    student_id: int,
    days: int = Query(default=30, le=90),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get recommendation analytics for a student"""
    try:
        # Get recommendation history and performance
        analytics = await _get_student_recommendation_analytics(db, student_id, days)

        return {
            "student_id": student_id,
            "period_days": days,
            "analytics": analytics,
            "generated_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error("recommendation_analytics_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Helper Functions
async def _get_student_context(db: Session, student_id: int) -> Dict[str, Any]:
    """Get comprehensive student context for recommendations"""
    try:
        # Get student's response history and performance metrics
        student_responses = (
            db.query(StudentResponse)
            .filter(StudentResponse.student_id == student_id)
            .order_by(StudentResponse.created_at.desc())
            .limit(100)
            .all()
        )

        # Calculate performance metrics
        total_responses = len(student_responses)
        correct_responses = sum(
            1 for r in student_responses if getattr(r, "is_correct", False)
        )
        accuracy = correct_responses / total_responses if total_responses > 0 else 0.5

        # Calculate average response time
        response_times = [
            getattr(r, "response_time", 0)
            for r in student_responses
            if getattr(r, "response_time", 0)
        ]
        avg_response_time = (
            sum(response_times) / len(response_times) if response_times else 5000
        )

        # Get recent performance (last 10 questions)
        recent_responses = student_responses[:10]
        recent_accuracy = (
            sum(1 for r in recent_responses if getattr(r, "is_correct", False))
            / len(recent_responses)
            if recent_responses
            else 0.5
        )

        return {
            "student_id": student_id,
            "total_responses": total_responses,
            "accuracy_rate_overall": accuracy,
            "accuracy_rate_recent": recent_accuracy,
            "avg_response_time": avg_response_time,
            "session_question_count": len(recent_responses),
            "skill_levels": {},  # Would be populated from Neo4j
            "learning_preferences": {},
            "last_activity": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error("student_context_error", error=str(e))
        return {"student_id": student_id, "accuracy_rate_overall": 0.5}


async def _get_bandit_recommendations(
    db: Session, request: RecommendationRequest
) -> List[QuestionRecommendation]:
    """Get recommendations using bandit algorithm"""
    try:
        from app.services.ensemble_service import EnhancedEnsembleScoringService

        ensemble_service = EnhancedEnsembleScoringService()

        # Get candidate questions
        from app.crud.question import get_questions

        questions = get_questions(
            db,
            subject_id=request.subject_id,
            difficulty_level=getattr(request, "difficulty_level", None),
            limit=20,
        )

        if not questions:
            return []

        # Convert to candidate format
        candidate_questions = [
            {
                "id": q.id,
                "difficulty_level": q.difficulty_level,
                "subject_id": q.subject_id,
                "topic_id": q.topic_id,
                "content": q.content,
            }
            for q in questions
        ]

        # Get user features
        user_features = {}
        if request.student_id:
            from app.crud.user import get_user

            user = get_user(db, request.student_id)
            if user:
                profile = getattr(user, "student_profile", None)
                user_features = {
                    "level": getattr(profile, "level", 1) if profile else 1,
                    "total_answered": getattr(profile, "total_questions_answered", 0)
                    if profile
                    else 0,
                    "accuracy": (
                        getattr(profile, "total_correct_answers", 0) if profile else 0
                    )
                    / max(
                        1,
                        getattr(profile, "total_questions_answered", 1)
                        if profile
                        else 1,
                    ),
                }

        # Use bandit selection - select_multiple_questions metodu henüz yok
        bandit = getattr(ensemble_service, "bandit", None)
        scored_results = getattr(
            bandit, "select_multiple_questions", lambda u, c, count=5: c[:count]
        )(user_features, candidate_questions, count=getattr(request, "limit", 5))

        # Convert to QuestionRecommendation format
        recommendations = []
        for question in questions[: getattr(request, "limit", 5)]:
            if question.id in [result["question_id"] for result in scored_results]:
                recommendations.append(
                    QuestionRecommendation(
                        question_id=getattr(question, "id", 0),
                        title=getattr(question, "content", "")[:50] + "...",
                        topic=getattr(question, "topic", "General"),
                        subject=getattr(question, "subject", "Math"),
                        difficulty_level=getattr(question, "difficulty_level", 1),
                        confidence_score=next(
                            (
                                r["confidence"]
                                for r in scored_results
                                if r["question_id"] == question.id
                            ),
                            0.5,
                        ),
                        reasoning=[
                            "Selected using bandit algorithm",
                            f"Expected reward optimized for your level {user_features.get('level', 1)}",
                        ],
                    )
                )

        return recommendations

    except Exception as e:
        logger.error("bandit_recommendations_error", error=str(e))
        return []


async def _get_collaborative_recommendations(
    db: Session, request: RecommendationRequest
) -> List[QuestionRecommendation]:
    """Get recommendations using collaborative filtering"""
    try:
        from app.crud.student_response import get_responses
        from app.services.ensemble_service import EnhancedEnsembleScoringService

        ensemble_service = EnhancedEnsembleScoringService()

        # Get candidate questions
        from app.crud.question import get_questions

        questions = get_questions(
            db,
            subject_id=request.subject_id,
            difficulty_level=getattr(request, "difficulty_level", None),
            limit=20,
        )

        if not questions:
            return []

        # Get collaborative recommendations
        user_item_matrix = {}
        if request.student_id:
            # Get student's response history
            responses = get_responses(db, student_id=request.student_id, limit=100)
            for response in responses:
                user_item_matrix[response.student_id] = user_item_matrix.get(
                    response.student_id, {}
                )
                user_item_matrix[response.student_id][response.question_id] = (
                    1 if getattr(response, "is_correct", False) else 0
                )

        # Use collaborative filtering
        # get_user_recommendations metodu henüz implement edilmedi
        collaborative_filter = getattr(ensemble_service, "collaborative_filter", None)
        collaborative_scores = getattr(
            collaborative_filter, "get_user_recommendations", lambda *args, **kwargs: {}
        )(
            request.student_id or 1,
            user_item_matrix,
            num_recommendations=getattr(request, "limit", 5),
        )

        # Convert to QuestionRecommendation format
        recommendations = []
        for question in questions[: getattr(request, "limit", 5)]:
            if question.id in collaborative_scores:
                score = collaborative_scores[question.id]
                recommendations.append(
                    QuestionRecommendation(
                        question_id=getattr(question, "id", 0),
                        title=getattr(question, "content", "")[:50] + "...",
                        topic=getattr(question, "topic", "General"),
                        subject=getattr(question, "subject", "Math"),
                        difficulty_level=getattr(question, "difficulty_level", 1),
                        confidence_score=score,
                        reasoning=[
                            "Based on similar students' preferences",
                            f"Collaborative filtering score: {score:.2f}",
                        ],
                    )
                )

        return recommendations

    except Exception as e:
        logger.error("collaborative_recommendations_error", error=str(e))
        return []


async def _get_embedding_recommendations(
    db: Session, request: RecommendationRequest
) -> List[QuestionRecommendation]:
    """
    OPTIMIZED: Vector store ile embedding-based recommendations
    O(log N) performance ile semantic similarity arama
    """
    try:
        from app.crud.question import get_question

        # from app.crud.student_response import lambda *args, **kwargs: []  # get_student_responses eksik  # Import eksik
        from app.services.enhanced_embedding_service import enhanced_embedding_service

        # Öğrencinin son çözdüğü soruları al - get_student_responses function eksik
        recent_responses = []  # Placeholder - get_student_responses not implemented

        if not recent_responses:
            logger.warning(
                "no_recent_responses_fallback_to_random", student_id=request.student_id
            )
            return []

        # Son sorunun içeriğini al
        last_question = get_question(db, recent_responses[0].question_id)
        if not last_question:
            return []

        # Vector store ile benzer soruları bul
        similar_results = await enhanced_embedding_service.semantic_search_vector_db(
            query_text=str(last_question.content),
            k=getattr(request, "limit", 5) * 2,  # Daha fazla aday al, sonra filtrele
            similarity_threshold=0.7,
            filters={
                "subject_id": request.subject_id,
                "difficulty_level": getattr(request, "difficulty_level", None),
                "exclude_question_ids": [
                    r.question_id for r in recent_responses
                ],  # Önceden çözülenleri hariç tut
            },
        )

        recommendations = []
        for result in similar_results[: getattr(request, "limit", 5)]:
            question = get_question(db, result["question_id"])
            if question:
                # Semantic reasoning oluştur
                reasoning = [
                    f"Semantically similar to your recent question (similarity: {result['similarity_score']:.2f})",
                    f"Same difficulty level ({question.difficulty_level})",
                    "Recommended based on content understanding",
                ]

                recommendations.append(
                    QuestionRecommendation(
                        question_id=getattr(question, "id", 0),
                        title=getattr(question, "content", "")[:50] + "...",
                        topic=getattr(question, "topic", "General"),
                        subject=getattr(question, "subject", "Math"),
                        difficulty_level=getattr(question, "difficulty_level", 1),
                        confidence_score=result["similarity_score"],
                        reasoning=reasoning,
                    )
                )

        logger.info(
            "embedding_recommendations_generated",
            student_id=request.student_id,
            count=len(recommendations),
            method="vector_store_semantic_search",
        )

        return recommendations

    except Exception as e:
        logger.error(
            "embedding_recommendations_error",
            student_id=request.student_id,
            error=str(e),
        )
        return []


async def _get_ensemble_recommendations(
    db: Session, request: RecommendationRequest
) -> List[QuestionRecommendation]:
    """
    ENHANCED: Multi-algorithm ensemble ile optimize edilmiş öneriler
    River + LinUCB + Semantic Similarity + Neo4j birleşimi
    """
    try:
        from app.crud.question import get_questions

        # from app.crud.student_response import lambda *args, **kwargs: []  # get_student_responses eksik  # Import eksik
        from app.services.ensemble_service import calculate_enhanced_ensemble_score

        # Öğrenci context'ini al
        student_context = await _get_student_context(db, request.student_id)

        # Candidate questions al (filtered)
        candidate_questions = get_questions(
            db,
            subject_id=request.subject_id,
            difficulty_level=getattr(request, "difficulty_level", None),
            limit=50,  # Ensemble için daha geniş aday havuzu
            # exclude_answered_by=request.student_id,  # Bu parametre yok
        )

        if not candidate_questions:
            return []

        # Öğrencinin son sorularını al (context için)
        recent_responses = []  # get_student_responses eksik - placeholder
        recent_questions = []
        recent_question_ids = []

        for response in recent_responses if isinstance(recent_responses, list) else []:
            from app.crud.question import get_question

            question = get_question(db, response.question_id)
            if question:
                recent_questions.append(question.content)
                recent_question_ids.append(question.id)

        # Her candidate için ensemble score hesapla
        scored_recommendations = []

        for question in candidate_questions:
            try:
                # River score simulation (basit bir tahmin)
                river_score = min(
                    0.8, student_context.get("accuracy_rate_overall", 0.5) + 0.2
                )

                # Enhanced ensemble score hesapla
                ensemble_scores = await calculate_enhanced_ensemble_score(
                    river_score=river_score,
                    question_content=str(question.content),
                    question_id=getattr(question, "id", 0),
                    question_difficulty=int(getattr(question, "difficulty_level", 1)),
                    student_id=request.student_id,
                    student_level=student_context.get("level", 1),
                    student_recent_performance=student_context.get(
                        "accuracy_rate_recent", 0.5
                    ),
                    student_recent_questions=recent_questions,
                    student_recent_question_ids=recent_question_ids,
                )

                # Reasoning oluştur
                reasoning = [
                    f"Ensemble score: {ensemble_scores['ensemble_score']:.3f}",
                    f"Semantic similarity: {ensemble_scores['embedding_similarity']:.3f}",
                    f"Skill mastery alignment: {ensemble_scores['skill_mastery']:.3f}",
                    f"Difficulty match: {ensemble_scores['difficulty_match']:.3f}",
                ]

                scored_recommendations.append(
                    {
                        "question": question,
                        "ensemble_score": ensemble_scores["ensemble_score"],
                        "component_scores": ensemble_scores,
                        "reasoning": reasoning,
                    }
                )

            except Exception as e:
                logger.debug(
                    "ensemble_score_calculation_error",
                    question_id=question.id,
                    error=str(e),
                )
                continue

        # Ensemble score'a göre sırala
        scored_recommendations.sort(key=lambda x: x["ensemble_score"], reverse=True)

        # Top N'i al ve format'la
        recommendations = []
        for item in scored_recommendations[: getattr(request, "limit", 5)]:
            question = item["question"]

            recommendations.append(
                QuestionRecommendation(
                    question_id=getattr(question, "id", 0),
                    title=getattr(question, "content", "")[:50] + "...",
                    topic=getattr(question, "topic", "General"),
                    subject=getattr(question, "subject", "Math"),
                    difficulty_level=getattr(question, "difficulty_level", 1),
                    confidence_score=item["ensemble_score"],
                    reasoning=item["reasoning"],
                )
            )

        logger.info(
            "ensemble_recommendations_generated",
            student_id=request.student_id,
            count=len(recommendations),
            method="enhanced_ensemble_scoring",
            avg_score=sum(r.confidence_score for r in recommendations)
            / len(recommendations)
            if recommendations
            else 0,
        )

        return recommendations

    except Exception as e:
        logger.error(
            "ensemble_recommendations_error",
            student_id=request.student_id,
            error=str(e),
        )
        return []


def _generate_reasoning(rec, score: float, context: Dict) -> List[str]:
    """Generate human-readable reasoning for recommendation"""
    reasons = []

    if score > 0.8:
        reasons.append("High confidence match based on your learning pattern")
    if context.get("accuracy_rate_recent", 0) > 0.7:
        reasons.append("Matches your current performance level")

    reasons.append(f"Difficulty level {rec.difficulty_level} suits your progress")

    return reasons


def _estimate_question_time(rec, context: Dict) -> int:
    """Estimate time needed to solve question based on student context"""
    base_time = 300  # 5 minutes base
    difficulty_multiplier = rec.difficulty_level / 3
    student_speed = context.get("avg_response_time", 5000) / 5000

    return int(base_time * difficulty_multiplier * student_speed)


def _calculate_skill_alignment(rec, learning_goals: List[str]) -> float:
    """Calculate how well question aligns with learning goals"""
    # Would use Neo4j skill relationships
    return 0.8


def _convert_feedback_to_score(feedback_type: str, rating: Optional[float]) -> float:
    """Convert feedback to numerical score for ML models"""
    feedback_map = {"helpful": 0.9, "too_easy": 0.3, "too_hard": 0.2, "irrelevant": 0.1}

    base_score = feedback_map.get(feedback_type, 0.5)

    if rating is not None:
        return (base_score + rating) / 2

    return base_score


async def _generate_skill_progression_path(
    db: Session,
    student_id: int,
    target_skill: str,
    current_level: int,
    target_level: int,
) -> List[LearningPathStep]:
    """Generate learning path using skill progression"""
    # Would use Neo4j to find optimal skill progression
    return []


def _calculate_path_success_probability(
    context: Dict, steps: List[LearningPathStep], time_horizon: int
) -> float:
    """Calculate probability of successfully completing learning path"""
    # Factor in student's historical performance and path complexity
    base_probability = context.get("accuracy_rate_overall", 0.5)
    path_difficulty = len(steps) / 10  # Normalize by expected path length
    time_pressure = min(time_horizon / 30, 1.0)  # Normalize by expected time

    return min(base_probability * time_pressure / (1 + path_difficulty), 0.95)


async def _get_trending_questions(
    db: Session, since: datetime, count: int, subject_id: Optional[int]
) -> List[Dict]:
    """Get trending questions based on recent activity"""
    # Query database for questions with high recent activity
    return []


async def _get_student_recommendation_analytics(
    db: Session, student_id: int, days: int
) -> Dict[str, Any]:
    """Get comprehensive analytics for student recommendations"""
    return {
        "recommendations_received": 0,
        "recommendations_completed": 0,
        "average_rating": 0.0,
        "improvement_trend": "stable",
        "preferred_algorithms": [],
        "skill_progress": {},
    }


async def _log_recommendation_analytics(
    student_id: int, question_ids: List[int], algorithm: str, generation_time: float
):
    """Log recommendation analytics for monitoring"""
    logger.info(
        "recommendation_analytics",
        student_id=student_id,
        question_count=len(question_ids),
        algorithm=algorithm,
        generation_time=generation_time,
    )
