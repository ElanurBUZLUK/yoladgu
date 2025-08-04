import json
import time
from datetime import datetime
from typing import List, Optional

import redis
import structlog
from app.core.config import settings
from app.crud.question import (
    create_question,
    delete_question,
    get_popular_questions,
    get_question,
    get_question_skill_centrality,
    get_question_statistics,
    get_questions,
    get_random_question,
    get_shared_skills_between_questions,
    get_similar_questions_from_neo4j,
    get_skill_name_by_question,
    search_questions,
    update_question,
    update_question_embedding,
)
from app.crud.student_response import (
    get_student_learning_path_from_neo4j,
    get_student_skill_mastery_from_neo4j,
)
from app.crud.user import get_current_user
from app.db.database import get_db
from app.db.models import Skill, Subject, Topic, User
from app.schemas.question import (
    AnswerResponse,
    AnswerSubmission,
    LearningPath,
    QuestionCreate,
    QuestionResponse,
    QuestionSimilarity,
    QuestionUpdate,
    RecommendationResponse,
    SkillCentrality,
    SkillResponse,
    StudentSkillMastery,
    SubjectResponse,
    TopicResponse,
)
from app.services.embedding_service import EmbeddingService
from app.services.recommendation_service import recommendation_service
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

logger = structlog.get_logger()
router = APIRouter()


@router.get("/questions/", response_model=List[QuestionResponse])
def get_questions_endpoint(
    skip: int = 0,
    limit: int = 100,
    subject_id: Optional[int] = None,
    topic_id: Optional[int] = None,
    difficulty_level: Optional[int] = None,
    question_type: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get all questions with optional filtering"""
    questions = get_questions(
        db,
        skip=skip,
        limit=limit,
        subject_id=subject_id,
        topic_id=topic_id,
        difficulty_level=difficulty_level,
        question_type=question_type,
    )
    return questions


@router.get("/questions/search", response_model=dict)
def search_questions_endpoint(
    q: str = Query(..., description="Search term"),
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Search questions by content"""
    questions = search_questions(db, q, limit)
    return {
        "search_term": q,
        "results": [
            {
                "id": q.id,
                "content": q.content,
                "difficulty_level": q.difficulty_level,
                "subject_id": q.subject_id,
                "question_type": q.question_type,
            }
            for q in questions
        ],
    }


@router.get("/questions/popular", response_model=dict)
def get_popular_questions_endpoint(
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get most popular questions (most answered)"""
    questions = get_popular_questions(db, limit)
    return {
        "popular_questions": [
            {
                "id": q.id,
                "content": q.content,
                "difficulty_level": q.difficulty_level,
                "subject_id": q.subject_id,
            }
            for q in questions
        ]
    }


@router.get("/questions/statistics", response_model=dict)
def get_questions_statistics(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    """Get question statistics"""
    stats = get_question_statistics(db)
    return stats


@router.get("/questions/random", response_model=QuestionResponse)
def get_random_question_endpoint(
    subject_id: Optional[int] = None,
    topic_id: Optional[int] = None,
    difficulty_level: Optional[int] = None,
    question_type: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a random question with optional filtering"""
    question = get_random_question(
        db=db,
        subject_id=subject_id,
        topic_id=topic_id,
        difficulty_level=difficulty_level,
        question_type=question_type,
    )
    if not question:
        raise HTTPException(status_code=404, detail="No questions available")
    return question


@router.get("/questions/{question_id}", response_model=QuestionResponse)
def get_question_endpoint(
    question_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a specific question by ID"""
    question = get_question(db, question_id)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    return question


@router.get("/questions/{question_id}/similar", response_model=List[QuestionSimilarity])
def get_similar_questions(
    question_id: int,
    limit: int = 10,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get similar questions from Neo4j"""
    try:
        similar_data = get_similar_questions_from_neo4j(question_id, limit)
        similar_questions = []

        for item in similar_data:
            question = get_question(db, item["question_id"])
            if question:
                # Ortak skill'leri getir
                shared_skills = get_shared_skills_between_questions(
                    db, question_id, item["question_id"]
                )

                similar_questions.append(
                    QuestionSimilarity(
                        question_id=getattr(question, "id", 0),
                        similarity_score=item["shared_skills"],
                        shared_skills=shared_skills,  # Gerçek skill isimleri
                        question=QuestionResponse.from_orm(question),
                    )
                )

        return similar_questions
    except Exception as e:
        logger.error("similar_questions_error", question_id=question_id, error=str(e))
        raise HTTPException(status_code=500, detail="Error fetching similar questions")


@router.get("/questions/{question_id}/skill-centrality", response_model=SkillCentrality)
def get_question_skill_centrality_endpoint(
    question_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get skill centrality scores for a question"""
    try:
        centrality_data = get_question_skill_centrality(question_id)
        _ = get_question(db, question_id)  # Variable not used, just checking if exists

        # Skill adını getir
        skill_name = get_skill_name_by_question(db, question_id)

        # Ortalama mastery hesapla (tüm öğrenciler için)
        try:
            from sqlalchemy import text

            mastery_query = text(
                """
                SELECT AVG(CASE WHEN sr.is_correct THEN 1.0 ELSE 0.0 END) as avg_mastery
                FROM student_responses sr
                WHERE sr.question_id = :question_id
                AND sr.created_at >= NOW() - INTERVAL '30 days'
            """
            )
            mastery_result = db.execute(
                mastery_query, {"question_id": question_id}
            ).fetchone()
            avg_mastery = (
                float(mastery_result.avg_mastery)
                if mastery_result and mastery_result.avg_mastery
                else 0.5
            )
        except Exception:
            avg_mastery = 0.5  # Default fallback

        return SkillCentrality(
            skill_id=question_id,
            skill_name=skill_name,  # Gerçek skill adı
            centrality_score=centrality_data["centrality"],
            question_count=centrality_data["skill_count"],
            student_mastery_avg=avg_mastery,  # Hesaplanan mastery
        )
    except Exception as e:
        logger.error("skill_centrality_error", question_id=question_id, error=str(e))
        raise HTTPException(status_code=500, detail="Error fetching skill centrality")


@router.post("/questions/", response_model=QuestionResponse)
def create_question_endpoint(
    question: QuestionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new question"""
    try:
        # Embedding hesapla
        embedding_service = EmbeddingService()
        embedding = embedding_service.compute_embedding(question.content)

        # Soruyu oluştur
        db_question = create_question(db, question, getattr(current_user, "id", 1))

        # Embedding'i kaydet
        update_question_embedding(
            db, getattr(db_question, "id", 0), json.dumps(embedding)
        )

        # Neo4j'e ekle
        if settings.USE_NEO4J:
            try:
                from app.services.recommendation_service import add_question_to_neo4j

                # Convert skill_ids from Dict[int, float] to List[int]
                skill_ids_list = (
                    list(question.skill_ids.keys()) if question.skill_ids else []
                )
                add_question_to_neo4j(db_question.id, skill_ids_list)
            except Exception as e:
                logger.error(
                    "neo4j_add_error", question_id=db_question.id, error=str(e)
                )

        return db_question
    except Exception as e:
        logger.error("create_question_error", error=str(e))
        raise HTTPException(status_code=500, detail="Error creating question")


@router.put("/questions/{question_id}", response_model=QuestionResponse)
def update_question_endpoint(
    question_id: int,
    question_update: QuestionUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update an existing question"""
    question = get_question(db, question_id)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    # Content değiştiyse embedding'i güncelle
    if question_update.content and question_update.content != question.content:
        embedding_service = EmbeddingService()
        embedding = embedding_service.compute_embedding(question_update.content)
        update_question_embedding(db, question_id, json.dumps(embedding))

    updated_question = update_question(db, question, question_update)
    return updated_question


@router.delete("/questions/{question_id}")
def delete_question_endpoint(
    question_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a question (soft delete)"""
    question = get_question(db, question_id)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    _ = delete_question(db, question_id)  # Variable not used
    return {"message": "Question deleted successfully"}


@router.post("/questions/{question_id}/answer", response_model=AnswerResponse)
def submit_answer(
    question_id: int,
    answer_data: AnswerSubmission,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Submit an answer to a question"""
    start_time = time.time()

    # Soruyu kontrol et
    question = get_question(db, question_id)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    # Cevabın doğruluğunu kontrol et
    is_correct = (
        answer_data.answer.strip().lower() == question.correct_answer.strip().lower()
    )

    # Response time'ı answer_data'dan al veya hesapla
    response_time = (
        answer_data.response_time
        if answer_data.response_time
        else (time.time() - start_time)
    )

    # Öğrenci cevabını kaydet
    from app.crud.student_response import create_response

    student_response = create_response(
        db=db,
        student_id=getattr(current_user, "id", 1),
        question_id=question_id,
        answer=answer_data.answer,
        is_correct=is_correct,
        response_time=response_time,
        confidence_level=answer_data.confidence_level,
        feedback=answer_data.feedback,
    )

    # --- Asenkron güncelleme için Redis Stream'e event yaz ---
    try:
        r = redis.Redis.from_url(settings.redis_url)
        event = {
            "student_id": current_user.id,
            "question_id": question_id,
            "is_correct": is_correct,
            "response_time": response_time,
            "timestamp": time.time(),
        }
        if r:
            r.xadd("student_responses_stream", event)
    except Exception as e:
        logger.error("redis_stream_error", error=str(e))
    # ------------------------------------------------------------------

    return AnswerResponse(
        question_id=question_id,
        is_correct=is_correct,
        correct_answer=str(question.correct_answer),
        explanation=str(question.explanation)
        if getattr(question, "explanation", None)
        else None,
        response_time=response_time,
        response_id=getattr(student_response, "id", 0),
        points_earned=10 if is_correct else 0,  # Basit puanlama sistemi
        current_streak=None,  # Bu ileride hesaplanabilir
        message="Doğru cevap!" if is_correct else "Yanlış cevap, tekrar dene!",
    )


@router.get("/recommendations/", response_model=RecommendationResponse)
async def get_recommendations(
    n_recommendations: int = 5,
    subject_id: Optional[int] = None,
    topic_id: Optional[int] = None,
    difficulty_range: Optional[str] = None,
    include_explanations: bool = False,
    use_ensemble: bool = True,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get personalized question recommendations"""
    try:
        # Difficulty range parsing - variables kept for future use
        _ = None  # min_difficulty placeholder
        _ = None  # max_difficulty placeholder
        if difficulty_range:
            try:
                _, _ = map(int, difficulty_range.split("-"))  # Parse but don't use yet
            except:
                pass

        recommendations = await recommendation_service.get_recommendations(
            db=db,
            student_id=getattr(current_user, "id", 1),
            n_recommendations=n_recommendations,
        )

        # Student profile bilgilerini al - placeholder için None kullan
        profile = None

        return RecommendationResponse(
            recommendations=recommendations,  # type: ignore
            student_level=profile.level if profile else 1.0,
            total_questions_answered=profile.total_questions_answered if profile else 0,
            accuracy_rate=(
                profile.total_correct_answers / profile.total_questions_answered * 100
            )
            if profile and profile.total_questions_answered > 0
            else 0.0,
            average_response_time=profile.average_response_time if profile else 0.0,
            generated_at=datetime.utcnow(),
        )
    except Exception as e:
        logger.error("recommendations_error", student_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Error generating recommendations")


@router.get("/recommendations/next-question", response_model=dict)
async def get_next_question(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    """Get the next recommended question"""
    try:
        recommendations = await recommendation_service.get_recommendations(
            db=db, student_id=getattr(current_user, "id", 1), n_recommendations=1
        )

        if not recommendations:
            raise HTTPException(status_code=404, detail="No recommendations available")

        return recommendations[0]
    except Exception as e:
        logger.error("next_question_error", student_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Error getting next question")


@router.get(
    "/students/{student_id}/skill-mastery", response_model=List[StudentSkillMastery]
)
def get_student_skill_mastery(
    student_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get student skill mastery levels"""
    try:
        if settings.USE_NEO4J:
            mastery_data = get_student_skill_mastery_from_neo4j(student_id)
            return [
                StudentSkillMastery(
                    skill_id=item["skill_id"],
                    skill_name=item["skill_name"],
                    mastery_level=item["mastery_level"],
                    questions_answered=item["questions_answered"],
                    correct_answers=item["correct_answers"],
                    average_response_time=item["average_response_time"],
                )
                for item in mastery_data
            ]
        else:
            return []
    except Exception as e:
        logger.error("skill_mastery_error", student_id=student_id, error=str(e))
        raise HTTPException(status_code=500, detail="Error fetching skill mastery")


@router.get("/students/{student_id}/learning-path", response_model=List[LearningPath])
def get_student_learning_path(
    student_id: int,
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get student learning path"""
    try:
        if settings.USE_NEO4J:
            path_data = get_student_learning_path_from_neo4j(student_id, limit)
            return [
                LearningPath(
                    skill_id=item["skill_id"],
                    skill_name=item["skill_name"],
                    current_level=item["current_level"],
                    target_level=item["target_level"],
                    recommended_questions=item["recommended_questions"],
                    estimated_time=item["estimated_time"],
                )
                for item in path_data
            ]
        else:
            return []
    except Exception as e:
        logger.error("learning_path_error", student_id=student_id, error=str(e))
        raise HTTPException(status_code=500, detail="Error fetching learning path")


@router.get("/subjects/", response_model=List[SubjectResponse])
def get_subjects(db: Session = Depends(get_db)):
    """Get all subjects"""
    subjects = db.query(Subject).all()
    return subjects


@router.get("/topics/", response_model=List[TopicResponse])
def get_topics(subject_id: Optional[int] = None, db: Session = Depends(get_db)):
    """Get topics with optional subject filtering"""
    query = db.query(Topic)
    if subject_id:
        query = query.filter(Topic.subject_id == subject_id)
    return query.all()


@router.get("/skills/", response_model=List[SkillResponse])
def get_skills(subject_id: Optional[int] = None, db: Session = Depends(get_db)):
    """Get skills with optional subject filtering"""
    query = db.query(Skill)
    if subject_id:
        query = query.filter(Skill.subject_id == subject_id)
    return query.all()


@router.post("/questions/batch-embedding")
def batch_update_embeddings(
    subject_id: Optional[int] = None,
    batch_size: int = 50,
    max_batches: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Batch update question embeddings"""
    try:
        from scripts.batch_embedding_update import update_embeddings_for_subject

        if subject_id:
            updated_count = update_embeddings_for_subject(subject_id)
        else:
            from scripts.batch_embedding_update import update_all_embeddings

            updated_count = update_all_embeddings()

        return {
            "message": f"Updated {updated_count} question embeddings",
            "subject_id": subject_id,
            "batch_size": batch_size,
        }
    except Exception as e:
        logger.error("batch_embedding_error", error=str(e))
        raise HTTPException(status_code=500, detail="Error updating embeddings")
