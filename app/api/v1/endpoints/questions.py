from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from app.db.database import get_db
from app.db.models import Question, Subject, Skill, QuestionSkill, StudentResponse, Topic
from app.schemas.question import (
    QuestionCreate, QuestionUpdate, QuestionResponse, QuestionRecommendation,
    QuestionSimilarity, StudentSkillMastery, LearningPath, RecommendationRequest,
    RecommendationResponse, SkillCentrality
)
from app.services.recommendation_service import recommendation_service
from app.services.embedding_service import EmbeddingService
from app.crud.user import get_current_user
from app.crud.question import (
    get_questions, get_question, create_question, update_question, 
    delete_question, get_popular_questions, get_recommended_questions,
    get_shared_skills_between_questions, get_skill_name_by_question,
    get_similar_questions_from_neo4j, get_question_skill_centrality,
    get_questions_by_skills, get_questions_without_embeddings, update_question_embedding,
    get_question_statistics, search_questions, get_questions_by_difficulty_range
)
from app.crud.student_response import get_student_skill_mastery_from_neo4j, get_student_learning_path_from_neo4j
from app.db.models import User
import time
import json
import redis
from app.core.config import settings
import structlog
from datetime import datetime

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
    current_user: User = Depends(get_current_user)
):
    """Get all questions with optional filtering"""
    questions = get_questions(
        db, skip=skip, limit=limit, subject_id=subject_id, 
        topic_id=topic_id, difficulty_level=difficulty_level, 
        question_type=question_type
    )
    return questions

@router.get("/questions/search")
def search_questions_endpoint(
    q: str = Query(..., description="Search term"),
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
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
                "question_type": q.question_type
            }
            for q in questions
        ]
    }

@router.get("/questions/popular")
def get_popular_questions_endpoint(
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get most popular questions (most answered)"""
    questions = get_popular_questions(db, limit)
    return {
        "popular_questions": [
            {
                "id": q.id,
                "content": q.content,
                "difficulty_level": q.difficulty_level,
                "subject_id": q.subject_id
            }
            for q in questions
        ]
    }

@router.get("/questions/statistics")
def get_questions_statistics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get question statistics"""
    stats = get_question_statistics(db)
    return stats

@router.get("/questions/{question_id}", response_model=QuestionResponse)
def get_question_endpoint(
    question_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
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
    current_user: User = Depends(get_current_user)
):
    """Get similar questions from Neo4j"""
    try:
        similar_data = get_similar_questions_from_neo4j(question_id, limit)
        similar_questions = []
        
        for item in similar_data:
            question = get_question(db, item["question_id"])
            if question:
                # Ortak skill'leri getir
                shared_skills = get_shared_skills_between_questions(db, question_id, item["question_id"])
                
                similar_questions.append(QuestionSimilarity(
                    question_id=question.id,
                    similarity_score=item["shared_skills"],
                    shared_skills=shared_skills,  # Gerçek skill isimleri
                    question=question
                ))
        
        return similar_questions
    except Exception as e:
        logger.error("similar_questions_error", question_id=question_id, error=str(e))
        raise HTTPException(status_code=500, detail="Error fetching similar questions")

@router.get("/questions/{question_id}/skill-centrality", response_model=SkillCentrality)
def get_question_skill_centrality_endpoint(
    question_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get skill centrality scores for a question"""
    try:
        centrality_data = get_question_skill_centrality(question_id)
        question = get_question(db, question_id)
        
        # Skill adını getir
        skill_name = get_skill_name_by_question(db, question_id)
        
        # Ortalama mastery hesapla (tüm öğrenciler için)
        try:
            from sqlalchemy import text
            mastery_query = text("""
                SELECT AVG(CASE WHEN sr.is_correct THEN 1.0 ELSE 0.0 END) as avg_mastery
                FROM student_responses sr
                WHERE sr.question_id = :question_id
                AND sr.created_at >= NOW() - INTERVAL '30 days'
            """)
            mastery_result = db.execute(mastery_query, {"question_id": question_id}).fetchone()
            avg_mastery = float(mastery_result.avg_mastery) if mastery_result.avg_mastery else 0.5
        except Exception:
            avg_mastery = 0.5  # Default fallback
        
        return SkillCentrality(
            skill_id=question_id,
            skill_name=skill_name,  # Gerçek skill adı
            centrality_score=centrality_data["centrality"],
            question_count=centrality_data["skill_count"],
            student_mastery_avg=avg_mastery  # Hesaplanan mastery
        )
    except Exception as e:
        logger.error("skill_centrality_error", question_id=question_id, error=str(e))
        raise HTTPException(status_code=500, detail="Error fetching skill centrality")

@router.post("/questions/", response_model=QuestionResponse)
def create_question_endpoint(
    question: QuestionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new question"""
    try:
        # Embedding hesapla
        embedding_service = EmbeddingService()
        embedding = embedding_service.compute_embedding(question.content)
        
        # Soruyu oluştur
        db_question = create_question(db, question, current_user.id)
        
        # Embedding'i kaydet
        update_question_embedding(db, db_question.id, json.dumps(embedding.tolist()))
        
        # Neo4j'e ekle
        if settings.USE_NEO4J:
            try:
                from app.services.recommendation_service import add_question_to_neo4j
                add_question_to_neo4j(db_question.id, question.skill_ids or {})
            except Exception as e:
                logger.error("neo4j_add_error", question_id=db_question.id, error=str(e))
        
        return db_question
    except Exception as e:
        logger.error("create_question_error", error=str(e))
        raise HTTPException(status_code=500, detail="Error creating question")

@router.put("/questions/{question_id}", response_model=QuestionResponse)
def update_question_endpoint(
    question_id: int,
    question_update: QuestionUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update an existing question"""
    question = get_question(db, question_id)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    # Content değiştiyse embedding'i güncelle
    if question_update.content and question_update.content != question.content:
        embedding_service = EmbeddingService()
        embedding = embedding_service.compute_embedding(question_update.content)
        update_question_embedding(db, question_id, json.dumps(embedding.tolist()))
    
    updated_question = update_question(db, question, question_update)
    return updated_question

@router.delete("/questions/{question_id}")
def delete_question_endpoint(
    question_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a question (soft delete)"""
    question = get_question(db, question_id)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    deleted_question = delete_question(db, question_id)
    return {"message": "Question deleted successfully"}

@router.post("/questions/{question_id}/answer")
def submit_answer(
    question_id: int,
    answer: str,
    confidence_level: Optional[int] = None,
    feedback: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Submit an answer to a question"""
    start_time = time.time()
    
    # Soruyu kontrol et
    question = get_question(db, question_id)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    # Cevabın doğruluğunu kontrol et
    is_correct = answer.strip().lower() == question.correct_answer.strip().lower()
    response_time = time.time() - start_time
    
    # Öğrenci cevabını kaydet
    from app.crud.student_response import create_response
    student_response = create_response(
        db=db,
        student_id=current_user.id,
        question_id=question_id,
        answer=answer,
        is_correct=is_correct,
        response_time=response_time,
        confidence_level=confidence_level,
        feedback=feedback
    )
    
    # --- Asenkron güncelleme için Redis Stream'e event yaz ---
    try:
        r = redis.Redis.from_url(settings.redis_url)
        event = {
            "student_id": current_user.id,
            "question_id": question_id,
            "is_correct": is_correct,
            "response_time": response_time,
            "timestamp": time.time()
        }
        r.xadd("student_responses_stream", event)
    except Exception as e:
        logger.error("redis_stream_error", error=str(e))
    # ------------------------------------------------------------------
    
    return {
        "question_id": question_id,
        "is_correct": is_correct,
        "correct_answer": question.correct_answer,
        "explanation": question.explanation,
        "response_time": response_time,
        "response_id": student_response.id
    }

@router.get("/recommendations/", response_model=RecommendationResponse)
def get_recommendations(
    n_recommendations: int = 5,
    subject_id: Optional[int] = None,
    topic_id: Optional[int] = None,
    difficulty_range: Optional[str] = None,
    include_explanations: bool = False,
    use_ensemble: bool = True,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get personalized question recommendations"""
    try:
        # Difficulty range parsing
        min_difficulty = None
        max_difficulty = None
        if difficulty_range:
            try:
                min_difficulty, max_difficulty = map(int, difficulty_range.split('-'))
            except:
                pass
        
        recommendations = recommendation_service.get_recommendations(
            db=db,
            student_id=current_user.id,
            n_recommendations=n_recommendations,
            subject_id=subject_id,
            topic_id=topic_id,
            min_difficulty=min_difficulty,
            max_difficulty=max_difficulty,
            use_ensemble=use_ensemble
        )
        
        # Student profile bilgilerini al
        from app.crud.student_response import get_student_profile
        profile = get_student_profile(db, current_user.id)
        
        return RecommendationResponse(
            recommendations=recommendations,
            student_level=profile.level if profile else 1.0,
            total_questions_answered=profile.total_questions_answered if profile else 0,
            accuracy_rate=(profile.total_correct_answers / profile.total_questions_answered * 100) if profile and profile.total_questions_answered > 0 else 0.0,
            average_response_time=profile.average_response_time if profile else 0.0,
            generated_at=datetime.utcnow()
        )
    except Exception as e:
        logger.error("recommendations_error", student_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Error generating recommendations")

@router.get("/recommendations/next-question")
def get_next_question(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get the next recommended question"""
    try:
        recommendations = recommendation_service.get_recommendations(
            db=db,
            student_id=current_user.id,
            n_recommendations=1
        )
        
        if not recommendations:
            raise HTTPException(status_code=404, detail="No recommendations available")
        
        return recommendations[0]
    except Exception as e:
        logger.error("next_question_error", student_id=current_user.id, error=str(e))
        raise HTTPException(status_code=500, detail="Error getting next question")

@router.get("/students/{student_id}/skill-mastery", response_model=List[StudentSkillMastery])
def get_student_skill_mastery(
    student_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
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
                    average_response_time=item["average_response_time"]
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
    current_user: User = Depends(get_current_user)
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
                    estimated_time=item["estimated_time"]
                )
                for item in path_data
            ]
        else:
            return []
    except Exception as e:
        logger.error("learning_path_error", student_id=student_id, error=str(e))
        raise HTTPException(status_code=500, detail="Error fetching learning path")

@router.get("/subjects/")
def get_subjects(db: Session = Depends(get_db)):
    """Get all subjects"""
    subjects = db.query(Subject).all()
    return subjects

@router.get("/topics/")
def get_topics(
    subject_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get topics with optional subject filtering"""
    query = db.query(Topic)
    if subject_id:
        query = query.filter(Topic.subject_id == subject_id)
    return query.all()

@router.get("/skills/")
def get_skills(
    subject_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
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
    current_user: User = Depends(get_current_user)
):
    """Batch update question embeddings"""
    try:
        from scripts.batch_embedding_update import update_embeddings_for_subject
        
        if subject_id:
            updated_count = update_embeddings_for_subject(db, subject_id, batch_size, max_batches)
        else:
            from scripts.batch_embedding_update import update_all_embeddings
            updated_count = update_all_embeddings(db, batch_size, max_batches)
        
        return {
            "message": f"Updated {updated_count} question embeddings",
            "subject_id": subject_id,
            "batch_size": batch_size
        }
    except Exception as e:
        logger.error("batch_embedding_error", error=str(e))
        raise HTTPException(status_code=500, detail="Error updating embeddings") 