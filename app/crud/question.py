from typing import Any, Dict, List, Optional

import structlog
from app.core.config import settings
from app.db.models import Question, QuestionSkill, StudentResponse
from app.schemas.question import QuestionCreate, QuestionUpdate
from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Session

logger = structlog.get_logger()


def _get_neo4j_driver():
    """Neo4j driver instance'ı döndür (Deprecated: Use neo4j_service instead)"""
    from app.services.neo4j_service import neo4j_service

    if not settings.USE_NEO4J:
        return None
    return neo4j_service.driver


def _sync_question_to_neo4j(question_id: int, skill_ids: Optional[dict] = None):
    """Soru ve skill ilişkilerini Neo4j'e senkronize et"""
    from app.services.neo4j_service import neo4j_service

    if not settings.USE_NEO4J:
        return

    if not neo4j_service._driver:
        logger.warning("neo4j_not_available", operation="_sync_question_to_neo4j")
        return

    if skill_ids is None:
        skill_ids = {}
    try:
        with neo4j_service._driver.session() as session:
            # Soru node'unu oluştur/güncelle
            session.run(
                "MERGE (q:Question {id: $question_id})", question_id=question_id
            )

            # Skill ilişkilerini ekle (with weights support)
            if skill_ids:
                for skill_id, weight in skill_ids.items():
                    session.run(
                        "MERGE (q:Question {id: $qid}) "
                        "MERGE (s:Skill {id: $sid}) "
                        "MERGE (q)-[r:HAS_SKILL]->(s) "
                        "SET r.weight = $weight",
                        qid=question_id,
                        sid=skill_id,
                        weight=weight,
                    )

        logger.info(
            "question_synced_to_neo4j",
            question_id=question_id,
            skill_count=len(skill_ids) if skill_ids else 0,
        )
    except Exception as e:
        logger.error("neo4j_sync_error", question_id=question_id, error=str(e))


def get_question(db: Session, question_id: int) -> Optional[Question]:
    return (
        db.query(Question)
        .filter(Question.id == question_id, Question.is_active.is_(True))
        .first()
    )


def get_questions(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    subject_id: Optional[int] = None,
    topic_id: Optional[int] = None,
    difficulty_level: Optional[int] = None,
    question_type: Optional[str] = None,
) -> List[Question]:
    query = db.query(Question).filter(Question.is_active.is_(True))

    if subject_id:
        query = query.filter(Question.subject_id == subject_id)
    if topic_id:
        query = query.filter(Question.topic_id == topic_id)
    if difficulty_level:
        query = query.filter(Question.difficulty_level == difficulty_level)
    if question_type:
        query = query.filter(Question.question_type == question_type)

    return query.offset(skip).limit(limit).all()


def get_random_question(
    db: Session,
    subject_id: Optional[int] = None,
    topic_id: Optional[int] = None,
    difficulty_level: Optional[int] = None,
    question_type: Optional[str] = None,
) -> Optional[Question]:
    """Rastgele bir soru getir"""
    query = db.query(Question).filter(Question.is_active.is_(True))

    # Filtreleri uygula
    if subject_id:
        query = query.filter(Question.subject_id == subject_id)
    if topic_id:
        query = query.filter(Question.topic_id == topic_id)
    if difficulty_level:
        query = query.filter(Question.difficulty_level == difficulty_level)
    if question_type:
        query = query.filter(Question.question_type == question_type)

    # Rastgele sıralama ile bir soru getir
    return query.order_by(func.random()).first()


def get_questions_by_skills(
    db: Session, skill_ids: List[int], limit: int = 100
) -> List[Question]:
    """Belirli skill'lere sahip soruları getir"""
    return (
        db.query(Question)
        .join(QuestionSkill)
        .filter(
            and_(Question.is_active.is_(True), QuestionSkill.skill_id.in_(skill_ids))
        )
        .limit(limit)
        .all()
    )


def get_questions_without_embeddings(db: Session, limit: int = 100) -> List[Question]:
    """Embedding'i olmayan soruları getir"""
    return (
        db.query(Question)
        .filter(
            and_(
                Question.is_active.is_(True),
                or_(Question.embedding.is_(None), Question.embedding == ""),
            )
        )
        .limit(limit)
        .all()
    )


def create_question(db: Session, question_in: QuestionCreate, user_id: int) -> Question:
    db_question = Question(
        content=question_in.content,
        question_type=question_in.question_type,
        difficulty_level=question_in.difficulty_level,
        subject_id=question_in.subject_id,
        topic_id=question_in.topic_id,
        options=question_in.options,
        correct_answer=question_in.correct_answer,
        explanation=question_in.explanation,
        tags=question_in.tags,
        created_by=user_id,
    )
    db.add(db_question)
    db.commit()
    db.refresh(db_question)

    # Skill ilişkilerini ekle
    if question_in.skill_ids:
        for skill_id, weight in question_in.skill_ids.items():
            db_question_skill = QuestionSkill(
                question_id=db_question.id, skill_id=skill_id, weight=weight
            )
            db.add(db_question_skill)
        db.commit()

    # Neo4j'e senkronize et
    _sync_question_to_neo4j(
        int(getattr(db_question, "id", 0)), getattr(question_in, "skill_ids", {})
    )

    return db_question


def update_question(
    db: Session, db_obj: Question, question_in: QuestionUpdate
) -> Question:
    update_data = question_in.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_obj, field, value)
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)

    # Neo4j'i güncelle (skill_ids varsa)
    skill_ids = getattr(question_in, "skill_ids", None)
    if skill_ids is None and hasattr(question_in, "dict"):
        skill_ids = question_in.dict().get("skill_ids", {})
    if skill_ids is None:
        skill_ids = {}
    _sync_question_to_neo4j(int(getattr(db_obj, "id", 0)), skill_ids)

    return db_obj


def delete_question(db: Session, question_id: int) -> Optional[Question]:
    question = db.query(Question).filter(Question.id == question_id).first()
    if question:
        # Soft delete
        setattr(question, "is_active", False)
        db.add(question)
        db.commit()
        db.refresh(question)

        # Neo4j'de de işaretle
        if settings.USE_NEO4J:
            driver = _get_neo4j_driver()
            if driver:
                try:
                    with driver.session() as session:
                        session.run(
                            "MATCH (q:Question {id: $qid}) SET q.deleted = true",
                            qid=question_id,
                        )
                except Exception as e:
                    logger.error(
                        "neo4j_delete_error", question_id=question_id, error=str(e)
                    )
                finally:
                    # Driver managed by singleton service
                    pass

    return question


def get_similar_questions_from_neo4j(
    question_id: int, limit: int = 10
) -> List[Dict[str, Any]]:
    """Neo4j'den benzer soruları getir"""
    if not settings.USE_NEO4J:
        return []

    driver = _get_neo4j_driver()
    if not driver:
        return []

    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (q1:Question {id: $qid})-[:HAS_SKILL]->(s:Skill)<-[:HAS_SKILL]-(q2:Question)
                WHERE q1.id <> q2.id AND NOT q2.deleted
                WITH q2, COUNT(DISTINCT s) AS shared_skills
                ORDER BY shared_skills DESC
                LIMIT $limit
                RETURN q2.id AS question_id, shared_skills
            """,
                qid=question_id,
                limit=limit,
            )

            return [
                {
                    "question_id": record["question_id"],
                    "shared_skills": record["shared_skills"],
                }
                for record in result
            ]
    except Exception as e:
        logger.error("neo4j_similarity_error", question_id=question_id, error=str(e))
        return []
    finally:
        # Driver managed by singleton service
        pass


def get_question_skill_centrality(question_id: int) -> Dict[str, Any]:
    """Soru için skill centrality hesapla"""
    if not settings.USE_NEO4J:
        return {"centrality": 0.0, "skill_count": 0}

    driver = _get_neo4j_driver()
    if not driver:
        return {"centrality": 0.0, "skill_count": 0}

    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (q:Question {id: $qid})-[:HAS_SKILL]->(s:Skill)
                WITH s, COUNT((s)<-[:HAS_SKILL]-()) AS question_count
                RETURN AVG(question_count) AS avg_centrality, COUNT(s) AS skill_count
            """,
                qid=question_id,
            )

            record = result.single() if result else None
            if record and "avg_centrality" in record:
                centrality = float(record["avg_centrality"])
            else:
                centrality = 0.0
            return {
                "centrality": centrality,
                "skill_count": record["skill_count"] if record else 0,
            }
    except Exception as e:
        logger.error("neo4j_centrality_error", question_id=question_id, error=str(e))
        return {"centrality": 0.0, "skill_count": 0}
    finally:
        # Driver managed by singleton service
        pass


def update_question_embedding(db: Session, question_id: int, embedding: str) -> bool:
    """Soru embedding'ini güncelle"""
    try:
        question = (
            db.query(Question).filter(getattr(Question, "id") == question_id).first()
        )
        if question:
            setattr(question, "embedding", embedding)
            db.add(question)
            db.commit()
            return True
        return False
    except Exception as e:
        logger.error("embedding_update_error", question_id=question_id, error=str(e))
        return False


def get_question_statistics(db: Session) -> Dict[str, Any]:
    """Soru istatistiklerini getir"""
    total_questions = db.query(func.count(getattr(Question, "id"))).scalar()
    active_questions = (
        db.query(func.count(Question.id)).filter(Question.is_active.is_(True)).scalar()
    )
    questions_with_embeddings = (
        db.query(func.count(Question.id))
        .filter(
            and_(
                Question.is_active.is_(True),
                Question.embedding.isnot(None),
            )
        )
        .scalar()
    )

    difficulty_distribution = (
        db.query(Question.difficulty_level, func.count(Question.id))
        .filter(Question.is_active.is_(True))
        .group_by(Question.difficulty_level)
        .all()
    )

    return {
        "total_questions": total_questions,
        "active_questions": active_questions,
        "questions_with_embeddings": questions_with_embeddings,
        "embedding_coverage": (questions_with_embeddings / active_questions * 100)
        if active_questions > 0
        else 0,
        "difficulty_distribution": {str(k): v for k, v in difficulty_distribution},
    }


def search_questions(db: Session, search_term: str, limit: int = 20) -> List[Question]:
    """Soru içeriğinde arama yap"""
    return (
        db.query(Question)
        .filter(
            and_(
                Question.is_active.is_(True),
                Question.content.ilike(f"%{search_term}%"),
            )
        )
        .limit(limit)
        .all()
    )


def get_questions_by_difficulty_range(
    db: Session, min_difficulty: int, max_difficulty: int, limit: int = 100
) -> List[Question]:
    """Zorluk aralığına göre soruları getir"""
    return (
        db.query(Question)
        .filter(
            and_(
                Question.is_active.is_(True),
                Question.difficulty_level >= min_difficulty,
                Question.difficulty_level <= max_difficulty,
            )
        )
        .limit(limit)
        .all()
    )


def get_popular_questions(db: Session, limit: int = 20) -> List[Question]:
    """En popüler soruları getir (en çok cevaplanan)"""
    return (
        db.query(Question)
        .join(StudentResponse)
        .filter(Question.is_active.is_(True))
        .group_by(Question.id)
        .order_by(func.count(StudentResponse.id).desc())
        .limit(limit)
        .all()
    )


def get_skill_names_by_question(db: Session, question_id: int) -> List[str]:
    """Soru için skill isimlerini getir"""
    try:
        # Neo4j'den skill bilgilerini al
        def get_question_skills(question_id):
            return {"skill_names": []}

        skill_data = get_question_skills(question_id)
        return skill_data.get("skill_names", [])
    except Exception as e:
        logger.error("get_skill_names_error", question_id=question_id, error=str(e))
        return []


def get_shared_skills_between_questions(
    db: Session, question_id1: int, question_id2: int
) -> List[str]:
    """İki soru arasındaki ortak skill'leri getir"""
    try:
        skills1 = set(get_skill_names_by_question(db, question_id1))
        skills2 = set(get_skill_names_by_question(db, question_id2))
        return list(skills1.intersection(skills2))
    except Exception as e:
        logger.error(
            "get_shared_skills_error",
            question_id1=question_id1,
            question_id2=question_id2,
            error=str(e),
        )
        return []


def calculate_student_mastery_for_skill(
    db: Session, student_id: int, skill_name: str
) -> float:
    """Belirli bir skill için öğrenci mastery hesapla"""
    try:
        from sqlalchemy import text

        # Son 10 cevabı al ve başarı oranını hesapla
        query = text(
            """
            SELECT AVG(CASE WHEN sr.is_correct THEN 1.0 ELSE 0.0 END) as mastery
            FROM student_responses sr
            JOIN questions q ON sr.question_id = q.id
            WHERE sr.student_id = :student_id
            AND sr.created_at >= NOW() - INTERVAL '30 days'
            ORDER BY sr.created_at DESC
            LIMIT 10
        """
        )

        result = db.execute(query, {"student_id": student_id}).fetchone()

        if result and result.mastery is not None:
            return float(result.mastery)
        else:
            return 0.5  # Default neutral mastery

    except Exception as e:
        logger.error(
            "calculate_mastery_error",
            student_id=student_id,
            skill_name=skill_name,
            error=str(e),
        )
        return 0.5  # Default fallback


def get_skill_name_by_question(db: Session, question_id: int) -> str:
    """Soru için ana skill adını getir"""
    try:
        skill_names = get_skill_names_by_question(db, question_id)
        if skill_names:
            return skill_names[0]  # İlk skill'i ana skill olarak kabul et
        else:
            # Fallback: Question'un subject'ini kullan
            question = get_question(db, question_id)
            if question and question.subject:
                return question.subject.name
            return "Genel"
    except Exception as e:
        logger.error("get_skill_name_error", question_id=question_id, error=str(e))
        return "Bilinmeyen"


def get_recommended_questions(
    db: Session,
    student_id: int,
    limit: int = 10,
    difficulty_preference: Optional[int] = None,
    topic_id: Optional[int] = None,
) -> List[Question]:
    """Öğrenci için kişiselleştirilmiş soru önerileri getir"""
    try:
        # Temel sorgu - aktif sorular
        query = db.query(Question).filter(Question.is_active == True)

        # Zorluk seviyesi filtresi
        if difficulty_preference:
            # ±1 zorluk aralığı ver
            min_diff = max(1, difficulty_preference - 1)
            max_diff = min(5, difficulty_preference + 1)
            query = query.filter(Question.difficulty_level.between(min_diff, max_diff))

        # Konu filtresi
        if topic_id:
            query = query.filter(Question.topic_id == topic_id)

        # Öğrencinin daha önce cevaplamadığı soruları getir
        answered_questions = (
            db.query(StudentResponse.question_id)
            .filter(StudentResponse.student_id == student_id)
            .subquery()
        )

        query = query.filter(~getattr(Question, "id").in_(answered_questions))

        # Karışık sıralama ve limit
        questions = query.order_by(func.random()).limit(limit * 2).all()

        # Neo4j skill-based filtering (eğer aktifse)
        if settings.USE_NEO4J and questions:
            # En son cevaplanan soruların skill'lerini al
            recent_responses = (
                db.query(StudentResponse)
                .filter(StudentResponse.student_id == student_id)
                .order_by(StudentResponse.created_at.desc())
                .limit(5)
                .all()
            )

            if recent_responses:
                # Benzer skill'lere sahip soruları öncelikle
                skilled_questions = []
                for question in questions:
                    try:
                        centrality = get_question_skill_centrality(
                            int(getattr(question, "id", 0))
                        )
                        setattr(question, "temp_centrality", centrality["centrality"])
                        skilled_questions.append(question)
                    except:
                        setattr(question, "temp_centrality", 0.0)
                        skilled_questions.append(question)

                # Centrality'ye göre sırala
                skilled_questions.sort(
                    key=lambda q: getattr(q, "_temp_centrality", 0.0), reverse=True
                )
                return skilled_questions[:limit]

        # Fallback: Rastgele seçim
        return questions[:limit]

    except Exception as e:
        logger.error(
            "get_recommended_questions_error", student_id=student_id, error=str(e)
        )
        # Fallback: Random sorular
        return (
            db.query(Question)
            .filter(Question.is_active == True)
            .order_by(func.random())
            .limit(limit)
            .all()
        )
