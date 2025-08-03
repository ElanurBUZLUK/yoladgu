from typing import Optional, List
from sqlalchemy.orm import Session
from app.db.models import StudentResponse
from app.core.config import settings
from neo4j import GraphDatabase
import structlog

logger = structlog.get_logger()

def _get_neo4j_driver():
    """Neo4j driver instance'ı döndür"""
    if not settings.USE_NEO4J:
        return None
    # Use the centralized Neo4j service instead of creating multiple drivers
    from app.services.neo4j_service import neo4j_service
    return neo4j_service.driver

def _sync_student_response_to_neo4j(student_id: int, question_id: int, is_correct: bool, response_time: float = None):
    """Öğrenci cevabını Neo4j'e senkronize et"""
    if not settings.USE_NEO4J:
        return
    
    driver = _get_neo4j_driver()
    if not driver:
        return
    
    try:
        with driver.session() as session:
            # Öğrenci ve soru node'larını oluştur
            session.run("MERGE (u:User {id: $student_id})", student_id=student_id)
            session.run("MERGE (q:Question {id: $question_id})", question_id=question_id)
            
            # Öğrenci-soru ilişkisini oluştur
            session.run(
                """
                MATCH (u:User {id: $student_id})
                MATCH (q:Question {id: $question_id})
                MERGE (u)-[r:SOLVED]->(q)
                SET r.correct = $is_correct,
                    r.response_time = $response_time,
                    r.timestamp = datetime()
                """,
                student_id=student_id,
                question_id=question_id,
                is_correct=is_correct,
                response_time=response_time
            )
        
        logger.info("student_response_synced_to_neo4j", 
                   student_id=student_id, question_id=question_id, is_correct=is_correct)
    except Exception as e:
        logger.error("neo4j_student_response_sync_error", 
                    student_id=student_id, question_id=question_id, error=str(e))
    finally:
        # Driver managed by singleton service

def get_response(db: Session, response_id: int) -> Optional[StudentResponse]:
    return db.query(StudentResponse).filter(StudentResponse.id == response_id).first()

def get_responses(db: Session, student_id: Optional[int] = None, skip: int = 0, limit: int = 100) -> List[StudentResponse]:
    query = db.query(StudentResponse)
    if student_id is not None:
        query = query.filter(StudentResponse.student_id == student_id)
    return query.offset(skip).limit(limit).all()

def create_response(
    db: Session,
    *,
    student_id: int,
    question_id: int,
    answer: str,
    is_correct: bool,
    response_time: Optional[float] = None,
    confidence_level: Optional[int] = None,
    feedback: Optional[str] = None,
) -> StudentResponse:
    db_response = StudentResponse(
        student_id=student_id,
        question_id=question_id,
        answer=answer,
        is_correct=is_correct,
        response_time=response_time,
        confidence_level=confidence_level,
        feedback=feedback,
    )
    db.add(db_response)
    db.commit()
    db.refresh(db_response)
    
    # Neo4j'e senkronize et
    _sync_student_response_to_neo4j(student_id, question_id, is_correct, response_time)
    
    return db_response

def delete_response(db: Session, response_id: int) -> Optional[StudentResponse]:
    db_obj = db.query(StudentResponse).get(response_id)
    if db_obj:
        db.delete(db_obj)
        db.commit()
        
        # Neo4j'den ilişkiyi kaldır
        if settings.USE_NEO4J:
            driver = _get_neo4j_driver()
            if driver:
                try:
                    with driver.session() as session:
                        session.run(
                            """
                            MATCH (u:User {id: $student_id})-[r:SOLVED]->(q:Question {id: $question_id})
                            DELETE r
                            """,
                            student_id=db_obj.student_id,
                            question_id=db_obj.question_id
                        )
                except Exception as e:
                    logger.error("neo4j_delete_response_error", response_id=response_id, error=str(e))
                finally:
                    # Driver managed by singleton service
    
    return db_obj

def get_student_skill_mastery_from_neo4j(student_id: int) -> dict:
    """Neo4j'den öğrencinin skill mastery skorlarını getir"""
    if not settings.USE_NEO4J:
        return {}
    
    driver = _get_neo4j_driver()
    if not driver:
        return {}
    
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (u:User {id: $student_id})-[:SOLVED]->(q:Question)-[:HAS_SKILL]->(s:Skill)
                WITH s, COUNT(*) AS total_attempts,
                     SUM(CASE WHEN q.correct THEN 1 ELSE 0 END) AS correct_attempts
                RETURN s.id AS skill_id, 
                       s.name AS skill_name,
                       correct_attempts * 1.0 / total_attempts AS mastery_score,
                       total_attempts AS attempt_count
                ORDER BY mastery_score DESC
                """,
                student_id=student_id
            )
            return {
                record["skill_id"]: {
                    "name": record["skill_name"],
                    "mastery": record["mastery_score"],
                    "attempts": record["attempt_count"]
                }
                for record in result
            }
    except Exception as e:
        logger.error("neo4j_skill_mastery_error", student_id=student_id, error=str(e))
        return {}
    finally:
        # Driver managed by singleton service

def get_student_learning_path_from_neo4j(student_id: int, limit: int = 20) -> list[dict]:
    """Neo4j'den öğrencinin öğrenme yolunu getir"""
    if not settings.USE_NEO4J:
        return []
    
    driver = _get_neo4j_driver()
    if not driver:
        return []
    
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (u:User {id: $student_id})-[r:SOLVED]->(q:Question)
                WITH q, r ORDER BY r.timestamp ASC
                LIMIT $limit
                RETURN q.id AS question_id,
                       q.text AS question_text,
                       r.correct AS is_correct,
                       r.response_time AS response_time,
                       r.timestamp AS timestamp
                """,
                student_id=student_id, limit=limit
            )
            return [
                {
                    "question_id": record["question_id"],
                    "question_text": record["question_text"],
                    "is_correct": record["is_correct"],
                    "response_time": record["response_time"],
                    "timestamp": record["timestamp"]
                }
                for record in result
            ]
    except Exception as e:
        logger.error("neo4j_learning_path_error", student_id=student_id, error=str(e))
        return []
    finally:
        # Driver managed by singleton service 