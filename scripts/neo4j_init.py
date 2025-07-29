#!/usr/bin/env python3
"""
Neo4j veritabanını başlatma ve örnek veri yükleme scripti
"""

import os
import sys
from neo4j import GraphDatabase
from app.core.config import settings
import structlog

logger = structlog.get_logger()

def create_constraints():
    """Neo4j'de constraint'ler oluştur"""
    driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Unique constraint'ler
            session.run("CREATE CONSTRAINT question_id IF NOT EXISTS ON (q:Question) ASSERT q.id IS UNIQUE")
            session.run("CREATE CONSTRAINT user_id IF NOT EXISTS ON (u:User) ASSERT u.id IS UNIQUE")
            session.run("CREATE CONSTRAINT skill_id IF NOT EXISTS ON (s:Skill) ASSERT s.id IS UNIQUE")
            session.run("CREATE CONSTRAINT subject_id IF NOT EXISTS ON (sub:Subject) ASSERT sub.id IS UNIQUE")
            
            # Index'ler
            session.run("CREATE INDEX question_text IF NOT EXISTS FOR (q:Question) ON (q.text)")
            session.run("CREATE INDEX skill_name IF NOT EXISTS FOR (s:Skill) ON (s.name)")
            
        logger.info("neo4j_constraints_created")
    except Exception as e:
        logger.error("neo4j_constraint_error", error=str(e))
    finally:
        driver.close()

def create_sample_data():
    """Örnek veri oluştur"""
    driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Örnek dersler
            subjects = [
                {"id": 1, "name": "Matematik", "description": "Temel matematik konuları"},
                {"id": 2, "name": "Fizik", "description": "Temel fizik konuları"},
                {"id": 3, "name": "Kimya", "description": "Temel kimya konuları"}
            ]
            
            for subject in subjects:
                session.run(
                    "MERGE (s:Subject {id: $id}) SET s.name = $name, s.description = $description",
                    id=subject["id"], name=subject["name"], description=subject["description"]
                )
            
            # Örnek beceriler
            skills = [
                {"id": 1, "name": "Cebir", "subject_id": 1, "difficulty": 1},
                {"id": 2, "name": "Geometri", "subject_id": 1, "difficulty": 2},
                {"id": 3, "name": "Trigonometri", "subject_id": 1, "difficulty": 3},
                {"id": 4, "name": "Mekanik", "subject_id": 2, "difficulty": 1},
                {"id": 5, "name": "Elektrik", "subject_id": 2, "difficulty": 2},
                {"id": 6, "name": "Organik Kimya", "subject_id": 3, "difficulty": 2}
            ]
            
            for skill in skills:
                session.run(
                    """
                    MERGE (s:Skill {id: $id}) 
                    SET s.name = $name, s.difficulty = $difficulty
                    WITH s
                    MATCH (sub:Subject {id: $subject_id})
                    MERGE (s)-[:BELONGS_TO]->(sub)
                    """,
                    id=skill["id"], name=skill["name"], 
                    difficulty=skill["difficulty"], subject_id=skill["subject_id"]
                )
            
            # Örnek sorular
            questions = [
                {"id": 1, "text": "2x + 3 = 7 denklemini çözün", "subject_id": 1},
                {"id": 2, "text": "Bir üçgenin alanını hesaplayın", "subject_id": 1},
                {"id": 3, "text": "Kuvvet ve ivme arasındaki ilişki nedir?", "subject_id": 2},
                {"id": 4, "text": "Kimyasal bağ türlerini açıklayın", "subject_id": 3}
            ]
            
            for question in questions:
                session.run(
                    "MERGE (q:Question {id: $id}) SET q.text = $text",
                    id=question["id"], text=question["text"]
                )
            
            # Soru-beceri ilişkileri
            question_skills = [
                (1, 1, 0.8),  # Soru 1 - Cebir
                (2, 2, 0.9),  # Soru 2 - Geometri
                (3, 4, 0.7),  # Soru 3 - Mekanik
                (4, 6, 0.6)   # Soru 4 - Organik Kimya
            ]
            
            for qid, sid, weight in question_skills:
                session.run(
                    """
                    MATCH (q:Question {id: $qid})
                    MATCH (s:Skill {id: $sid})
                    MERGE (q)-[r:HAS_SKILL]->(s)
                    SET r.weight = $weight
                    """,
                    qid=qid, sid=sid, weight=weight
                )
            
            # Örnek öğrenciler
            students = [
                {"id": 1, "name": "Ahmet Yılmaz"},
                {"id": 2, "name": "Ayşe Demir"},
                {"id": 3, "name": "Mehmet Kaya"}
            ]
            
            for student in students:
                session.run(
                    "MERGE (u:User {id: $id}) SET u.name = $name",
                    id=student["id"], name=student["name"]
                )
            
            # Örnek öğrenci cevapları
            student_responses = [
                (1, 1, True, 45.2),   # Ahmet - Soru 1 - Doğru
                (1, 2, False, 120.5), # Ahmet - Soru 2 - Yanlış
                (2, 1, True, 38.1),   # Ayşe - Soru 1 - Doğru
                (2, 3, True, 85.3),   # Ayşe - Soru 3 - Doğru
                (3, 4, False, 200.1)  # Mehmet - Soru 4 - Yanlış
            ]
            
            for student_id, question_id, is_correct, response_time in student_responses:
                session.run(
                    """
                    MATCH (u:User {id: $student_id})
                    MATCH (q:Question {id: $question_id})
                    MERGE (u)-[r:SOLVED]->(q)
                    SET r.correct = $is_correct,
                        r.response_time = $response_time,
                        r.timestamp = datetime()
                    """,
                    student_id=student_id, question_id=question_id,
                    is_correct=is_correct, response_time=response_time
                )
            
        logger.info("neo4j_sample_data_created")
    except Exception as e:
        logger.error("neo4j_sample_data_error", error=str(e))
    finally:
        driver.close()

def create_similarity_relationships():
    """Soru benzerlik ilişkilerini oluştur"""
    driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Ortak skill'lere sahip sorular arasında benzerlik ilişkisi kur
            session.run(
                """
                MATCH (q1:Question)-[:HAS_SKILL]->(s:Skill)<-[:HAS_SKILL]-(q2:Question)
                WHERE q1.id < q2.id
                WITH q1, q2, COUNT(DISTINCT s) AS shared_skills
                WHERE shared_skills > 0
                MERGE (q1)-[r:SIMILAR]->(q2)
                SET r.weight = shared_skills
                """
            )
            
        logger.info("neo4j_similarity_relationships_created")
    except Exception as e:
        logger.error("neo4j_similarity_error", error=str(e))
    finally:
        driver.close()

def verify_setup():
    """Kurulumu doğrula"""
    driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Node sayılarını kontrol et
            result = session.run("MATCH (n) RETURN labels(n) as labels, count(*) as count")
            for record in result:
                logger.info("neo4j_node_count", labels=record["labels"], count=record["count"])
            
            # İlişki sayılarını kontrol et
            result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(*) as count")
            for record in result:
                logger.info("neo4j_relationship_count", type=record["type"], count=record["count"])
                
    except Exception as e:
        logger.error("neo4j_verification_error", error=str(e))
    finally:
        driver.close()

def main():
    """Ana fonksiyon"""
    logger.info("neo4j_initialization_started")
    
    try:
        # 1. Constraint'leri oluştur
        create_constraints()
        
        # 2. Örnek veriyi yükle
        create_sample_data()
        
        # 3. Benzerlik ilişkilerini oluştur
        create_similarity_relationships()
        
        # 4. Kurulumu doğrula
        verify_setup()
        
        logger.info("neo4j_initialization_completed")
        print("✅ Neo4j başarıyla başlatıldı ve örnek veri yüklendi!")
        
    except Exception as e:
        logger.error("neo4j_initialization_failed", error=str(e))
        print(f"❌ Neo4j başlatma hatası: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 