"""
Neo4j Service for Graph Database Operations
Handles skill relationships, question similarity, and learning paths
"""

from typing import List, Optional
import structlog
from neo4j import GraphDatabase
from app.core.config import settings

logger = structlog.get_logger()

class Neo4jService:
    """Singleton service for Neo4j graph database operations"""
    
    _instance = None
    _driver = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super(Neo4jService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Neo4j connection (only once)"""
        if self._initialized:
            return
            
        if settings.USE_NEO4J:
            try:
                self._driver = GraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
                )
                # Test connection
                with self._driver.session() as session:
                    session.run("RETURN 1")
                logger.info("neo4j_connection_established", uri=settings.NEO4J_URI)
            except Exception as e:
                logger.error("neo4j_connection_failed", error=str(e))
                self._driver = None
        else:
            self._driver = None
            logger.info("neo4j_disabled_in_config")
            
        self._initialized = True
    
    @property
    def driver(self):
        """Get the Neo4j driver instance"""
        return self._driver

    def close(self):
        """Close Neo4j connection"""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("neo4j_connection_closed")

    def add_question_skills(self, question_id: int, skill_ids: List[int]):
        """Add skill relationships to a question"""
        if not self._driver:
            logger.warning("neo4j_not_available", operation="add_question_skills")
            return
        
        try:
            with self._driver.session() as session:
                for skill_id in skill_ids:
                    session.run(
                        """
                        MERGE (q:Question {id: $qid})
                        MERGE (s:Skill {id: $sid})
                        MERGE (q)-[:HAS_SKILL]->(s)
                        """,
                        qid=question_id, sid=skill_id
                    )
                logger.info("question_skills_added", 
                          question_id=question_id, 
                          skill_count=len(skill_ids))
        except Exception as e:
            logger.error("add_question_skills_error", 
                        question_id=question_id, 
                        error=str(e))

    def record_student_solution(self, student_id: int, question_id: int, correct: bool):
        """Record a student's solution attempt"""
        if not self._driver:
            logger.warning("neo4j_not_available", operation="record_student_solution")
            return
        
        try:
            with self._driver.session() as session:
                session.run(
                    """
                    MERGE (u:User {id: $uid})
                    MERGE (q:Question {id: $qid})
                    MERGE (u)-[r:SOLVED]->(q)
                    SET r.correct = $correct, r.timestamp = timestamp()
                    """,
                    uid=student_id, qid=question_id, correct=correct
                )
                logger.info("student_solution_recorded", 
                          student_id=student_id, 
                          question_id=question_id, 
                          correct=correct)
        except Exception as e:
            logger.error("record_student_solution_error", 
                        student_id=student_id, 
                        question_id=question_id, 
                        error=str(e))

    def get_similar_questions(self, question_id: int, min_shared_skills: int = 1, limit: int = 10) -> List[int]:
        """Get questions similar to given question based on shared skills"""
        if not self._driver:
            logger.warning("neo4j_not_available", operation="get_similar_questions")
            return []
        
        try:
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (q1:Question {id: $qid})-[:HAS_SKILL]->(s:Skill)<-[:HAS_SKILL]-(q2:Question)
                    WHERE q1 <> q2
                    WITH q2, count(s) AS shared
                    WHERE shared >= $min_shared_skills
                    RETURN q2.id AS id, shared
                    ORDER BY shared DESC
                    LIMIT $limit
                    """,
                    qid=question_id, 
                    min_shared_skills=min_shared_skills,
                    limit=limit
                )
                similar_questions = [record["id"] for record in result]
                logger.info("similar_questions_found", 
                          question_id=question_id, 
                          count=len(similar_questions))
                return similar_questions
        except Exception as e:
            logger.error("get_similar_questions_error", 
                        question_id=question_id, 
                        error=str(e))
            return []

    def get_student_skill_gaps(self, student_id: int) -> List[dict]:
        """Identify skill gaps for a student based on incorrect answers"""
        if not self._driver:
            logger.warning("neo4j_not_available", operation="get_student_skill_gaps")
            return []
        
        try:
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (u:User {id: $uid})-[r:SOLVED {correct: false}]->(q:Question)-[:HAS_SKILL]->(s:Skill)
                    WITH s, count(r) as wrong_count
                    MATCH (s)<-[:HAS_SKILL]-(all_q:Question)
                    WITH s, wrong_count, count(all_q) as total_questions
                    WHERE wrong_count > 0
                    RETURN s.id as skill_id, s.name as skill_name, 
                           wrong_count, total_questions,
                           (wrong_count * 1.0 / total_questions) as error_rate
                    ORDER BY error_rate DESC
                    LIMIT 10
                    """,
                    uid=student_id
                )
                gaps = []
                for record in result:
                    gaps.append({
                        "skill_id": record["skill_id"],
                        "skill_name": record["skill_name"],
                        "wrong_count": record["wrong_count"],
                        "total_questions": record["total_questions"],
                        "error_rate": record["error_rate"]
                    })
                logger.info("skill_gaps_identified", 
                          student_id=student_id, 
                          gap_count=len(gaps))
                return gaps
        except Exception as e:
            logger.error("get_student_skill_gaps_error", 
                        student_id=student_id, 
                        error=str(e))
            return []

    def get_learning_path(self, student_id: int, target_skill_id: int) -> List[int]:
        """Get recommended learning path for a student to master a skill"""
        if not self._driver:
            logger.warning("neo4j_not_available", operation="get_learning_path")
            return []
        
        try:
            with self._driver.session() as session:
                # Find prerequisite skills and questions
                result = session.run(
                    """
                    MATCH (target:Skill {id: $target_skill_id})
                    OPTIONAL MATCH (prereq:Skill)-[:PREREQUISITE]->(target)
                    OPTIONAL MATCH (u:User {id: $uid})-[solved:SOLVED]->(q:Question)-[:HAS_SKILL]->(prereq)
                    WITH target, prereq, 
                         CASE WHEN solved IS NULL THEN false ELSE solved.correct END as mastered
                    WHERE prereq IS NULL OR mastered = true
                    MATCH (target)<-[:HAS_SKILL]-(questions:Question)
                    OPTIONAL MATCH (u:User {id: $uid})-[attempted:SOLVED]->(questions)
                    WHERE attempted IS NULL OR attempted.correct = false
                    RETURN questions.id as question_id
                    ORDER BY questions.difficulty_level ASC
                    LIMIT 5
                    """,
                    uid=student_id, 
                    target_skill_id=target_skill_id
                )
                path = [record["question_id"] for record in result]
                logger.info("learning_path_generated", 
                          student_id=student_id, 
                          target_skill_id=target_skill_id, 
                          path_length=len(path))
                return path
        except Exception as e:
            logger.error("get_learning_path_error", 
                        student_id=student_id, 
                        target_skill_id=target_skill_id, 
                        error=str(e))
            return []

    def create_skill_prerequisites(self, skill_id: int, prerequisite_ids: List[int]):
        """Create prerequisite relationships between skills"""
        if not self._driver:
            logger.warning("neo4j_not_available", operation="create_skill_prerequisites")
            return
        
        try:
            with self._driver.session() as session:
                for prereq_id in prerequisite_ids:
                    session.run(
                        """
                        MERGE (s1:Skill {id: $prereq_id})
                        MERGE (s2:Skill {id: $skill_id})
                        MERGE (s1)-[:PREREQUISITE]->(s2)
                        """,
                        prereq_id=prereq_id, 
                        skill_id=skill_id
                    )
                logger.info("skill_prerequisites_created", 
                          skill_id=skill_id, 
                          prerequisite_count=len(prerequisite_ids))
        except Exception as e:
            logger.error("create_skill_prerequisites_error", 
                        skill_id=skill_id, 
                        error=str(e))

    def get_skill_centrality(self, limit: int = 20) -> List[dict]:
        """Get skills ordered by their centrality in the skill graph"""
        if not self._driver:
            logger.warning("neo4j_not_available", operation="get_skill_centrality")
            return []
        
        try:
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (s:Skill)
                    OPTIONAL MATCH (s)-[:PREREQUISITE]->(dependent:Skill)
                    OPTIONAL MATCH (prerequisite:Skill)-[:PREREQUISITE]->(s)
                    WITH s, count(dependent) as dependents, count(prerequisite) as prerequisites
                    RETURN s.id as skill_id, s.name as skill_name,
                           dependents, prerequisites,
                           (dependents + prerequisites) as centrality
                    ORDER BY centrality DESC
                    LIMIT $limit
                    """,
                    limit=limit
                )
                skills = []
                for record in result:
                    skills.append({
                        "skill_id": record["skill_id"],
                        "skill_name": record["skill_name"],
                        "dependents": record["dependents"],
                        "prerequisites": record["prerequisites"],
                        "centrality": record["centrality"]
                    })
                logger.info("skill_centrality_calculated", skill_count=len(skills))
                return skills
        except Exception as e:
            logger.error("get_skill_centrality_error", error=str(e))
            return []


# Global Neo4j service instance
neo4j_service = Neo4jService()