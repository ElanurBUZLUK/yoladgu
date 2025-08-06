"""
Neo4j Service
Graph database işlemleri için Neo4j servisi
"""

import structlog
from typing import Dict, List, Optional, Any
from neo4j import GraphDatabase
from app.core.config import settings

logger = structlog.get_logger()


class Neo4jService:
    """Neo4j graph database servisi"""
    
    def __init__(self):
        self.driver = None
        self.initialized = False
        
    async def initialize(self):
        """Neo4j servisini başlat"""
        try:
            # Neo4j driver'ı oluştur
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI or "bolt://localhost:7687",
                auth=(
                    settings.NEO4J_USER or "neo4j",
                    settings.NEO4J_PASSWORD or "password"
                )
            )
            
            # Bağlantıyı test et
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            
            self.initialized = True
            logger.info("neo4j_service_initialized")
            
        except Exception as e:
            logger.error("neo4j_service_initialization_error", error=str(e))
            raise

    # Generic Node CRUD Operations
    async def create_node(self, label: str, node_id: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Generic node oluştur"""
        try:
            with self.driver.session() as session:
                query = f"""
                MERGE (n:{label} {{id: $node_id}})
                SET n += $properties
                RETURN n
                """
                result = session.run(query, node_id=node_id, properties=properties)
                node = result.single()
                return dict(node["n"]) if node else {}
                
        except Exception as e:
            logger.error("create_node_error", label=label, node_id=node_id, error=str(e))
            raise

    async def update_node(self, label: str, node_id: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Node güncelle"""
        try:
            with self.driver.session() as session:
                query = f"""
                MATCH (n:{label} {{id: $node_id}})
                SET n += $properties
                RETURN n
                """
                result = session.run(query, node_id=node_id, properties=properties)
                node = result.single()
                return dict(node["n"]) if node else {}
                
        except Exception as e:
            logger.error("update_node_error", label=label, node_id=node_id, error=str(e))
            raise

    async def delete_node(self, label: str, node_id: str) -> bool:
        """Node sil"""
        try:
            with self.driver.session() as session:
                query = f"""
                MATCH (n:{label} {{id: $node_id}})
                DETACH DELETE n
                RETURN count(n) as deleted
                """
                result = session.run(query, node_id=node_id)
                deleted = result.single()
                return deleted["deleted"] > 0
                
        except Exception as e:
            logger.error("delete_node_error", label=label, node_id=node_id, error=str(e))
            raise

    # Generic Edge CRUD Operations
    async def create_edge(
        self, 
        src_label: str, 
        src_id: str, 
        relationship: str, 
        dst_label: str, 
        dst_id: str, 
        properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Edge oluştur"""
        try:
            with self.driver.session() as session:
                query = f"""
                MATCH (a:{src_label} {{id: $src_id}})
                MATCH (b:{dst_label} {{id: $dst_id}})
                MERGE (a)-[r:{relationship}]->(b)
                SET r += $properties
                RETURN r
                """
                result = session.run(query, 
                                  src_id=src_id, 
                                  dst_id=dst_id, 
                                  properties=properties or {})
                edge = result.single()
                return dict(edge["r"]) if edge else {}
                
        except Exception as e:
            logger.error("create_edge_error", 
                        src_label=src_label, src_id=src_id,
                        relationship=relationship,
                        dst_label=dst_label, dst_id=dst_id,
                        error=str(e))
            raise

    async def update_edge(
        self, 
        src_label: str, 
        src_id: str, 
        relationship: str, 
        dst_label: str, 
        dst_id: str, 
        properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Edge güncelle"""
        try:
            with self.driver.session() as session:
                query = f"""
                MATCH (a:{src_label} {{id: $src_id}})-[r:{relationship}]->(b:{dst_label} {{id: $dst_id}})
                SET r += $properties
                RETURN r
                """
                result = session.run(query, 
                                  src_id=src_id, 
                                  dst_id=dst_id, 
                                  properties=properties)
                edge = result.single()
                return dict(edge["r"]) if edge else {}
                
        except Exception as e:
            logger.error("update_edge_error", 
                        src_label=src_label, src_id=src_id,
                        relationship=relationship,
                        dst_label=dst_label, dst_id=dst_id,
                        error=str(e))
            raise

    async def delete_edge(
        self, 
        src_label: str, 
        src_id: str, 
        relationship: str, 
        dst_label: str, 
        dst_id: str
    ) -> bool:
        """Edge sil"""
        try:
            with self.driver.session() as session:
                query = f"""
                MATCH (a:{src_label} {{id: $src_id}})-[r:{relationship}]->(b:{dst_label} {{id: $dst_id}})
                DELETE r
                RETURN count(r) as deleted
                """
                result = session.run(query, src_id=src_id, dst_id=dst_id)
                deleted = result.single()
                return deleted["deleted"] > 0
                
        except Exception as e:
            logger.error("delete_edge_error", 
                        src_label=src_label, src_id=src_id,
                        relationship=relationship,
                        dst_label=dst_label, dst_id=dst_id,
                        error=str(e))
            raise

    # Graph Queries
    async def get_student_question_subgraph(self, user_id: int) -> Dict[str, Any]:
        """Öğrenci → soru alt grafı çıkarma"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (u:User {user_id: $user_id})-[r:ANSWERED]->(q:Question)
                OPTIONAL MATCH (q)-[:SIMILAR_TO]->(similar:Question)
                RETURN u, r, q, similar
                """
                result = session.run(query, user_id=user_id)
                nodes = []
                relationships = []
                
                for record in result:
                    if record["u"]:
                        nodes.append({"type": "User", "data": dict(record["u"])})
                    if record["q"]:
                        nodes.append({"type": "Question", "data": dict(record["q"])})
                    if record["similar"]:
                        nodes.append({"type": "Question", "data": dict(record["similar"])})
                    if record["r"]:
                        relationships.append(dict(record["r"]))
                
                return {
                    "nodes": nodes,
                    "relationships": relationships,
                    "user_id": user_id
                }
                
        except Exception as e:
            logger.error("get_student_question_subgraph_error", user_id=user_id, error=str(e))
            raise

    async def get_similar_question_clusters(self, similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Benzer soru kümeleri (node similarity)"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (q1:Question)-[r:SIMILAR_TO]->(q2:Question)
                WHERE r.similarity >= $threshold
                WITH q1, q2, r.similarity as similarity
                ORDER BY similarity DESC
                RETURN q1.question_id as q1_id, q2.question_id as q2_id, similarity
                """
                result = session.run(query, threshold=similarity_threshold)
                
                clusters = []
                processed = set()
                
                for record in result:
                    q1_id = record["q1_id"]
                    q2_id = record["q2_id"]
                    similarity = record["similarity"]
                    
                    # Check if either question is already in a cluster
                    found_cluster = None
                    for cluster in clusters:
                        if q1_id in cluster["questions"] or q2_id in cluster["questions"]:
                            found_cluster = cluster
                            break
                    
                    if found_cluster:
                        # Add to existing cluster
                        if q1_id not in found_cluster["questions"]:
                            found_cluster["questions"].append(q1_id)
                        if q2_id not in found_cluster["questions"]:
                            found_cluster["questions"].append(q2_id)
                        found_cluster["similarities"].append(similarity)
                    else:
                        # Create new cluster
                        clusters.append({
                            "questions": [q1_id, q2_id],
                            "similarities": [similarity],
                            "avg_similarity": similarity
                        })
                
                # Calculate average similarity for each cluster
                for cluster in clusters:
                    cluster["avg_similarity"] = sum(cluster["similarities"]) / len(cluster["similarities"])
                
                return clusters
                
        except Exception as e:
            logger.error("get_similar_question_clusters_error", error=str(e))
            raise

    # Transaction Management
    async def execute_transaction(self, operations: List[Dict[str, Any]]) -> bool:
        """Transaction yönetimi ve error handling"""
        try:
            with self.driver.session() as session:
                with session.begin_transaction() as tx:
                    for operation in operations:
                        op_type = operation.get("type")
                        query = operation.get("query")
                        parameters = operation.get("parameters", {})
                        
                        if op_type == "read":
                            result = tx.run(query, **parameters)
                            operation["result"] = [dict(record) for record in result]
                        elif op_type == "write":
                            result = tx.run(query, **parameters)
                            operation["result"] = result.consume()
                        else:
                            raise ValueError(f"Unknown operation type: {op_type}")
                    
                    # Commit transaction
                    tx.commit()
                    logger.info("transaction_executed", operations_count=len(operations))
                    return True
                    
        except Exception as e:
            logger.error("transaction_error", error=str(e), operations=operations)
            # Transaction will be automatically rolled back
            raise

    # Legacy methods for backward compatibility
    async def create_user_node(self, user_id: int, user_data: Dict[str, Any]):
        """Kullanıcı node'u oluştur"""
        return await self.create_node("User", str(user_id), {"user_id": user_id, **user_data})
    
    async def create_question_node(self, question_id: int, question_data: Dict[str, Any]):
        """Soru node'u oluştur"""
        return await self.create_node("Question", str(question_id), {"question_id": question_id, **question_data})
    
    async def create_answered_relationship(
        self, 
        user_id: int, 
        question_id: int, 
        answer_data: Dict[str, Any]
    ):
        """Kullanıcı-soru ilişkisi oluştur"""
        return await self.create_edge(
            "User", str(user_id), "ANSWERED", "Question", str(question_id), answer_data
        )
    
    async def get_user_recommendations(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Kullanıcı için öneriler getir"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (u:User {user_id: $user_id})-[r:ANSWERED]->(q:Question)
                WITH u, q, r
                MATCH (q)-[:SIMILAR_TO]->(rec:Question)
                WHERE NOT EXISTS((u)-[:ANSWERED]->(rec))
                RETURN rec.question_id as question_id, 
                       rec.content as content,
                       rec.difficulty_level as difficulty_level,
                       count(r) as similarity_score
                ORDER BY similarity_score DESC
                LIMIT $limit
                """
                result = session.run(query, user_id=user_id, limit=limit)
                return [dict(record) for record in result]
                
        except Exception as e:
            logger.error("get_user_recommendations_error", error=str(e))
            return []
    
    async def get_user_progress(self, user_id: int) -> Dict[str, Any]:
        """Kullanıcı ilerlemesini getir"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (u:User {user_id: $user_id})-[r:ANSWERED]->(q:Question)
                RETURN count(r) as total_answered,
                       sum(CASE WHEN r.is_correct THEN 1 ELSE 0 END) as correct_answers,
                       avg(r.response_time) as avg_response_time
                """
                result = session.run(query, user_id=user_id)
                record = result.single()
                
                if record:
                    total_answered = record["total_answered"]
                    correct_answers = record["correct_answers"]
                    accuracy = (correct_answers / total_answered * 100) if total_answered > 0 else 0
                    
                    return {
                        "total_answered": total_answered,
                        "correct_answers": correct_answers,
                        "accuracy": accuracy,
                        "avg_response_time": record["avg_response_time"] or 0
                    }
                else:
                    return {
                        "total_answered": 0,
                        "correct_answers": 0,
                        "accuracy": 0,
                        "avg_response_time": 0
                    }
                    
        except Exception as e:
            logger.error("get_user_progress_error", error=str(e))
            return {
                "total_answered": 0,
                "correct_answers": 0,
                "accuracy": 0,
                "avg_response_time": 0
            }
    
    async def get_similar_questions(self, question_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """Benzer soruları getir"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (q:Question {question_id: $question_id})-[:SIMILAR_TO]->(similar:Question)
                RETURN similar.question_id as question_id,
                       similar.content as content,
                       similar.difficulty_level as difficulty_level
                LIMIT $limit
                """
                result = session.run(query, question_id=question_id, limit=limit)
                return [dict(record) for record in result]
                
        except Exception as e:
            logger.error("get_similar_questions_error", error=str(e))
            return []
    
    async def create_similarity_relationship(self, question1_id: int, question2_id: int, similarity_score: float):
        """Soru benzerlik ilişkisi oluştur"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (q1:Question {question_id: $question1_id})
                MATCH (q2:Question {question_id: $question2_id})
                MERGE (q1)-[r:SIMILAR_TO]->(q2)
                SET r.similarity_score = $similarity_score
                RETURN r
                """
                result = session.run(query, 
                                  question1_id=question1_id,
                                  question2_id=question2_id,
                                  similarity_score=similarity_score)
                return result.single()
                
        except Exception as e:
            logger.error("create_similarity_relationship_error", error=str(e))
            raise
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Graph istatistiklerini getir"""
        try:
            with self.driver.session() as session:
                # Node sayıları
                user_count = session.run("MATCH (u:User) RETURN count(u) as count").single()["count"]
                question_count = session.run("MATCH (q:Question) RETURN count(q) as count").single()["count"]
                
                # Relationship sayıları
                answered_count = session.run("MATCH ()-[r:ANSWERED]->() RETURN count(r) as count").single()["count"]
                similar_count = session.run("MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) as count").single()["count"]
                
                return {
                    "users_count": user_count,
                    "questions_count": question_count,
                    "answered_relationships": answered_count,
                    "similarity_relationships": similar_count,
                    "initialized": self.initialized
                }
                
        except Exception as e:
            logger.error("get_graph_stats_error", error=str(e))
            return {
                "users_count": 0,
                "questions_count": 0,
                "answered_relationships": 0,
                "similarity_relationships": 0,
                "initialized": self.initialized
            }
    
    async def is_healthy(self) -> bool:
        """Neo4j sağlık kontrolü"""
        try:
            if not self.initialized or not self.driver:
                return False
            
            with self.driver.session() as session:
                result = session.run("RETURN 1 as health_check")
                result.single()
                return True
                
        except Exception as e:
            logger.error("neo4j_health_check_error", error=str(e))
            return False
    
    async def cleanup(self):
        """Neo4j servisini temizle"""
        try:
            if self.driver:
                self.driver.close()
                self.initialized = False
            
            logger.info("neo4j_service_cleanup_completed")
            
        except Exception as e:
            logger.error("neo4j_cleanup_error", error=str(e))


# Global instance
neo4j_service = Neo4jService() 