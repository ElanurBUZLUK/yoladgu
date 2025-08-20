from neo4j import GraphDatabase
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class GraphClient:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USER")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.enabled = all([self.uri, self.user, self.password])
        
        if self.enabled:
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                logger.info("Neo4j graph client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Neo4j driver: {e}")
                self.enabled = False
        else:
            logger.warning("Neo4j configuration incomplete. Graph features disabled.")
            self.driver = None

    def neighbors(self, obj_ref: str, max_neighbors: int = 3) -> List[Dict[str, Any]]:
        if not self.enabled or not self.driver:
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (start {obj_ref: $obj_ref})-[r]-(neighbor)
                    RETURN neighbor.obj_ref AS obj_ref,
                           neighbor.namespace AS namespace,
                           type(r) AS relationship,
                           r.strength AS strength
                    ORDER BY r.strength DESC
                    LIMIT $limit
                    """,
                    obj_ref=obj_ref,
                    limit=max_neighbors
                )
                return [
                    {
                        "obj_ref": record["obj_ref"],
                        "meta": {
                            "namespace": record["namespace"],
                            "relationship": record["relationship"],
                            "strength": record["strength"]
                        },
                        "src": "graph"
                    }
                    for record in result
                ]
        except Exception as e:
            logger.error(f"Graph query failed for {obj_ref}: {e}")
            return []