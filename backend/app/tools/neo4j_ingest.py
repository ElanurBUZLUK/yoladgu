import os
from neo4j import GraphDatabase
from sqlalchemy import text
from app.core.database import get_db

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PWD  = os.getenv("NEO4J_PASSWORD")

def run():
    drv = GraphDatabase.driver(URI, auth=(USER, PWD))
    db = next(get_db())
    with drv.session() as s:
        # Grammar nodes
        rules = db.execute(text("SELECT id, code, title FROM grammar_rules")).fetchall()
        for rid, code, title in rules:
            s.run("MERGE (:Concept {obj_ref:$ref, namespace:'en_grammar', title:$t})", 
                  ref=f"grammar:{code}", t=title)
        
        # English text links
        texts = db.execute(text("SELECT id, title FROM english_texts")).fetchall()
        for tid, title in texts:
            s.run("MERGE (:Document {obj_ref:$ref, namespace:'en_texts', title:$t})", 
                  ref=f"text:{tid}", t=title)
        
        # Sample relations
        s.run("""
            MATCH (c:Concept {namespace:'en_grammar'}), (d:Document {namespace:'en_texts'})
            WITH c,d LIMIT 100
            MERGE (c)-[:OCCURS_IN {strength:0.5}]->(d)
        """)
    drv.close()

if __name__ == "__main__":
    run()