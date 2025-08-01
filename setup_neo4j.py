#!/usr/bin/env python3
"""
Setup script for Neo4j skill graph database
"""

from neo4j import GraphDatabase
import os

class Neo4jSetup:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def setup_constraints(self):
        """Create constraints and indexes"""
        with self.driver.session() as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT skill_id IF NOT EXISTS FOR (s:Skill) REQUIRE s.id IS UNIQUE",
                "CREATE CONSTRAINT question_id IF NOT EXISTS FOR (q:Question) REQUIRE q.id IS UNIQUE",
                "CREATE CONSTRAINT student_id IF NOT EXISTS FOR (st:Student) REQUIRE st.id IS UNIQUE",
                "CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    print(f"✅ Created constraint: {constraint.split()[2]}")
                except Exception as e:
                    print(f"⚠️ Constraint already exists or error: {e}")

    def setup_sample_data(self):
        """Create sample skill graph data"""
        with self.driver.session() as session:
            # Create sample topics
            topics_query = """
            MERGE (t1:Topic {id: 'math', name: 'Mathematics', description: 'Mathematical concepts'})
            MERGE (t2:Topic {id: 'algebra', name: 'Algebra', description: 'Algebraic operations'})
            MERGE (t3:Topic {id: 'geometry', name: 'Geometry', description: 'Geometric concepts'})
            MERGE (t4:Topic {id: 'calculus', name: 'Calculus', description: 'Calculus concepts'})
            """
            session.run(topics_query)
            print("✅ Created sample topics")

            # Create sample skills
            skills_query = """
            MERGE (s1:Skill {id: 'basic_arithmetic', name: 'Basic Arithmetic', difficulty: 1, topic: 'math'})
            MERGE (s2:Skill {id: 'linear_equations', name: 'Linear Equations', difficulty: 2, topic: 'algebra'})
            MERGE (s3:Skill {id: 'quadratic_equations', name: 'Quadratic Equations', difficulty: 3, topic: 'algebra'})
            MERGE (s4:Skill {id: 'triangle_properties', name: 'Triangle Properties', difficulty: 2, topic: 'geometry'})
            MERGE (s5:Skill {id: 'derivatives', name: 'Derivatives', difficulty: 4, topic: 'calculus'})
            """
            session.run(skills_query)
            print("✅ Created sample skills")

            # Create relationships between topics
            topic_relations = [
                """MATCH (t1:Topic {id: 'math'}), (t2:Topic {id: 'algebra'})
                   MERGE (t1)-[:CONTAINS]->(t2)""",
                """MATCH (t1:Topic {id: 'math'}), (t3:Topic {id: 'geometry'})
                   MERGE (t1)-[:CONTAINS]->(t3)""",
                """MATCH (t1:Topic {id: 'math'}), (t4:Topic {id: 'calculus'})
                   MERGE (t1)-[:CONTAINS]->(t4)"""
            ]
            for query in topic_relations:
                session.run(query)
            print("✅ Created topic relationships")

            # Create skill prerequisites
            skill_prereqs = [
                """MATCH (s1:Skill {id: 'basic_arithmetic'}), (s2:Skill {id: 'linear_equations'})
                   MERGE (s1)-[:PREREQUISITE_FOR]->(s2)""",
                """MATCH (s2:Skill {id: 'linear_equations'}), (s3:Skill {id: 'quadratic_equations'})
                   MERGE (s2)-[:PREREQUISITE_FOR]->(s3)""",
                """MATCH (s3:Skill {id: 'quadratic_equations'}), (s5:Skill {id: 'derivatives'})
                   MERGE (s3)-[:PREREQUISITE_FOR]->(s5)"""
            ]
            for query in skill_prereqs:
                session.run(query)
            print("✅ Created skill prerequisite relationships")

            # Link skills to topics
            skill_topic_links = """
            MATCH (s:Skill), (t:Topic)
            WHERE s.topic = t.id
            MERGE (s)-[:BELONGS_TO]->(t)
            """
            session.run(skill_topic_links)
            print("✅ Linked skills to topics")

    def create_sample_questions(self):
        """Create sample questions for testing"""
        with self.driver.session() as session:
            questions_query = """
            MERGE (q1:Question {
                id: 'q1',
                text: 'What is 2 + 2?',
                difficulty: 1,
                correct_answer: '4',
                skill_id: 'basic_arithmetic'
            })
            MERGE (q2:Question {
                id: 'q2', 
                text: 'Solve for x: 2x + 5 = 11',
                difficulty: 2,
                correct_answer: 'x = 3',
                skill_id: 'linear_equations'
            })
            MERGE (q3:Question {
                id: 'q3',
                text: 'Find the area of a triangle with base 6 and height 4',
                difficulty: 2,
                correct_answer: '12',
                skill_id: 'triangle_properties'
            })
            """
            session.run(questions_query)
            print("✅ Created sample questions")

            # Link questions to skills
            question_skill_links = """
            MATCH (q:Question), (s:Skill)
            WHERE q.skill_id = s.id
            MERGE (q)-[:TESTS]->(s)
            """
            session.run(question_skill_links)
            print("✅ Linked questions to skills")

    def verify_setup(self):
        """Verify the setup by running some queries"""
        with self.driver.session() as session:
            # Count nodes
            result = session.run("MATCH (n) RETURN labels(n) as type, count(*) as count")
            print("\n📊 Node counts:")
            for record in result:
                print(f"  {record['type']}: {record['count']}")

            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(*) as count")
            print("\n🔗 Relationship counts:")
            for record in result:
                print(f"  {record['rel_type']}: {record['count']}")

            # Show skill prerequisites path
            print("\n🎯 Sample skill prerequisite path:")
            result = session.run("""
            MATCH path = (s1:Skill {id: 'basic_arithmetic'})-[:PREREQUISITE_FOR*]->(s5:Skill {id: 'derivatives'})
            RETURN [node in nodes(path) | node.name] as skill_path
            """)
            for record in result:
                print(f"  {' → '.join(record['skill_path'])}")

def main():
    # Configuration
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j" 
    NEO4J_PASSWORD = "password"
    
    print("🚀 Setting up Neo4j skill graph database...")
    
    try:
        setup = Neo4jSetup(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        print("\n1️⃣ Creating constraints and indexes...")
        setup.setup_constraints()
        
        print("\n2️⃣ Setting up sample data...")
        setup.setup_sample_data()
        
        print("\n3️⃣ Creating sample questions...")
        setup.create_sample_questions()
        
        print("\n4️⃣ Verifying setup...")
        setup.verify_setup()
        
        setup.close()
        print("\n✅ Neo4j skill graph setup completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error setting up Neo4j: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())