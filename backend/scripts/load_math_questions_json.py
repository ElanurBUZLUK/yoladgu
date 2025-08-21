#!/usr/bin/env python3
"""
Enhanced JSON Math Questions Loader

Loads math questions from JSON files with automatic embedding generation
and database insertion using the new enhanced format.

Usage:
    python scripts/load_math_questions_json.py --json data/questions/math_questions_enhanced.json
    python scripts/load_math_questions_json.py --dir data/questions/
"""

import asyncio
import json
import os
import sys
import argparse
from uuid import uuid4
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
try:
    from sqlalchemy.orm import sessionmaker
except ImportError:
    # Fallback for older SQLAlchemy versions
    from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from app.models.question import Question, Subject, QuestionType, SourceType
from app.services.embedding_service import embedding_service
from app.core.config import settings

# Configuration
EMBED_DIM = 1536
BATCH_SIZE = 50  # Process questions in batches

class EnhancedMathQuestionLoader:
    """Enhanced loader for math questions with embedding generation"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.Session = None
        
    async def initialize(self):
        """Initialize database connection and embedding service"""
        self.engine = create_async_engine(
            self.database_url, 
            pool_size=10, 
            max_overflow=20,
            echo=False
        )
        try:
            self.Session = sessionmaker(
                self.engine, 
                class_=AsyncSession, 
                expire_on_commit=False
            )
        except TypeError:
            # Fallback for older SQLAlchemy versions
            from sqlalchemy.orm import sessionmaker
            self.Session = sessionmaker(
                bind=self.engine,
                expire_on_commit=False
            )
        
        # Initialize embedding service
        await embedding_service.initialize()
        
    async def cleanup(self):
        """Clean up resources"""
        if self.engine:
            await self.engine.dispose()
        await embedding_service.cleanup()
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            embeddings = []
            for text in texts:
                embedding = await embedding_service.generate_embedding(text)
                embeddings.append(embedding)
            return embeddings
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            # Return zero vectors as fallback
            return [[0.0] * EMBED_DIM for _ in texts]
    
    def convert_difficulty(self, difficulty: float) -> int:
        """Convert continuous difficulty (0.0-2.0) to discrete level (1-5)"""
        if difficulty <= 0.5:
            return 1
        elif difficulty <= 1.0:
            return 2
        elif difficulty <= 1.5:
            return 3
        elif difficulty <= 1.8:
            return 4
        else:
            return 5
    
    def convert_question_type(self, options: Dict[str, str]) -> QuestionType:
        """Determine question type based on options"""
        if len(options) == 2 and all(opt in ['True', 'False', 'true', 'false'] for opt in options.values()):
            return QuestionType.TRUE_FALSE
        elif len(options) == 0:
            return QuestionType.OPEN_ENDED
        else:
            return QuestionType.MULTIPLE_CHOICE
    
    async def load_single_file(self, file_path: str) -> Dict[str, Any]:
        """Load questions from a single JSON file"""
        print(f"📄 Loading questions from: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("JSON must contain an array of questions")
            
            print(f"   Found {len(data)} questions")
            return {"file": file_path, "questions": data, "success": True}
            
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            return {"file": file_path, "questions": [], "success": False, "error": "File not found"}
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in {file_path}: {e}")
            return {"file": file_path, "questions": [], "success": False, "error": f"Invalid JSON: {e}"}
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            return {"file": file_path, "questions": [], "success": False, "error": str(e)}
    
    async def process_questions_batch(self, questions: List[Dict[str, Any]], session: AsyncSession) -> Dict[str, int]:
        """Process a batch of questions"""
        processed = 0
        failed = 0
        
        try:
            # Prepare texts for embedding
            texts = []
            for q in questions:
                # Combine stem and options for embedding
                text_parts = [q.get('stem', '')]
                options = q.get('options', {})
                text_parts.extend(options.values())
                texts.append(' '.join(text_parts))
            
            # Generate embeddings
            print(f"   Generating embeddings for {len(texts)} questions...")
            embeddings = await self.embed_texts(texts)
            
            # Process each question
            for i, (question_data, embedding) in enumerate(zip(questions, embeddings)):
                try:
                    # Validate required fields
                    required_fields = ['stem', 'options', 'correct_answer', 'topic']
                    for field in required_fields:
                        if field not in question_data:
                            raise ValueError(f"Missing required field: {field}")
                    
                    # Convert data
                    difficulty_level = self.convert_difficulty(question_data.get('difficulty', 1.0))
                    question_type = self.convert_question_type(question_data['options'])
                    
                    # Create question object
                    question = Question(
                        subject=Subject.MATH,
                        content=question_data['stem'],
                        question_type=question_type,
                        difficulty_level=difficulty_level,
                        original_difficulty=difficulty_level,
                        topic_category=question_data['topic'],
                        correct_answer=question_data['correct_answer'],
                        options=question_data['options'],
                        source_type=SourceType.MANUAL,
                        estimated_difficulty=question_data.get('difficulty', 1.0),
                        embedding=embedding,
                        embedding_model="text-embedding-ada-002",
                        question_metadata={
                            "subtopic": question_data.get('subtopic'),
                            "source": question_data.get('source', 'seed'),
                            "estimated_time": question_data.get('metadata', {}).get('estimated_time', 60),
                            "learning_objectives": question_data.get('metadata', {}).get('learning_objectives', []),
                            "tags": question_data.get('metadata', {}).get('tags', []),
                            "cefr_level": question_data.get('metadata', {}).get('cefr_level', 'A1')
                        }
                    )
                    
                    session.add(question)
                    processed += 1
                    
                except Exception as e:
                    failed += 1
                    print(f"   ❌ Question {i+1}: {e}")
            
            # Commit batch
            await session.commit()
            print(f"   ✅ Processed {processed} questions, {failed} failed")
            
        except Exception as e:
            print(f"   ❌ Batch processing error: {e}")
            await session.rollback()
            failed += len(questions)
        
        return {"processed": processed, "failed": failed}
    
    async def load_json(self, file_path: str) -> Dict[str, Any]:
        """Load questions from a single JSON file"""
        await self.initialize()
        
        try:
            # Load file
            file_result = await self.load_single_file(file_path)
            if not file_result['success']:
                return file_result
            
            questions = file_result['questions']
            if not questions:
                return {"success": True, "message": "No questions found", "processed": 0, "failed": 0}
            
            # Process in batches
            total_processed = 0
            total_failed = 0
            
            async with self.Session() as session:
                for i in range(0, len(questions), BATCH_SIZE):
                    batch = questions[i:i + BATCH_SIZE]
                    print(f"   Processing batch {i//BATCH_SIZE + 1}/{(len(questions) + BATCH_SIZE - 1)//BATCH_SIZE}")
                    
                    result = await self.process_questions_batch(batch, session)
                    total_processed += result["processed"]
                    total_failed += result["failed"]
            
            return {
                "success": True,
                "message": f"Successfully processed {total_processed} questions",
                "processed": total_processed,
                "failed": total_failed,
                "file": file_path
            }
            
        finally:
            await self.cleanup()
    
    async def load_directory(self, directory_path: str) -> Dict[str, Any]:
        """Load all JSON files from a directory"""
        await self.initialize()
        
        try:
            directory = Path(directory_path)
            if not directory.exists():
                return {"success": False, "error": f"Directory not found: {directory_path}"}
            
            json_files = list(directory.glob("*.json"))
            if not json_files:
                return {"success": True, "message": "No JSON files found", "processed": 0, "failed": 0}
            
            print(f"📁 Found {len(json_files)} JSON files in {directory_path}")
            
            total_processed = 0
            total_failed = 0
            results = []
            
            for json_file in json_files:
                result = await self.load_json(str(json_file))
                results.append(result)
                
                if result.get('success'):
                    total_processed += result.get('processed', 0)
                    total_failed += result.get('failed', 0)
            
            return {
                "success": True,
                "message": f"Processed {len(json_files)} files",
                "processed": total_processed,
                "failed": total_failed,
                "files": results
            }
            
        finally:
            await self.cleanup()

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Load math questions from JSON files")
    parser.add_argument("--json", help="Path to single JSON file")
    parser.add_argument("--dir", help="Path to directory containing JSON files")
    parser.add_argument("--database-url", help="Database URL (optional)")
    
    args = parser.parse_args()
    
    if not args.json and not args.dir:
        print("❌ Please specify either --json or --dir")
        parser.print_help()
        return
    
    # Get database URL
    database_url = args.database_url or settings.database_url
    
    # Create loader
    loader = EnhancedMathQuestionLoader(database_url)
    
    try:
        if args.json:
            result = await loader.load_json(args.json)
        else:
            result = await loader.load_directory(args.dir)
        
        if result.get('success'):
            print(f"✅ {result['message']}")
            print(f"   📊 Processed: {result.get('processed', 0)}")
            print(f"   ❌ Failed: {result.get('failed', 0)}")
        else:
            print(f"❌ {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
