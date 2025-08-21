#!/usr/bin/env python3
"""
English Questions JSON Loader

This script loads English questions from JSON files into the database with automatic embedding generation.
Supports both single file and directory loading.

Usage:
    python scripts/load_english_questions_json.py --json data/questions/english_questions_enhanced.json
    python scripts/load_english_questions_json.py --dir data/questions/
"""

import asyncio
import json
import os
import sys
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import text
from app.models.question import Question, Subject, QuestionType, SourceType
from app.services.embedding_service import embedding_service
from app.core.config import settings

logger = logging.getLogger(__name__)

class EnhancedEnglishQuestionLoader:
    """Enhanced English question loader with embedding generation"""
    
    def __init__(self):
        self.embedding_service = embedding_service
        self.batch_size = 10  # Process questions in batches
        
    def _convert_difficulty(self, difficulty: float) -> int:
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
    
    def _determine_question_type(self, options: Dict[str, str]) -> QuestionType:
        """Determine question type based on options"""
        if len(options) == 2 and all(opt in ['True', 'False', 'true', 'false'] for opt in options.values()):
            return QuestionType.TRUE_FALSE
        elif len(options) == 0:
            return QuestionType.OPEN_ENDED
        else:
            return QuestionType.MULTIPLE_CHOICE
    
    async def _generate_embedding(self, question_text: str, options_text: str) -> List[float]:
        """Generate embedding for question content"""
        try:
            # Combine question and options for embedding
            full_text = f"{question_text} {' '.join(options_text.split())}"
            embedding = await self.embedding_service.get_embedding(full_text, domain="english")
            return embedding
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536
    
    async def _process_questions_batch(
        self, 
        questions: List[Dict[str, Any]], 
        session: AsyncSession
    ) -> Dict[str, int]:
        """Process a batch of questions with embeddings"""
        processed = 0
        failed = 0
        
        for question_data in questions:
            try:
                # Validate required fields
                required_fields = ['stem', 'options', 'correct_answer', 'topic']
                for field in required_fields:
                    if field not in question_data:
                        raise ValueError(f"Missing required field: {field}")
                
                # Convert difficulty
                difficulty = question_data.get('difficulty', 1.0)
                difficulty_level = self._convert_difficulty(difficulty)
                
                # Determine question type
                question_type = self._determine_question_type(question_data['options'])
                
                # Generate embedding
                options_text = ' '.join(question_data['options'].values())
                embedding = await self._generate_embedding(question_data['stem'], options_text)
                
                # Create question object
                question = Question(
                    subject=Subject.ENGLISH,
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
                    embedding_model="text-embedding-3-small",
                    embedding_generated_at=datetime.utcnow(),
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
                logger.error(f"Failed to process question: {e}")
        
        # Commit batch
        try:
            await session.commit()
            logger.info(f"✅ Processed batch: {processed} successful, {failed} failed")
        except Exception as e:
            await session.rollback()
            logger.error(f"❌ Batch commit failed: {e}")
            failed += processed
            processed = 0
        
        return {"processed": processed, "failed": failed}
    
    async def load_single_file(self, file_path: str) -> Dict[str, Any]:
        """Load questions from a single JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                questions_data = json.load(f)
            
            if not isinstance(questions_data, list):
                raise ValueError("JSON must contain an array of questions")
            
            logger.info(f"📖 Loading {len(questions_data)} questions from {file_path}")
            
            # Create database session
            engine = create_async_engine(settings.database_url)
            async_session = async_sessionmaker(engine, expire_on_commit=False)
            
            total_processed = 0
            total_failed = 0
            
            async with async_session() as session:
                # Process in batches
                for i in range(0, len(questions_data), self.batch_size):
                    batch = questions_data[i:i + self.batch_size]
                    result = await self._process_questions_batch(batch, session)
                    total_processed += result["processed"]
                    total_failed += result["failed"]
            
            await engine.dispose()
            
            return {
                "success": True,
                "file_path": file_path,
                "total_questions": len(questions_data),
                "processed": total_processed,
                "failed": total_failed,
                "message": f"Successfully processed {total_processed} questions from {file_path}"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to load file {file_path}: {e}")
            return {
                "success": False,
                "file_path": file_path,
                "error": str(e),
                "message": f"Failed to load {file_path}: {e}"
            }
    
    async def load_directory(self, directory_path: str) -> Dict[str, Any]:
        """Load questions from all JSON files in a directory"""
        try:
            if not os.path.exists(directory_path):
                raise ValueError(f"Directory does not exist: {directory_path}")
            
            json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
            if not json_files:
                raise ValueError(f"No JSON files found in {directory_path}")
            
            logger.info(f"📁 Found {len(json_files)} JSON files in {directory_path}")
            
            total_processed = 0
            total_failed = 0
            results = []
            
            for json_file in json_files:
                file_path = os.path.join(directory_path, json_file)
                result = await self.load_single_file(file_path)
                results.append(result)
                
                if result["success"]:
                    total_processed += result.get("processed", 0)
                    total_failed += result.get("failed", 0)
                else:
                    total_failed += 1
            
            return {
                "success": True,
                "directory_path": directory_path,
                "files_processed": len(json_files),
                "total_processed": total_processed,
                "total_failed": total_failed,
                "results": results,
                "message": f"Processed {len(json_files)} files: {total_processed} questions imported, {total_failed} failed"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to load directory {directory_path}: {e}")
            return {
                "success": False,
                "directory_path": directory_path,
                "error": str(e),
                "message": f"Failed to load directory {directory_path}: {e}"
            }

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Load English questions from JSON files")
    parser.add_argument("--json", help="Path to JSON file")
    parser.add_argument("--dir", help="Path to directory containing JSON files")
    
    args = parser.parse_args()
    
    if not args.json and not args.dir:
        print("❌ Please specify either --json or --dir")
        parser.print_help()
        return 1
    
    loader = EnhancedEnglishQuestionLoader()
    
    try:
        if args.json:
            result = await loader.load_single_file(args.json)
        else:
            result = await loader.load_directory(args.dir)
        
        if result["success"]:
            print(f"✅ {result['message']}")
            return 0
        else:
            print(f"❌ {result['message']}")
            return 1
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    exit(asyncio.run(main()))
