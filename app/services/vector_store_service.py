"""
Vector Store Service
pgvector tabanlı vector database işlemleri
"""

import structlog
from typing import List, Dict, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db.database import get_db
from app.db.models import Question
logger = structlog.get_logger()


class VectorStoreService:
    """Vector database işlemleri için service"""
    
    def __init__(self):
        self.initialized = False
        self.embedding_service = None
    
    def _get_embedding_service(self):
        """Lazy loading for embedding service"""
        if self.embedding_service is None:
            from app.services.enhanced_embedding_service import enhanced_embedding_service
            self.embedding_service = enhanced_embedding_service
        return self.embedding_service
    
    async def initialize(self) -> bool:
        """Vector store'u başlat"""
        try:
            # Database bağlantısını test et
            db = next(get_db())
            
            # pgvector extension'ının kurulu olup olmadığını kontrol et
            result = db.execute(text("SELECT * FROM pg_extension WHERE extname = 'vector';"))
            if not result.fetchone():
                logger.error("pgvector_extension_not_found")
                return False
            
            # Embedding sütununun var olup olmadığını kontrol et
            result = db.execute(text("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'questions' AND column_name = 'embedding_vector';
            """))
            if not result.fetchone():
                logger.error("embedding_vector_column_not_found")
                return False
            
            self.initialized = True
            logger.info("vector_store_initialized")
            return True
            
        except Exception as e:
            logger.error("vector_store_initialization_error", error=str(e))
            return False
    
    async def upsert(
        self,
        question_id: int,
        embedding: List[float],
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Embedding'i vector store'a upsert et"""
        try:
            if not self.initialized:
                await self.initialize()
            
            # Database'e kaydet
            db = next(get_db())
            question = db.query(Question).filter(Question.id == question_id).first()
            
            if question:
                question.embedding_vector = embedding
                db.commit()
                logger.info("embedding_upserted", question_id=question_id)
                return True
            else:
                logger.error("question_not_found", question_id=question_id)
                return False
                
        except Exception as e:
            logger.error("upsert_error", error=str(e), question_id=question_id)
            return False

    async def store_question_embedding(
        self,
        question_id: int,
        question_text: str,
        metadata: Optional[Dict] = None,
        subject_id: Optional[int] = None,
        topic_id: Optional[int] = None,
        difficulty_level: Optional[int] = None,
    ) -> bool:
        """Soru embedding'ini vector store'a kaydet"""
        try:
            if not self.initialized:
                await self.initialize()
            
            # Embedding hesapla
            embedding_service = self._get_embedding_service()
            embedding = await embedding_service.compute_embedding_cached(question_text)
            
            # Database'e kaydet
            db = next(get_db())
            question = db.query(Question).filter(Question.id == question_id).first()
            
            if question:
                question.embedding_vector = embedding
                db.commit()
                logger.info("question_embedding_stored", question_id=question_id)
                return True
            else:
                logger.error("question_not_found", question_id=question_id)
                return False
                
        except Exception as e:
            logger.error("store_question_embedding_error", error=str(e), question_id=question_id)
            return False
    
    async def semantic_search(
        self,
        query_text: str,
        k: int = 10,
        similarity_threshold: float = 0.7,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """Vector database'de semantic arama yap"""
        try:
            if not self.initialized:
                await self.initialize()
            
            # Query embedding'ini hesapla
            embedding_service = self._get_embedding_service()
            query_embedding = await embedding_service.compute_embedding_cached(query_text)
            
            # Vector similarity search
            db = next(get_db())
            
            # Base query
            query = db.query(Question).filter(
                Question.embedding_vector.isnot(None)
            )
            
            # Filters uygula
            if filters:
                if filters.get("subject_id"):
                    query = query.filter(Question.subject_id == filters["subject_id"])
                if filters.get("topic_id"):
                    query = query.filter(Question.topic_id == filters["topic_id"])
                if filters.get("difficulty_level"):
                    query = query.filter(Question.difficulty_level == filters["difficulty_level"])
            
            # Vector similarity ile sırala
            query = query.order_by(
                Question.embedding_vector.cosine_distance(query_embedding)
            ).limit(k)
            
            results = query.all()
            
            # Sonuçları formatla
            formatted_results = []
            for question in results:
                # Similarity hesapla
                similarity = 1 - question.embedding_vector.cosine_distance(query_embedding)
                
                if similarity >= similarity_threshold:
                    formatted_results.append({
                        "question_id": question.id,
                        "content": question.content,
                        "similarity": float(similarity),
                        "subject_id": question.subject_id,
                        "topic_id": question.topic_id,
                        "difficulty_level": question.difficulty_level,
                    })
            
            logger.info("semantic_search_completed", 
                       query=query_text, 
                       results_count=len(formatted_results))
            return formatted_results
            
        except Exception as e:
            logger.error("semantic_search_error", error=str(e))
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Vector store istatistiklerini getir"""
        try:
            db = next(get_db())
            
            # Toplam soru sayısı
            total_questions = db.query(Question).count()
            
            # Embedding'i olan soru sayısı
            questions_with_embedding = db.query(Question).filter(
                Question.embedding_vector.isnot(None)
            ).count()
            
            # Embedding'i olmayan soru sayısı
            questions_without_embedding = total_questions - questions_with_embedding
            
            return {
                "total_questions": total_questions,
                "questions_with_embedding": questions_with_embedding,
                "questions_without_embedding": questions_without_embedding,
                "embedding_coverage_percentage": (
                    (questions_with_embedding / total_questions * 100) 
                    if total_questions > 0 else 0
                ),
                "vector_store_initialized": self.initialized,
            }
            
        except Exception as e:
            logger.error("get_vector_store_stats_error", error=str(e))
            return {
                "error": str(e),
                "vector_store_initialized": self.initialized,
            }
    
    async def batch_store_embeddings(self, batch_size: int = 100) -> Dict[str, int]:
        """Batch olarak embedding'leri kaydet"""
        try:
            db = next(get_db())
            
            # Embedding'i olmayan soruları getir
            questions = db.query(Question).filter(
                Question.embedding_vector.isnull()
            ).limit(batch_size).all()
            
            stored_count = 0
            for question in questions:
                try:
                    embedding_service = self._get_embedding_service()
                    embedding = await embedding_service.compute_embedding_cached(
                        question.content
                    )
                    question.embedding_vector = embedding
                    stored_count += 1
                except Exception as e:
                    logger.error("batch_store_embedding_error", 
                               question_id=question.id, error=str(e))
                    continue
            
            db.commit()
            logger.info("batch_store_embeddings_completed", 
                       processed=len(questions), 
                       stored=stored_count)
            
            return {
                "processed": len(questions),
                "stored": stored_count,
                "failed": len(questions) - stored_count,
            }
            
        except Exception as e:
            logger.error("batch_store_embeddings_error", error=str(e))
            return {"processed": 0, "stored": 0, "failed": 0}


# Global instance
vector_store_service = VectorStoreService() 