"""
SBERT Embedding Service
Gerçek embedding hesaplama ve yönetim servisi
"""

import os
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer
import threading
import structlog
from app.core.config import settings
from app.db.database import SessionLocal
from app.db.models import Question
import asyncio
from typing import List, Dict, Optional, Tuple
import time

logger = structlog.get_logger()

class EmbeddingService:
    def __init__(self):
        self.model = None
        self.model_lock = threading.Lock()
        self.model_name = "paraphrase-MiniLM-L6-v2"  # 384 boyutlu embedding
        self.embedding_dim = 384
        
    def _load_model(self):
        """Thread-safe model yükleme"""
        with self.model_lock:
            if self.model is None:
                logger.info("loading_sbert_model", model_name=self.model_name)
                self.model = SentenceTransformer(self.model_name)
                logger.info("sbert_model_loaded", model_name=self.model_name)
        return self.model
    
    def compute_embedding(self, text: str) -> List[float]:
        """Tek bir metin için embedding hesapla"""
        if not text or not text.strip():
            return [0.0] * self.embedding_dim
            
        try:
            model = self._load_model()
            embedding = model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error("embedding_computation_error", text=text[:100], error=str(e))
            return [0.0] * self.embedding_dim
    
    def compute_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Toplu embedding hesaplama"""
        if not texts:
            return []
            
        try:
            model = self._load_model()
            # Boş metinleri filtrele
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts:
                return [[0.0] * self.embedding_dim] * len(texts)
            
            embeddings = model.encode(valid_texts, convert_to_tensor=False)
            
            # Sonuçları orijinal sıraya göre düzenle
            result = []
            valid_idx = 0
            for text in texts:
                if text and text.strip():
                    result.append(embeddings[valid_idx].tolist())
                    valid_idx += 1
                else:
                    result.append([0.0] * self.embedding_dim)
            
            return result
        except Exception as e:
            logger.error("batch_embedding_error", text_count=len(texts), error=str(e))
            return [[0.0] * self.embedding_dim] * len(texts)
    
    def get_postgres_connection(self):
        """PostgreSQL bağlantısı"""
        db_url = settings.DATABASE_URL
        if db_url.startswith('postgresql+psycopg2://'):
            db_url = db_url.replace('postgresql+psycopg2://', 'postgresql://')
        
        return psycopg2.connect(db_url)
    
    def save_embedding_to_db(self, question_id: int, embedding: List[float]) -> bool:
        """Embedding'i PostgreSQL'e kaydet (JSON formatında)"""
        try:
            import json
            conn = self.get_postgres_connection()
            cursor = conn.cursor()
            
            # Embedding'i JSON string'e çevir
            embedding_json = json.dumps(embedding)
            
            cursor.execute(
                "UPDATE questions SET embedding = %s WHERE id = %s",
                (embedding_json, question_id)
            )
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("embedding_saved", question_id=question_id, dimensions=len(embedding))
            return True
            
        except Exception as e:
            logger.error("embedding_save_error", question_id=question_id, error=str(e))
            return False
    
    def save_embeddings_batch(self, question_embeddings: List[Tuple[int, List[float]]]) -> int:
        """Toplu embedding kaydetme (JSON formatında)"""
        if not question_embeddings:
            return 0
            
        try:
            import json
            conn = self.get_postgres_connection()
            cursor = conn.cursor()
            
            # JSON formatına çevir
            json_embeddings = [(json.dumps(embedding), question_id) for question_id, embedding in question_embeddings]
            
            # Batch update
            cursor.executemany(
                "UPDATE questions SET embedding = %s WHERE id = %s",
                json_embeddings
            )
            
            updated_count = cursor.rowcount
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("batch_embeddings_saved", count=updated_count)
            return updated_count
            
        except Exception as e:
            logger.error("batch_embedding_save_error", error=str(e))
            return 0
    
    def find_similar_questions(self, query_embedding: List[float], 
                             threshold: float = 0.8, limit: int = 10) -> List[Dict]:
        """Benzer soruları bul (Cosine Similarity ile)"""
        try:
            import json
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            conn = self.get_postgres_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Tüm embedding'li soruları al
            cursor.execute("""
                SELECT 
                    q.id,
                    q.content,
                    q.difficulty_level,
                    q.subject_id,
                    q.topic_id,
                    q.embedding
                FROM questions q
                WHERE q.embedding IS NOT NULL AND q.is_active = true
            """)
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not results:
                return []
            
            # Similarity hesapla
            similar_questions = []
            query_vector = np.array(query_embedding).reshape(1, -1)
            
            for row in results:
                try:
                    # JSON'dan embedding'i parse et
                    stored_embedding = json.loads(row['embedding'])
                    stored_vector = np.array(stored_embedding).reshape(1, -1)
                    
                    # Cosine similarity hesapla
                    similarity = cosine_similarity(query_vector, stored_vector)[0][0]
                    
                    if similarity >= threshold:
                        question_dict = dict(row)
                        question_dict['similarity'] = float(similarity)
                        question_dict.pop('embedding')  # Embedding'i response'dan kaldır
                        similar_questions.append(question_dict)
                        
                except (json.JSONDecodeError, ValueError) as e:
                    logger.debug("embedding_parse_error", question_id=row['id'], error=str(e))
                    continue
            
            # Similarity'ye göre sırala
            similar_questions.sort(key=lambda x: x['similarity'], reverse=True)
            
            return similar_questions[:limit]
            
        except Exception as e:
            logger.error("similar_questions_error", error=str(e))
            return []
    
    def get_questions_without_embeddings(self, limit: int = 100) -> List[Dict]:
        """Embedding'i olmayan soruları getir"""
        try:
            conn = self.get_postgres_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT id, content, question_type, difficulty_level, subject_id
                FROM questions
                WHERE embedding IS NULL AND is_active = true
                LIMIT %s
            """, (limit,))
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error("questions_without_embeddings_error", error=str(e))
            return []
    
    def get_embedding_stats(self) -> Dict:
        """Embedding istatistiklerini getir"""
        try:
            conn = self.get_postgres_connection()
            cursor = conn.cursor()
            
            # Toplam soru sayısı
            cursor.execute("SELECT COUNT(*) FROM questions WHERE is_active = true")
            total_questions = cursor.fetchone()[0]
            
            # Embedding'i olan soru sayısı
            cursor.execute("SELECT COUNT(*) FROM questions WHERE embedding IS NOT NULL AND is_active = true")
            questions_with_embedding = cursor.fetchone()[0]
            
            # Embedding'i olmayan soru sayısı
            cursor.execute("SELECT COUNT(*) FROM questions WHERE embedding IS NULL AND is_active = true")
            questions_without_embedding = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            return {
                "total_questions": total_questions,
                "questions_with_embedding": questions_with_embedding,  # Düzeltildi
                "questions_without_embedding": questions_without_embedding,
                "embedding_coverage": (questions_with_embedding / total_questions * 100) if total_questions > 0 else 0
            }
            
        except Exception as e:
            logger.error("embedding_stats_error", error=str(e))
            return {}
    
    async def update_question_embedding(self, question_id: int, content: str) -> bool:
        """Tek soru için embedding güncelle"""
        try:
            embedding = self.compute_embedding(content)
            success = self.save_embedding_to_db(question_id, embedding)
            
            if success:
                logger.info("question_embedding_updated", question_id=question_id)
            else:
                logger.error("question_embedding_update_failed", question_id=question_id)
            
            return success
            
        except Exception as e:
            logger.error("update_question_embedding_error", question_id=question_id, error=str(e))
            return False
    
    async def batch_update_embeddings(self, batch_size: int = 50) -> Dict:
        """Toplu embedding güncelleme"""
        start_time = time.time()
        
        try:
            # Embedding'i olmayan soruları getir
            questions = self.get_questions_without_embeddings(batch_size)
            
            if not questions:
                return {
                    "processed": 0,
                    "success": 0,
                    "failed": 0,
                    "duration": time.time() - start_time
                }
            
            # Metinleri ve ID'leri ayır
            question_ids = [q['id'] for q in questions]
            texts = [q['content'] for q in questions]
            
            # Toplu embedding hesapla
            embeddings = self.compute_embeddings_batch(texts)
            
            # Embedding'leri veritabanına kaydet
            question_embeddings = list(zip(question_ids, embeddings))
            saved_count = self.save_embeddings_batch(question_embeddings)
            
            duration = time.time() - start_time
            
            logger.info("batch_embedding_update_completed", 
                       processed=len(questions), 
                       saved=saved_count, 
                       duration=duration)
            
            return {
                "processed": len(questions),
                "success": saved_count,
                "failed": len(questions) - saved_count,
                "duration": duration
            }
            
        except Exception as e:
            logger.error("batch_embedding_update_error", error=str(e))
            return {
                "processed": 0,
                "success": 0,
                "failed": 0,
                "duration": time.time() - start_time,
                "error": str(e)
            }

# Global instance
embedding_service = EmbeddingService()

# Convenience functions
def compute_embedding(text: str) -> List[float]:
    """Tek metin için embedding hesapla"""
    return embedding_service.compute_embedding(text)

def compute_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Toplu embedding hesapla"""
    return embedding_service.compute_embeddings_batch(texts)

def find_similar_questions(query_embedding: List[float], 
                         threshold: float = 0.8, 
                         limit: int = 10) -> List[Dict]:
    """Benzer soruları bul"""
    return embedding_service.find_similar_questions(query_embedding, threshold, limit)

async def update_question_embedding(question_id: int, content: str) -> bool:
    """Soru embedding'ini güncelle"""
    return await embedding_service.update_question_embedding(question_id, content)

async def batch_update_embeddings(batch_size: int = 50) -> Dict:
    """Toplu embedding güncelleme"""
    return await embedding_service.batch_update_embeddings(batch_size) 