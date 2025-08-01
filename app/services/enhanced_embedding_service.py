"""
Enhanced SBERT Embedding Service
Gelişmiş semantic similarity ve embedding yönetimi
"""

import os
import numpy as np
import json
import redis
import hashlib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import threading
import structlog
from typing import List, Dict, Optional, Tuple, Union
import time
from datetime import datetime, timedelta

from app.core.config import settings
from app.services.embedding_service import EmbeddingService

logger = structlog.get_logger()

class EnhancedEmbeddingService(EmbeddingService):
    """Gelişmiş SBERT Embedding Servisi"""
    
    def __init__(self):
        super().__init__()
        
        # Redis for caching
        self.redis_client = redis.from_url(settings.redis_url)
        self.cache_ttl = 3600  # 1 hour cache
        
        # Available models
        self.available_models = {
            'small': {
                'name': 'paraphrase-MiniLM-L6-v2',
                'dimensions': 384,
                'description': 'Fast, lightweight model'
            },
            'medium': {
                'name': 'paraphrase-mpnet-base-v2', 
                'dimensions': 768,
                'description': 'Balanced speed and quality'
            },
            'large': {
                'name': 'paraphrase-multilingual-mpnet-base-v2',
                'dimensions': 768,
                'description': 'Multilingual support'
            }
        }
        
        # Current model
        self.current_model_key = 'small'
        self.model_name = self.available_models[self.current_model_key]['name']
        self.embedding_dim = self.available_models[self.current_model_key]['dimensions']
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'embeddings_computed': 0,
            'similarities_computed': 0
        }

    def switch_model(self, model_key: str) -> bool:
        """SBERT modelini değiştir"""
        if model_key not in self.available_models:
            logger.error("invalid_model_key", 
                        available=list(self.available_models.keys()),
                        requested=model_key)
            return False
        
        try:
            with self.model_lock:
                old_model = self.current_model_key
                self.current_model_key = model_key
                self.model_name = self.available_models[model_key]['name']
                self.embedding_dim = self.available_models[model_key]['dimensions']
                self.model = None  # Force reload
                
                # Test new model
                self._load_model()
                
                logger.info("model_switched", 
                           from_model=old_model,
                           to_model=model_key,
                           new_dimensions=self.embedding_dim)
                return True
                
        except Exception as e:
            logger.error("model_switch_error", error=str(e))
            return False

    def get_embedding_cache_key(self, text: str) -> str:
        """Cache key oluştur"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"embedding:{self.current_model_key}:{text_hash}"

    def compute_embedding_cached(self, text: str) -> List[float]:
        """Cache'li embedding hesaplama"""
        if not text or not text.strip():
            return [0.0] * self.embedding_dim
        
        # Cache'den kontrol et
        cache_key = self.get_embedding_cache_key(text)
        
        try:
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                self.stats['cache_hits'] += 1
                return json.loads(cached_result)
        except Exception as e:
            logger.debug("cache_read_error", error=str(e))
        
        # Cache miss - hesapla
        self.stats['cache_misses'] += 1
        embedding = self.compute_embedding(text)
        
        # Cache'e kaydet
        try:
            self.redis_client.setex(
                cache_key, 
                self.cache_ttl, 
                json.dumps(embedding)
            )
        except Exception as e:
            logger.debug("cache_write_error", error=str(e))
        
        self.stats['embeddings_computed'] += 1
        return embedding

    def compute_embeddings_batch_cached(self, texts: List[str]) -> List[List[float]]:
        """Cache'li toplu embedding hesaplama"""
        if not texts:
            return []
        
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # Cache'den mevcut olanları al
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results.append([0.0] * self.embedding_dim)
                continue
                
            cache_key = self.get_embedding_cache_key(text)
            try:
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    results.append(json.loads(cached_result))
                    self.stats['cache_hits'] += 1
                else:
                    results.append(None)  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    self.stats['cache_misses'] += 1
            except Exception as e:
                logger.debug("cache_read_error", error=str(e))
                results.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Cache'de olmayan metinleri hesapla
        if uncached_texts:
            uncached_embeddings = self.compute_embeddings_batch(uncached_texts)
            
            # Sonuçları yerleştir ve cache'e kaydet
            for i, (text, embedding) in enumerate(zip(uncached_texts, uncached_embeddings)):
                index = uncached_indices[i]
                results[index] = embedding
                
                # Cache'e kaydet
                try:
                    cache_key = self.get_embedding_cache_key(text)
                    self.redis_client.setex(
                        cache_key,
                        self.cache_ttl,
                        json.dumps(embedding)
                    )
                except Exception as e:
                    logger.debug("cache_write_error", error=str(e))
            
            self.stats['embeddings_computed'] += len(uncached_texts)
        
        return results

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """İki metin arasındaki semantic benzerlik"""
        try:
            embeddings = self.compute_embeddings_batch_cached([text1, text2])
            if len(embeddings) != 2:
                return 0.0
            
            similarity = cosine_similarity(
                np.array(embeddings[0]).reshape(1, -1),
                np.array(embeddings[1]).reshape(1, -1)
            )[0][0]
            
            self.stats['similarities_computed'] += 1
            return float(similarity)
            
        except Exception as e:
            logger.error("semantic_similarity_error", error=str(e))
            return 0.0

    def batch_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Metinler arası similarity matrisi"""
        try:
            embeddings = self.compute_embeddings_batch_cached(texts)
            if not embeddings:
                return np.array([])
            
            embedding_matrix = np.array(embeddings)
            similarity_matrix = cosine_similarity(embedding_matrix)
            
            self.stats['similarities_computed'] += len(texts) ** 2
            return similarity_matrix
            
        except Exception as e:
            logger.error("similarity_matrix_error", error=str(e))
            return np.array([])

    def semantic_clustering(self, texts: List[str], n_clusters: int = 5) -> Dict:
        """Metinleri semantic olarak kümelere ayır"""
        try:
            if len(texts) < n_clusters:
                logger.warning("insufficient_texts_for_clustering", 
                              texts_count=len(texts), 
                              clusters_requested=n_clusters)
                return {"clusters": [], "labels": [], "centroids": []}
            
            # Embeddings'i hesapla
            embeddings = self.compute_embeddings_batch_cached(texts)
            embedding_matrix = np.array(embeddings)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embedding_matrix)
            
            # Sonuçları organize et
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append({
                    'text': texts[i],
                    'index': i
                })
            
            # Cluster merkezlerini metin olarak tanımla
            cluster_descriptions = []
            for cluster_id in range(n_clusters):
                cluster_texts = [item['text'] for item in clusters.get(cluster_id, [])]
                if cluster_texts:
                    # En uzun metni cluster'ın temsilcisi olarak al
                    representative = max(cluster_texts, key=len)
                    cluster_descriptions.append({
                        'id': cluster_id,
                        'representative': representative,
                        'size': len(cluster_texts)
                    })
            
            logger.info("semantic_clustering_completed", 
                       n_texts=len(texts),
                       n_clusters=n_clusters,
                       cluster_sizes=[desc['size'] for desc in cluster_descriptions])
            
            return {
                "clusters": clusters,
                "labels": labels.tolist(),
                "cluster_descriptions": cluster_descriptions,
                "centroids": kmeans.cluster_centers_.tolist()
            }
            
        except Exception as e:
            logger.error("semantic_clustering_error", error=str(e))
            return {"clusters": [], "labels": [], "centroids": []}

    def find_semantic_outliers(self, texts: List[str], threshold: float = 0.3) -> List[Dict]:
        """Semantic olarak farklı metinleri bul"""
        try:
            if len(texts) < 3:
                return []
            
            # Similarity matrix hesapla
            similarity_matrix = self.batch_similarity_matrix(texts)
            
            outliers = []
            for i, text in enumerate(texts):
                # Bu metinin diğerleriyle ortalama benzerliği
                avg_similarity = np.mean([similarity_matrix[i][j] 
                                        for j in range(len(texts)) 
                                        if i != j])
                
                if avg_similarity < threshold:
                    outliers.append({
                        'text': text,
                        'index': i,
                        'avg_similarity': float(avg_similarity)
                    })
            
            # En düşük benzerlik skoruna göre sırala
            outliers.sort(key=lambda x: x['avg_similarity'])
            
            logger.info("semantic_outliers_found", 
                       count=len(outliers),
                       threshold=threshold)
            
            return outliers
            
        except Exception as e:
            logger.error("semantic_outliers_error", error=str(e))
            return []

    def semantic_search(self, query: str, question_pool: List[Dict], 
                       top_k: int = 10, similarity_threshold: float = 0.6) -> List[Dict]:
        """Gelişmiş semantic arama"""
        try:
            if not question_pool:
                return []
            
            # Query embedding'i hesapla
            query_embedding = self.compute_embedding_cached(query)
            
            # Question embeddings'leri hesapla
            question_texts = [q.get('content', '') for q in question_pool]
            question_embeddings = self.compute_embeddings_batch_cached(question_texts)
            
            # Similarity skorları hesapla
            query_vector = np.array(query_embedding).reshape(1, -1)
            results_with_scores = []
            
            for i, (question, embedding) in enumerate(zip(question_pool, question_embeddings)):
                if not embedding or sum(embedding) == 0:
                    continue
                
                question_vector = np.array(embedding).reshape(1, -1)
                similarity = cosine_similarity(query_vector, question_vector)[0][0]
                
                if similarity >= similarity_threshold:
                    result = question.copy()
                    result['semantic_similarity'] = float(similarity)
                    results_with_scores.append(result)
            
            # Similarity'ye göre sırala
            results_with_scores.sort(key=lambda x: x['semantic_similarity'], reverse=True)
            
            self.stats['similarities_computed'] += len(question_pool)
            
            logger.info("semantic_search_completed",
                       query_length=len(query),
                       pool_size=len(question_pool),
                       results_found=len(results_with_scores),
                       top_similarity=results_with_scores[0]['semantic_similarity'] if results_with_scores else 0)
            
            return results_with_scores[:top_k]
            
        except Exception as e:
            logger.error("semantic_search_error", error=str(e))
            return []

    def dimensionality_reduction(self, texts: List[str], 
                               target_dims: int = 2) -> Tuple[np.ndarray, PCA]:
        """Embedding'leri boyut azaltma ile görselleştirme için hazırla"""
        try:
            embeddings = self.compute_embeddings_batch_cached(texts)
            if not embeddings:
                return np.array([]), None
            
            embedding_matrix = np.array(embeddings)
            
            # PCA ile boyut azaltma
            pca = PCA(n_components=target_dims, random_state=42)
            reduced_embeddings = pca.fit_transform(embedding_matrix)
            
            logger.info("dimensionality_reduction_completed",
                       original_dims=embedding_matrix.shape[1],
                       target_dims=target_dims,
                       explained_variance=float(pca.explained_variance_ratio_.sum()))
            
            return reduced_embeddings, pca
            
        except Exception as e:
            logger.error("dimensionality_reduction_error", error=str(e))
            return np.array([]), None

    def get_enhanced_stats(self) -> Dict:
        """Gelişmiş istatistikler"""
        basic_stats = self.get_embedding_stats()
        
        # Cache statistics
        cache_total = self.stats['cache_hits'] + self.stats['cache_misses']
        cache_hit_rate = (self.stats['cache_hits'] / cache_total * 100) if cache_total > 0 else 0
        
        enhanced_stats = {
            **basic_stats,
            'model_info': {
                'current_model': self.current_model_key,
                'model_name': self.model_name,
                'embedding_dimensions': self.embedding_dim,
                'available_models': list(self.available_models.keys())
            },
            'cache_stats': {
                'cache_hits': self.stats['cache_hits'],
                'cache_misses': self.stats['cache_misses'],
                'cache_hit_rate': round(cache_hit_rate, 2),
                'cache_ttl_seconds': self.cache_ttl
            },
            'computation_stats': {
                'embeddings_computed': self.stats['embeddings_computed'],
                'similarities_computed': self.stats['similarities_computed']
            }
        }
        
        return enhanced_stats

    def clear_cache(self) -> bool:
        """Embedding cache'ini temizle"""
        try:
            # Pattern ile embedding cache'ini temizle
            pattern = f"embedding:{self.current_model_key}:*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                deleted_count = self.redis_client.delete(*keys)
                logger.info("cache_cleared", deleted_keys=deleted_count)
            else:
                logger.info("cache_already_empty")
            
            # Stats'ı sıfırla
            self.stats['cache_hits'] = 0
            self.stats['cache_misses'] = 0
            
            return True
            
        except Exception as e:
            logger.error("cache_clear_error", error=str(e))
            return False

# Global instance with enhanced features
enhanced_embedding_service = EnhancedEmbeddingService()

# Enhanced convenience functions
def compute_embedding_cached(text: str) -> List[float]:
    """Cache'li embedding hesaplama"""
    return enhanced_embedding_service.compute_embedding_cached(text)

def semantic_similarity(text1: str, text2: str) -> float:
    """Semantic benzerlik"""
    return enhanced_embedding_service.semantic_similarity(text1, text2)

def semantic_clustering(texts: List[str], n_clusters: int = 5) -> Dict:
    """Semantic clustering"""
    return enhanced_embedding_service.semantic_clustering(texts, n_clusters)

def semantic_search(query: str, question_pool: List[Dict], 
                   top_k: int = 10, similarity_threshold: float = 0.6) -> List[Dict]:
    """Gelişmiş semantic arama"""
    return enhanced_embedding_service.semantic_search(query, question_pool, top_k, similarity_threshold)

def find_semantic_outliers(texts: List[str], threshold: float = 0.3) -> List[Dict]:
    """Semantic outlier'ları bul"""
    return enhanced_embedding_service.find_semantic_outliers(texts, threshold)