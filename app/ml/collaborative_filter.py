"""
Scikit-learn tabanlı Collaborative Filtering Recommendation Engine
LightFM'in yerine geçen alternatif implementasyon
"""

import numpy as np
import pickle
import json
import structlog
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import pandas as pd

logger = structlog.get_logger()

class CollaborativeFilterEngine:
    """Scikit-learn tabanlı Collaborative Filtering sistemi"""
    
    def __init__(self, redis_client, n_components: int = 50):
        self.redis = redis_client
        self.n_components = n_components
        
        # Redis keys
        self.model_key = "cf:model"
        self.user_features_key = "cf:user_features"
        self.item_features_key = "cf:item_features"
        self.interaction_matrix_key = "cf:interaction_matrix"
        self.stats_key = "cf:statistics"
        
        # Models
        self.nmf_model = None
        self.svd_model = None
        self.user_features = None
        self.item_features = None
        self.interaction_matrix = None
        self.user_to_index = {}
        self.item_to_index = {}
        self.index_to_user = {}
        self.index_to_item = {}
        
        # Statistics
        self.stats = {
            'total_users': 0,
            'total_items': 0,
            'total_interactions': 0,
            'model_version': 0,
            'last_training': None
        }
        
        # Load existing model
        self._load_model()

    def add_interaction(self, user_id: int, item_id: int, rating: float, 
                       implicit: bool = True):
        """Yeni user-item etkileşimini ekle"""
        try:
            # Get current interactions
            interactions = self._get_interactions()
            
            # Add new interaction
            interaction = {
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating,
                'implicit': implicit,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            interactions.append(interaction)
            
            # Save back to Redis
            self._save_interactions(interactions)
            
            # Update stats
            self.stats['total_interactions'] += 1
            self._save_stats()
            
            logger.info("interaction_added", 
                       user_id=user_id, 
                       item_id=item_id, 
                       rating=rating)
            
        except Exception as e:
            logger.error("add_interaction_error", error=str(e))

    def get_recommendations(self, user_id: int, n_recommendations: int = 10,
                          exclude_seen: bool = True) -> List[Dict]:
        """Kullanıcı için öneriler üret"""
        try:
            if self.nmf_model is None:
                logger.warning("no_trained_model_available")
                return self._fallback_recommendations(user_id, n_recommendations)
            
            # Get user index
            if user_id not in self.user_to_index:
                logger.info("new_user_cold_start", user_id=user_id)
                return self._cold_start_recommendations(user_id, n_recommendations)
            
            user_idx = self.user_to_index[user_id]
            
            # Get user profile from NMF
            user_profile = self.nmf_model.transform(
                self.interaction_matrix[user_idx:user_idx+1]
            )[0]
            
            # Calculate scores for all items
            item_scores = self.nmf_model.components_.T.dot(user_profile)
            
            # Get top recommendations
            item_indices = np.argsort(item_scores)[::-1]
            
            recommendations = []
            seen_items = set()
            
            if exclude_seen:
                # Get user's seen items
                user_interactions = self.interaction_matrix[user_idx].nonzero()[1]
                seen_items = {self.index_to_item[idx] for idx in user_interactions}
            
            for item_idx in item_indices:
                if len(recommendations) >= n_recommendations:
                    break
                
                item_id = self.index_to_item.get(item_idx)
                if item_id is None:
                    continue
                    
                if exclude_seen and item_id in seen_items:
                    continue
                
                recommendations.append({
                    'item_id': item_id,
                    'score': float(item_scores[item_idx]),
                    'method': 'collaborative_filtering'
                })
            
            logger.info("recommendations_generated", 
                       user_id=user_id, 
                       count=len(recommendations))
            
            return recommendations
            
        except Exception as e:
            logger.error("recommendation_error", error=str(e))
            return self._fallback_recommendations(user_id, n_recommendations)

    def get_similar_items(self, item_id: int, n_similar: int = 10) -> List[Dict]:
        """Benzer item'ları bul"""
        try:
            if self.nmf_model is None or item_id not in self.item_to_index:
                return []
            
            item_idx = self.item_to_index[item_id]
            
            # Get item profile
            item_profile = self.nmf_model.components_[:, item_idx].reshape(1, -1)
            
            # Calculate similarity with all items
            similarities = cosine_similarity(
                item_profile, 
                self.nmf_model.components_.T
            )[0]
            
            # Get most similar items (excluding itself)
            similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
            
            similar_items = []
            for idx in similar_indices:
                similar_item_id = self.index_to_item.get(idx)
                if similar_item_id:
                    similar_items.append({
                        'item_id': similar_item_id,
                        'similarity': float(similarities[idx])
                    })
            
            return similar_items
            
        except Exception as e:
            logger.error("similar_items_error", error=str(e))
            return []

    def train_model(self, min_interactions: int = 10):
        """Collaborative filtering modelini eğit"""
        try:
            interactions = self._get_interactions()
            
            if len(interactions) < min_interactions:
                logger.warning("insufficient_interactions", 
                              count=len(interactions),
                              min_required=min_interactions)
                return False
            
            # Create interaction matrix
            self._build_interaction_matrix(interactions)
            
            # Train NMF model
            self.nmf_model = NMF(
                n_components=self.n_components,
                init='random',
                random_state=42,
                max_iter=200
            )
            
            self.nmf_model.fit(self.interaction_matrix)
            
            # Train SVD model for diversity
            self.svd_model = TruncatedSVD(
                n_components=min(self.n_components, self.interaction_matrix.shape[1]-1),
                random_state=42
            )
            
            self.svd_model.fit(self.interaction_matrix)
            
            # Update stats
            self.stats['model_version'] += 1
            self.stats['last_training'] = pd.Timestamp.now().isoformat()
            self._save_stats()
            
            # Save model
            self._save_model()
            
            logger.info("model_trained", 
                       n_users=self.interaction_matrix.shape[0],
                       n_items=self.interaction_matrix.shape[1],
                       n_interactions=len(interactions))
            
            return True
            
        except Exception as e:
            logger.error("model_training_error", error=str(e))
            return False

    def _build_interaction_matrix(self, interactions: List[Dict]):
        """Interaction matrix oluştur"""
        df = pd.DataFrame(interactions)
        
        # Create user and item mappings
        unique_users = df['user_id'].unique()
        unique_items = df['item_id'].unique()
        
        self.user_to_index = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_index = {item: idx for idx, item in enumerate(unique_items)}
        self.index_to_user = {idx: user for user, idx in self.user_to_index.items()}
        self.index_to_item = {idx: item for item, idx in self.item_to_index.items()}
        
        # Build sparse matrix
        rows = [self.user_to_index[user] for user in df['user_id']]
        cols = [self.item_to_index[item] for item in df['item_id']]
        data = df['rating'].values
        
        self.interaction_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(unique_users), len(unique_items))
        )
        
        # Update stats
        self.stats['total_users'] = len(unique_users)
        self.stats['total_items'] = len(unique_items)

    def _cold_start_recommendations(self, user_id: int, n_recommendations: int) -> List[Dict]:
        """Yeni kullanıcılar için cold start önerileri"""
        try:
            # Get popular items (highest average ratings)
            interactions = self._get_interactions()
            df = pd.DataFrame(interactions)
            
            if df.empty:
                return []
            
            popular_items = (
                df.groupby('item_id')['rating']
                .agg(['mean', 'count'])
                .query('count >= 3')  # En az 3 rating
                .sort_values('mean', ascending=False)
                .head(n_recommendations)
            )
            
            recommendations = []
            for item_id, row in popular_items.iterrows():
                recommendations.append({
                    'item_id': item_id,
                    'score': float(row['mean']),
                    'method': 'popularity_based'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error("cold_start_error", error=str(e))
            return []

    def _fallback_recommendations(self, user_id: int, n_recommendations: int) -> List[Dict]:
        """Fallback öneriler"""
        try:
            interactions = self._get_interactions()
            if not interactions:
                return []
            
            # Random sampling from available items
            df = pd.DataFrame(interactions)
            item_ids = df['item_id'].unique()
            
            if len(item_ids) <= n_recommendations:
                selected_items = item_ids
            else:
                selected_items = np.random.choice(
                    item_ids, 
                    size=n_recommendations, 
                    replace=False
                )
            
            recommendations = []
            for item_id in selected_items:
                recommendations.append({
                    'item_id': item_id,
                    'score': 0.5,
                    'method': 'random_fallback'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error("fallback_error", error=str(e))
            return []

    def _get_interactions(self) -> List[Dict]:
        """Etkileşimleri Redis'ten al"""
        try:
            data = self.redis.get(self.interaction_matrix_key)
            if data:
                return json.loads(data)
            return []
        except Exception as e:
            logger.error("get_interactions_error", error=str(e))
            return []

    def _save_interactions(self, interactions: List[Dict]):
        """Etkileşimleri Redis'e kaydet"""
        try:
            self.redis.setex(
                self.interaction_matrix_key, 
                86400 * 7,  # 1 week
                json.dumps(interactions)
            )
        except Exception as e:
            logger.error("save_interactions_error", error=str(e))

    def _save_model(self):
        """Modeli Redis'e kaydet"""
        try:
            model_data = {
                'nmf_model': pickle.dumps(self.nmf_model).hex() if self.nmf_model else None,
                'svd_model': pickle.dumps(self.svd_model).hex() if self.svd_model else None,
                'user_to_index': self.user_to_index,
                'item_to_index': self.item_to_index,
                'index_to_user': self.index_to_user,
                'index_to_item': self.index_to_item,
                'interaction_matrix': pickle.dumps(self.interaction_matrix).hex() if self.interaction_matrix is not None else None
            }
            
            self.redis.setex(self.model_key, 86400 * 7, json.dumps(model_data))
            logger.info("model_saved")
            
        except Exception as e:
            logger.error("model_save_error", error=str(e))

    def _load_model(self):
        """Modeli Redis'ten yükle"""
        try:
            data = self.redis.get(self.model_key)
            if not data:
                return
            
            model_data = json.loads(data)
            
            if model_data.get('nmf_model'):
                self.nmf_model = pickle.loads(bytes.fromhex(model_data['nmf_model']))
            
            if model_data.get('svd_model'):
                self.svd_model = pickle.loads(bytes.fromhex(model_data['svd_model']))
            
            self.user_to_index = model_data.get('user_to_index', {})
            self.item_to_index = model_data.get('item_to_index', {})
            self.index_to_user = model_data.get('index_to_user', {})
            self.index_to_item = model_data.get('index_to_item', {})
            
            if model_data.get('interaction_matrix'):
                self.interaction_matrix = pickle.loads(bytes.fromhex(model_data['interaction_matrix']))
            
            # Load stats
            stats_data = self.redis.get(self.stats_key)
            if stats_data:
                self.stats.update(json.loads(stats_data))
            
            logger.info("model_loaded", 
                       has_nmf=self.nmf_model is not None,
                       n_users=len(self.user_to_index),
                       n_items=len(self.item_to_index))
            
        except Exception as e:
            logger.info("model_load_failed", error=str(e))

    def _save_stats(self):
        """İstatistikleri kaydet"""
        try:
            self.redis.setex(self.stats_key, 86400, json.dumps(self.stats))
        except Exception as e:
            logger.error("stats_save_error", error=str(e))

    def get_statistics(self) -> Dict:
        """Model istatistiklerini al"""
        stats = self.stats.copy()
        stats['has_model'] = self.nmf_model is not None
        stats['n_components'] = self.n_components
        return stats

    def reset(self):
        """Modeli sıfırla"""
        try:
            self.nmf_model = None
            self.svd_model = None
            self.interaction_matrix = None
            self.user_to_index.clear()
            self.item_to_index.clear()
            self.index_to_user.clear()
            self.index_to_item.clear()
            
            self.stats = {
                'total_users': 0,
                'total_items': 0,
                'total_interactions': 0,
                'model_version': 0,
                'last_training': None
            }
            
            # Clear Redis
            self.redis.delete(
                self.model_key,
                self.interaction_matrix_key,
                self.stats_key
            )
            
            logger.info("collaborative_filter_reset")
            
        except Exception as e:
            logger.error("reset_error", error=str(e))