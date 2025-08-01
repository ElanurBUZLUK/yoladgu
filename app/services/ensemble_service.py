"""
Enhanced Ensemble Scoring Service
Farklı ML modellerini birleştirerek optimal öneriler üretir:
- River (Online Learning)
- LinUCB (Contextual Bandit) 
- Collaborative Filtering
- Embedding Similarity
- Neo4j Graph Analysis
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import structlog
import redis
from app.core.config import settings
from app.services.embedding_service import find_similar_questions, compute_embedding
from app.crud.question import get_question_skill_centrality
from app.crud.student_response import get_student_skill_mastery_from_neo4j
from app.ml.online_learner import EnhancedOnlineLearner
from app.ml.bandits import EnhancedLinUCBBandit
from app.ml.collaborative_filter import CollaborativeFilterEngine
from app.ml.feature_extractor import UserFeatureExtractor
import asyncio

logger = structlog.get_logger()

class EnhancedEnsembleScoringService:
    def __init__(self, redis_client=None):
        # Redis client
        if redis_client is None:
            self.redis = redis.from_url(settings.redis_url)
        else:
            self.redis = redis_client
        
        # Initialize ML models
        self.river_learner = EnhancedOnlineLearner(self.redis)
        self.bandit = EnhancedLinUCBBandit(self.redis)
        self.collaborative_filter = CollaborativeFilterEngine(self.redis)
        
        # Model ağırlıkları (dinamik olarak ayarlanabilir)
        self.weights = settings.ENSEMBLE_WEIGHTS if hasattr(settings, 'ENSEMBLE_WEIGHTS') else {
            'river_score': 0.25,           # River online learning
            'bandit_score': 0.20,          # LinUCB bandit
            'collaborative_score': 0.20,   # Collaborative filtering
            'embedding_similarity': 0.15,  # SBERT embedding benzerliği
            'skill_mastery': 0.10,         # Skill mastery uyumu
            'difficulty_match': 0.05,      # Zorluk seviyesi uyumu
            'neo4j_similarity': 0.05       # Neo4j graph benzerliği
        }
        
        # Threshold değerleri
        self.thresholds = {
            'min_similarity': 0.6,         # Minimum embedding benzerliği
            'max_difficulty_gap': 2,       # Maksimum zorluk farkı
            'min_skill_mastery': 0.3,      # Minimum skill mastery
            'min_confidence': 0.1          # Minimum prediction confidence
        }
    
    def calculate_comprehensive_ensemble_score(self, 
                                             user_features: Dict,
                                             question_features: Dict,
                                             candidate_questions: List[Dict],
                                             context: Optional[Dict] = None) -> List[Dict]:
        """Tüm ML modellerini birleştirerek kapsamlı skor hesapla"""
        try:
            scored_questions = []
            
            for question in candidate_questions:
                question_id = question['id']
                
                # Initialize scores
                scores = {
                    'river_score': 0.5,
                    'bandit_score': 0.5,
                    'collaborative_score': 0.5,
                    'embedding_similarity': 0.5,
                    'skill_mastery': 0.5,
                    'difficulty_match': 0.5,
                    'neo4j_similarity': 0.5
                }
                
                # 1. River online learning score
                try:
                    scores['river_score'] = self.river_learner.predict_success_probability(
                        user_features, question_features
                    )
                except Exception as e:
                    logger.debug("river_score_error", error=str(e))
                
                # 2. LinUCB bandit score
                try:
                    if len(candidate_questions) > 1:
                        selected_id, selection_info = self.bandit.select_question(
                            user_features, candidate_questions, context
                        )
                        if selected_id == question_id:
                            scores['bandit_score'] = selection_info.get('ucb_value', 0.5)
                        else:
                            scores['bandit_score'] = selection_info.get('expected_reward', 0.3)
                except Exception as e:
                    logger.debug("bandit_score_error", error=str(e))
                
                # 3. Collaborative filtering score
                try:
                    user_id = user_features.get('user_id')
                    if user_id:
                        cf_recommendations = self.collaborative_filter.get_recommendations(
                            user_id, n_recommendations=20, exclude_seen=True
                        )
                        cf_scores = {rec['item_id']: rec['score'] for rec in cf_recommendations}
                        scores['collaborative_score'] = cf_scores.get(question_id, 0.3)
                except Exception as e:
                    logger.debug("collaborative_score_error", error=str(e))
                
                # 4. Embedding similarity score
                try:
                    question_content = question.get('content', '')
                    recent_questions = context.get('recent_questions', []) if context else []
                    scores['embedding_similarity'] = self.calculate_embedding_similarity_score(
                        question_content, recent_questions
                    )
                except Exception as e:
                    logger.debug("embedding_score_error", error=str(e))
                
                # 5. Skill mastery score
                try:
                    scores['skill_mastery'] = self.calculate_skill_mastery_score(
                        question_id, user_features.get('user_id', 0),
                        user_features.get('topic_mastery_math', 0.5)
                    )
                except Exception as e:
                    logger.debug("skill_mastery_error", error=str(e))
                
                # 6. Difficulty match score
                try:
                    scores['difficulty_match'] = self.calculate_difficulty_match_score(
                        question_features.get('difficulty_level', 3),
                        user_features.get('accuracy_rate_overall', 0.5)
                    )
                except Exception as e:
                    logger.debug("difficulty_match_error", error=str(e))
                
                # 7. Neo4j similarity score
                try:
                    scores['neo4j_similarity'] = self.calculate_neo4j_similarity_score(
                        question_id, context.get('recent_question_ids', []) if context else []
                    )
                except Exception as e:
                    logger.debug("neo4j_similarity_error", error=str(e))
                
                # Calculate final ensemble score
                final_score = sum(
                    self.weights[score_type] * score_value 
                    for score_type, score_value in scores.items()
                    if score_type in self.weights
                )
                
                # Add diversity bonus if enabled
                if settings.USE_DIVERSITY_FILTER:
                    diversity_bonus = self._calculate_diversity_bonus(
                        question, context.get('recent_questions', []) if context else []
                    )
                    final_score += diversity_bonus * 0.1
                
                scored_question = {
                    **question,
                    'ensemble_score': final_score,
                    'component_scores': scores,
                    'selection_method': 'enhanced_ensemble'
                }
                
                scored_questions.append(scored_question)
            
            # Sort by ensemble score
            scored_questions.sort(key=lambda x: x['ensemble_score'], reverse=True)
            
            logger.info("ensemble_scoring_completed", 
                       n_questions=len(scored_questions),
                       top_score=scored_questions[0]['ensemble_score'] if scored_questions else 0)
            
            return scored_questions
            
        except Exception as e:
            logger.error("comprehensive_ensemble_error", error=str(e))
            return candidate_questions  # Fallback to original list

    def update_models_with_feedback(self,
                                   user_features: Dict,
                                   question_features: Dict, 
                                   is_correct: bool,
                                   response_time: int,
                                   context: Optional[Dict] = None):
        """Öğrenci geri bildirimi ile tüm modelleri güncelle"""
        try:
            # 1. Update River online learner
            self.river_learner.learn_from_answer(
                user_features, question_features, is_correct, response_time
            )
            
            # 2. Update LinUCB bandit
            reward = 1.0 if is_correct else 0.0
            # Add response time bonus/penalty
            if response_time < user_features.get('avg_response_time', 5000):
                reward += 0.1  # Quick response bonus
            elif response_time > user_features.get('avg_response_time', 5000) * 2:
                reward -= 0.1  # Slow response penalty
            
            self.bandit.update(
                question_features.get('question_id'),
                user_features, question_features, reward, context
            )
            
            # 3. Update collaborative filter
            user_id = user_features.get('user_id')
            question_id = question_features.get('question_id') 
            if user_id and question_id:
                # Convert boolean to rating scale
                rating = 1.0 if is_correct else 0.0
                self.collaborative_filter.add_interaction(
                    user_id, question_id, rating, implicit=True
                )
            
            logger.info("models_updated_with_feedback",
                       user_id=user_id,
                       question_id=question_id,
                       is_correct=is_correct,
                       response_time=response_time)
            
        except Exception as e:
            logger.error("model_update_error", error=str(e))

    def train_collaborative_model(self, min_interactions: int = 50):
        """Collaborative filtering modelini eğit"""
        try:
            success = self.collaborative_filter.train_model(min_interactions)
            logger.info("collaborative_model_training", success=success)
            return success
        except Exception as e:
            logger.error("collaborative_training_error", error=str(e))
            return False

    def get_model_statistics(self) -> Dict:
        """Tüm modellerin istatistiklerini al"""
        try:
            return {
                'river_stats': self.river_learner.get_performance_metrics(),
                'bandit_stats': self.bandit.get_statistics(),
                'collaborative_stats': self.collaborative_filter.get_statistics(),
                'ensemble_weights': self.weights,
                'thresholds': self.thresholds
            }
        except Exception as e:
            logger.error("stats_collection_error", error=str(e))
            return {}

    def _calculate_diversity_bonus(self, question: Dict, recent_questions: List[str]) -> float:
        """Çeşitlilik bonusu hesapla"""
        try:
            if not recent_questions:
                return 0.5
            
            question_content = question.get('content', '')
            if not question_content:
                return 0.0
            
            # Simple diversity check based on content overlap
            overlaps = []
            for recent_q in recent_questions[-3:]:  # Son 3 soru
                common_words = set(question_content.lower().split()) & set(recent_q.lower().split())
                overlap_ratio = len(common_words) / max(len(question_content.split()), 1)
                overlaps.append(overlap_ratio)
            
            avg_overlap = np.mean(overlaps) if overlaps else 0
            diversity_score = 1.0 - avg_overlap  # Lower overlap = higher diversity
            
            return max(0.0, min(1.0, diversity_score))
            
        except Exception as e:
            logger.debug("diversity_calculation_error", error=str(e))
            return 0.5

    def calculate_embedding_similarity_score(self, question_content: str, 
                                           student_recent_questions: List[str]) -> float:
        """Embedding benzerlik skoru hesapla"""
        if not student_recent_questions:
            return 0.5  # Varsayılan skor
        
        try:
            # Öğrencinin son sorularının embedding'lerini hesapla
            recent_embeddings = [compute_embedding(q) for q in student_recent_questions]
            
            # Mevcut sorunun embedding'ini hesapla
            question_embedding = compute_embedding(question_content)
            
            # Ortalama benzerlik hesapla
            similarities = []
            for recent_emb in recent_embeddings:
                if recent_emb and question_embedding:
                    # Cosine similarity
                    similarity = np.dot(recent_emb, question_embedding) / (
                        np.linalg.norm(recent_emb) * np.linalg.norm(question_embedding) + 1e-8
                    )
                    similarities.append(similarity)
            
            if similarities:
                return np.mean(similarities)
            else:
                return 0.5
                
        except Exception as e:
            logger.error("embedding_similarity_error", error=str(e))
            return 0.5
    
    def calculate_skill_mastery_score(self, question_id: int, student_id: int) -> float:
        """Skill mastery uyum skoru hesapla"""
        try:
            # Sorunun skill centrality'sini al
            skill_centrality = get_question_skill_centrality(question_id)
            
            # Öğrencinin skill mastery'sini al
            student_mastery = get_student_skill_mastery_from_neo4j(student_id)
            
            if not skill_centrality or not student_mastery:
                return 0.5
            
            # Skill bazında uyum hesapla
            total_weight = 0
            weighted_score = 0
            
            for skill_id, centrality in skill_centrality.items():
                if skill_id in student_mastery:
                    mastery = student_mastery[skill_id]['mastery']
                    weight = centrality
                    
                    # Mastery düşükse daha yüksek skor (öğrenme fırsatı)
                    skill_score = 1 - mastery
                    
                    weighted_score += skill_score * weight
                    total_weight += weight
            
            if total_weight > 0:
                return weighted_score / total_weight
            else:
                return 0.5
                
        except Exception as e:
            logger.error("skill_mastery_score_error", error=str(e))
            return 0.5
    
    def calculate_difficulty_match_score(self, question_difficulty: int, 
                                       student_level: float, 
                                       student_recent_performance: float) -> float:
        """Zorluk seviyesi uyum skoru hesapla"""
        try:
            # Öğrenci seviyesine göre ideal zorluk
            ideal_difficulty = max(1, min(5, round(student_level)))
            
            # Performansa göre zorluk ayarı
            if student_recent_performance > 0.8:  # Yüksek performans
                ideal_difficulty = min(5, ideal_difficulty + 1)
            elif student_recent_performance < 0.4:  # Düşük performans
                ideal_difficulty = max(1, ideal_difficulty - 1)
            
            # Zorluk farkına göre skor hesapla
            difficulty_gap = abs(question_difficulty - ideal_difficulty)
            
            if difficulty_gap == 0:
                return 1.0
            elif difficulty_gap == 1:
                return 0.8
            elif difficulty_gap == 2:
                return 0.6
            else:
                return 0.3
                
        except Exception as e:
            logger.error("difficulty_match_error", error=str(e))
            return 0.5
    
    def calculate_neo4j_similarity_score(self, question_id: int, 
                                       student_recent_question_ids: List[int]) -> float:
        """Neo4j graph benzerlik skoru hesapla"""
        if not student_recent_question_ids:
            return 0.5
        
        try:
            # Son soruların benzer sorularını bul
            similar_question_ids = []
            for recent_id in student_recent_question_ids[-3:]:  # Son 3 soru
                from app.crud.question import get_similar_questions_from_neo4j
                similar_data = get_similar_questions_from_neo4j(recent_id, limit=5)
                # get_similar_questions_from_neo4j returns List[Dict] with question_id key
                for item in similar_data:
                    if isinstance(item, dict) and 'question_id' in item:
                        similar_question_ids.append(item['question_id'])
                    elif isinstance(item, int):  # Fallback for direct IDs
                        similar_question_ids.append(item)
            
            # Mevcut sorunun benzerlik skorunu hesapla
            if question_id in similar_question_ids:
                # Benzerlik sıklığına göre skor
                frequency = similar_question_ids.count(question_id)
                return min(1.0, frequency * 0.3)
            else:
                return 0.3
                
        except Exception as e:
            logger.error("neo4j_similarity_error", error=str(e))
            return 0.5
    
    def calculate_ensemble_score(self, 
                               river_score: float,
                               question_content: str,
                               question_id: int,
                               question_difficulty: int,
                               student_id: int,
                               student_level: float,
                               student_recent_performance: float,
                               student_recent_questions: List[str],
                               student_recent_question_ids: List[int]) -> Dict[str, float]:
        """Ensemble skor hesapla"""
        try:
            # Her bileşenin skorunu hesapla
            embedding_score = self.calculate_embedding_similarity_score(
                question_content, student_recent_questions
            )
            
            skill_mastery_score = self.calculate_skill_mastery_score(question_id, student_id)
            
            difficulty_score = self.calculate_difficulty_match_score(
                question_difficulty, student_level, student_recent_performance
            )
            
            neo4j_score = self.calculate_neo4j_similarity_score(
                question_id, student_recent_question_ids
            )
            
            # Ağırlıklı toplam hesapla
            ensemble_score = (
                self.weights['river_score'] * river_score +
                self.weights['embedding_similarity'] * embedding_score +
                self.weights['skill_mastery'] * skill_mastery_score +
                self.weights['difficulty_match'] * difficulty_score +
                self.weights['neo4j_similarity'] * neo4j_score
            )
            
            # Bileşen skorlarını döndür
            component_scores = {
                'ensemble_score': ensemble_score,
                'river_score': river_score,
                'embedding_similarity': embedding_score,
                'skill_mastery': skill_mastery_score,
                'difficulty_match': difficulty_score,
                'neo4j_similarity': neo4j_score
            }
            
            logger.info("ensemble_score_calculated", 
                       question_id=question_id, 
                       student_id=student_id,
                       ensemble_score=ensemble_score)
            
            return component_scores
            
        except Exception as e:
            logger.error("ensemble_score_error", error=str(e))
            return {
                'ensemble_score': river_score,  # Fallback to river score
                'river_score': river_score,
                'embedding_similarity': 0.5,
                'skill_mastery': 0.5,
                'difficulty_match': 0.5,
                'neo4j_similarity': 0.5
            }
    
    def filter_questions_by_thresholds(self, questions: List[Dict], 
                                     student_level: float,
                                     student_recent_performance: float) -> List[Dict]:
        """Threshold'lara göre soruları filtrele"""
        filtered_questions = []
        
        for question in questions:
            # Zorluk farkı kontrolü
            difficulty_gap = abs(question.get('difficulty_level', 3) - student_level)
            if difficulty_gap > self.thresholds['max_difficulty_gap']:
                continue
            
            # Embedding benzerlik kontrolü (eğer embedding varsa)
            if 'embedding_similarity' in question:
                if question['embedding_similarity'] < self.thresholds['min_similarity']:
                    continue
            
            # Skill mastery kontrolü
            if 'skill_mastery' in question:
                if question['skill_mastery'] < self.thresholds['min_skill_mastery']:
                    continue
            
            filtered_questions.append(question)
        
        return filtered_questions
    
    def adjust_weights_dynamically(self, student_performance: float, 
                                 question_count: int) -> None:
        """Öğrenci performansına göre ağırlıkları dinamik olarak ayarla"""
        try:
            # Performans düşükse embedding benzerliğini artır
            if student_performance < 0.4:
                self.weights['embedding_similarity'] = 0.35
                self.weights['river_score'] = 0.25
            # Performans yüksekse skill mastery'yi artır
            elif student_performance > 0.8:
                self.weights['skill_mastery'] = 0.30
                self.weights['river_score'] = 0.30
            # Normal performans
            else:
                self.weights = {
                    'river_score': 0.35,
                    'embedding_similarity': 0.25,
                    'skill_mastery': 0.20,
                    'difficulty_match': 0.15,
                    'neo4j_similarity': 0.05
                }
            
            # Soru sayısına göre threshold ayarı
            if question_count < 10:
                self.thresholds['min_similarity'] = 0.5  # Daha esnek
            else:
                self.thresholds['min_similarity'] = 0.6  # Daha katı
                
        except Exception as e:
            logger.error("weight_adjustment_error", error=str(e))

# Global instance
ensemble_service = EnhancedEnsembleScoringService()

# Convenience functions
def calculate_ensemble_score(river_score: float,
                           question_content: str,
                           question_id: int,
                           question_difficulty: int,
                           student_id: int,
                           student_level: float,
                           student_recent_performance: float,
                           student_recent_questions: List[str],
                           student_recent_question_ids: List[int]) -> Dict[str, float]:
    """Ensemble skor hesapla"""
    return ensemble_service.calculate_ensemble_score(
        river_score, question_content, question_id, question_difficulty,
        student_id, student_level, student_recent_performance,
        student_recent_questions, student_recent_question_ids
    )

def filter_questions_by_thresholds(questions: List[Dict], 
                                 student_level: float,
                                 student_recent_performance: float) -> List[Dict]:
    """Threshold'lara göre soruları filtrele"""
    return ensemble_service.filter_questions_by_thresholds(
        questions, student_level, student_recent_performance
    )

def adjust_weights_dynamically(student_performance: float, question_count: int) -> None:
    """Ağırlıkları dinamik olarak ayarla"""
    ensemble_service.adjust_weights_dynamically(student_performance, question_count) 