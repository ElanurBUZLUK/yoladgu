"""
Ensemble Scoring Service
Farklı model skorlarını birleştirerek daha iyi öneriler üretir
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import structlog
from app.core.config import settings
from app.services.embedding_service import find_similar_questions, compute_embedding
from app.crud.question import get_question_skill_centrality
from app.crud.student_response import get_student_skill_mastery_from_neo4j
import asyncio

logger = structlog.get_logger()

class EnsembleScoringService:
    def __init__(self):
        # Model ağırlıkları (ayarlanabilir)
        self.weights = {
            'river_score': 0.35,           # River ML model skoru
            'embedding_similarity': 0.25,  # SBERT embedding benzerliği
            'skill_mastery': 0.20,         # Skill mastery uyumu
            'difficulty_match': 0.15,      # Zorluk seviyesi uyumu
            'neo4j_similarity': 0.05       # Neo4j graph benzerliği
        }
        
        # Threshold değerleri
        self.thresholds = {
            'min_similarity': 0.6,         # Minimum embedding benzerliği
            'max_difficulty_gap': 2,       # Maksimum zorluk farkı
            'min_skill_mastery': 0.3       # Minimum skill mastery
        }
    
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
            similar_questions = []
            for recent_id in student_recent_question_ids[-3:]:  # Son 3 soru
                from app.crud.question import get_similar_questions_from_neo4j
                similar_ids = get_similar_questions_from_neo4j(recent_id, limit=5)
                similar_questions.extend(similar_ids)
            
            # Mevcut sorunun benzerlik skorunu hesapla
            if question_id in similar_questions:
                # Benzerlik sıklığına göre skor
                frequency = similar_questions.count(question_id)
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
ensemble_service = EnsembleScoringService()

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