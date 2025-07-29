from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from app.db.models import User, Question, StudentResponse, Skill, QuestionSkill
from app.core.config import settings
import redis
import json
# 'feature_extraction' importu artık gerekli değil ama kalsa da zararı yok
from river import compose, linear_model, preprocessing, feature_extraction
import os
from app.services.level_service import update_student_level
from app.ml.bandits import LinUCBBandit
from neo4j import GraphDatabase
import structlog
from prometheus_client import Counter, Histogram
model_update_counter = Counter("model_update_total", "Model güncelleme sayısı")
model_update_duration = Histogram("model_update_duration_seconds", "Model update süresi", ["subject", "environment"])

from sentence_transformers import SentenceTransformer
import threading

# Modeli thread-safe şekilde cache'le
_model = None
_model_lock = threading.Lock()
def get_sbert_model():
    global _model
    with _model_lock:
        if _model is None:
            _model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        return _model

# --- River pipeline ---
def build_river_model():
    return compose.Pipeline(
        # HATA DÜZELTİLDİ: OneHotEncoder 'preprocessing' modülünden çağrılıyor
        preprocessing.OneHotEncoder(),
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression()
    )

def get_skill_centrality(student_id):
    driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
    with driver.session() as session:
        res = session.run(
            """
            MATCH (u:User {id: $student_id})-[:SOLVED]->(q:Question)-[:HAS_SKILL]->(s:Skill)
            RETURN s.id AS skill_id, count(*) AS freq
            """,
            student_id=student_id
        )
        centrality = {r["skill_id"]: r["freq"] for r in res}
    driver.close()
    return centrality

def diversity_filter(candidates, question_embeddings, top_n=5):
    # candidates: [(question_id, score), ...]
    # question_embeddings: {question_id: embedding_vector}
    # En yakın top_n soruyu diskalifiye et, kalanlardan rastgele seç
    import numpy as np
    if len(candidates) <= top_n:
        return candidates
    # İlk sorunun embedding'ini referans al
    ref_id = candidates[0][0]
    ref_emb = question_embeddings.get(ref_id)
    if ref_emb is None:
        return candidates[top_n:]
    # Benzerlikleri hesapla
    sims = []
    for qid, _ in candidates:
        emb = question_embeddings.get(qid)
        if emb is not None:
            sim = np.dot(ref_emb, emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(emb) + 1e-8)
        else:
            sim = 0
        sims.append((qid, sim))
    # En benzer top_n soruyu bul
    sims = sorted(sims, key=lambda x: -x[1])
    filtered = [c for c in candidates if c[0] not in [qid for qid, _ in sims[:top_n]]]
    return filtered

def add_question_to_neo4j(question_id, skill_ids):
    driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
    with driver.session() as session:
        for skill_id in skill_ids:
            session.run(
                "MERGE (q:Question {id: $qid}) "
                "MERGE (s:Skill {id: $sid}) "
                "MERGE (q)-[:HAS_SKILL]->(s)",
                qid=question_id, sid=skill_id
            )
    driver.close()

# Embedding hesaplama artık embedding_service'den geliyor
from app.services.embedding_service import compute_embedding
from app.services.ensemble_service import calculate_ensemble_score, filter_questions_by_thresholds, adjust_weights_dynamically

def get_logger_with_context(**context):
    return structlog.get_logger().bind(**context)

class RecommendationService:
    def __init__(self, model_type: str = 'river'):
        self.redis_client = redis.Redis.from_url(settings.redis_url)
        self.model_type = model_type
        self.river_model = build_river_model()
        self.linucb_bandit = LinUCBBandit(alpha=1.0, feature_dim=10)
        self.model_cache_dir = settings.MODEL_CACHE_DIR
        os.makedirs(self.model_cache_dir, exist_ok=True)
        self._load_model()

    def _load_model(self):
        river_path = os.path.join(self.model_cache_dir, "river_model.bin")
        linucb_path = os.path.join(self.model_cache_dir, "linucb_model.npz")
        if os.path.exists(river_path):
            with open(river_path, "rb") as f:
                self.river_model = build_river_model().from_bytes(f.read())
        if os.path.exists(linucb_path):
            data = np.load(linucb_path, allow_pickle=True)
            self.linucb_bandit.A = data['A'].item()
            self.linucb_bandit.b = data['b'].item()

    def _save_model(self):
        river_path = os.path.join(self.model_cache_dir, "river_model.bin")
        linucb_path = os.path.join(self.model_cache_dir, "linucb_model.npz")
        with open(river_path, "wb") as f:
            f.write(self.river_model.to_bytes())
        np.savez(linucb_path, A=self.linucb_bandit.A, b=self.linucb_bandit.b)

    def get_student_features(self, db: Session, student_id: int) -> Dict[str, Any]:
        # HATA DÜZELTİLDİ: 'timestamp' yerine 'created_at' kullanılıyor
        responses = db.query(StudentResponse).filter(
            StudentResponse.student_id == student_id
        ).order_by(StudentResponse.created_at.desc()).all()
        features = {
            'total_questions': len(responses),
            'correct_answers': sum(1 for r in responses if r.is_correct),
            'avg_response_time': np.mean([r.response_time for r in responses if r.response_time]) if responses else 0,
            'avg_confidence': np.mean([r.confidence_level for r in responses if r.confidence_level]) if responses else 0,
            'hour_of_day': responses[0].created_at.hour if responses else 0,
        }
        # Son 20 soruda başarı oranı (sliding window)
        last_n = 20
        last_n_responses = responses[:last_n]
        features['last20_accuracy'] = np.mean([r.is_correct for r in last_n_responses]) if last_n_responses else 0
        # Konu başına doğruluk oranı
        topic_correct = {}
        topic_total = {}
        for r in responses:
            q = db.query(Question).filter(Question.id == r.question_id).first()
            if q:
                topic = q.subject_id
                topic_total[topic] = topic_total.get(topic, 0) + 1
                if r.is_correct:
                    topic_correct[topic] = topic_correct.get(topic, 0) + 1
        for topic in topic_total:
            features[f'topic_{topic}_accuracy'] = topic_correct.get(topic, 0) / topic_total[topic]
        # Skill mastery
        skill_mastery = {}
        for response in responses:
            question_skills = db.query(QuestionSkill).filter(
                QuestionSkill.question_id == response.question_id
            ).all()
            for qs in question_skills:
                skill_id = qs.skill_id
                if skill_id not in skill_mastery:
                    skill_mastery[skill_id] = {'correct': 0, 'total': 0}
                skill_mastery[skill_id]['total'] += 1
                if response.is_correct:
                    skill_mastery[skill_id]['correct'] += 1
        for skill_id, mastery in skill_mastery.items():
            features[f'skill_{skill_id}_mastery'] = mastery['correct'] / mastery['total'] if mastery['total'] > 0 else 0
        # --- Neo4j skill centrality feature ---
        try:
            centrality = get_skill_centrality(student_id)
            for skill_id, freq in centrality.items():
                features[f'skill_{skill_id}_centrality'] = freq
        except Exception as e:
            pass
        return features

    def get_question_features(self, db: Session, question_id: int) -> Dict[str, Any]:
        question = db.query(Question).filter(Question.id == question_id).first()
        if not question:
            return {}
        question_skills = db.query(QuestionSkill).filter(
            QuestionSkill.question_id == question_id
        ).all()
        features = {
            'difficulty_level': question.difficulty_level,
            'question_type': question.question_type,
            'subject_id': question.subject_id,
            'skill_count': len(question_skills),
            # HATA DÜZELTİLDİ: 'text' yerine 'content' kullanılıyor
            'question_length': len(question.content.split()) if hasattr(question, 'content') else 0,
            'avg_skill_difficulty': np.mean([qs.skill.difficulty_level for qs in question_skills]) if question_skills else 0,
            'num_tags': len(question.tags) if hasattr(question, 'tags') else 0,
        }
        for qs in question_skills:
            features[f'skill_{qs.skill_id}_weight'] = qs.weight
        # Gömme benzerliği (varsa)
        if hasattr(question, 'bert_sim'):
            features['bert_sim'] = question.bert_sim
        return features

    def update_river_model(self, student_features: Dict[str, Any], question_features: Dict[str, Any],
                          is_correct: bool, response_time: float):
        combined_features = {**student_features, **question_features}
        x = combined_features
        y = 1 if is_correct else 0
        self.river_model.learn_one(x, y)
        self._save_model()

    def update_linucb_model(self, student_features: Dict[str, Any], question_features: Dict[str, Any], question_id: int, reward: float):
        self.linucb_bandit.update(question_id, student_features, question_features, reward)
        self._save_model()

    async def get_recommendations(self, db: Session, student_id: int, n_recommendations: int = 10, request_id: str = None) -> List[Dict[str, Any]]:
        """Öğrenci için soru önerileri üret (Ensemble yaklaşım)"""
        cache_key = f"recommendations:{student_id}:{self.model_type}:ensemble"
        cached = self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached.decode('utf-8'))
        
        student_features = self.get_student_features(db, student_id)
        student_level = student_features.get('level', 1)
        student_recent_performance = student_features.get('last20_accuracy', 0.5)
        
        # Öğrencinin son sorularını al
        recent_responses = db.query(StudentResponse).filter(
            StudentResponse.student_id == student_id
        ).order_by(StudentResponse.created_at.desc()).limit(10).all()
        
        student_recent_questions = []
        student_recent_question_ids = []
        for response in recent_responses:
            question = db.query(Question).filter(Question.id == response.question_id).first()
            if question:
                student_recent_questions.append(question.content)
                student_recent_question_ids.append(question.id)
        
        # Ağırlıkları dinamik olarak ayarla
        adjust_weights_dynamically(student_recent_performance, len(recent_responses))
        
        questions = db.query(Question).filter(Question.is_active == True).all()
        recommendations = []
        
        for question in questions:
            # Daha önce cevaplanmış soruları atla
            answered = db.query(StudentResponse).filter(
                StudentResponse.student_id == student_id,
                StudentResponse.question_id == question.id
            ).first()
            if answered:
                continue

            # Runtime: Dinamik zorluk ayarı
            adjusted_difficulty = await self._adjust_question_difficulty_runtime(
                question, student_level, student_features
            )

            question_features = self.get_question_features(db, question.id)
            question_features['adjusted_difficulty'] = adjusted_difficulty
            combined_features = {**student_features, **question_features}

            # River model skoru
            river_score = self.river_model.predict_proba_one(combined_features).get(1, 0.5)

            # Ensemble skor hesapla
            ensemble_scores = calculate_ensemble_score(
                river_score=river_score,
                question_content=question.content,
                question_id=question.id,
                question_difficulty=question.difficulty_level,
                student_id=student_id,
                student_level=student_level,
                student_recent_performance=student_recent_performance,
                student_recent_questions=student_recent_questions,
                student_recent_question_ids=student_recent_question_ids
            )

            recommendation = {
                'question_id': question.id,
                'ensemble_score': ensemble_scores['ensemble_score'],
                'river_score': ensemble_scores['river_score'],
                'embedding_similarity': ensemble_scores['embedding_similarity'],
                'skill_mastery': ensemble_scores['skill_mastery'],
                'difficulty_match': ensemble_scores['difficulty_match'],
                'neo4j_similarity': ensemble_scores['neo4j_similarity'],
                'adjusted_difficulty': adjusted_difficulty,
                'original_difficulty': question.difficulty_level,
                'question': {
                    "id": question.id,
                    "content": question.content,
                    "question_type": question.question_type,
                    "difficulty_level": question.difficulty_level,
                    "subject_id": question.subject_id,
                    "options": question.options,
                    "correct_answer": question.correct_answer,
                    "explanation": question.explanation,
                    "tags": question.tags,
                    "created_by": question.created_by,
                    "is_active": question.is_active,
                    "created_at": str(question.created_at),
                    "updated_at": str(question.updated_at)
                }
            }
            
            recommendations.append(recommendation)

        # Threshold'lara göre filtrele
        recommendations = filter_questions_by_thresholds(
            recommendations, student_level, student_recent_performance
        )
        
        # Ensemble skora göre sırala
        recommendations.sort(key=lambda x: x['ensemble_score'], reverse=True)
        top_recommendations = recommendations[:n_recommendations]

        # Cache'e kaydet
        self.redis_client.setex(
            cache_key,
            300,  # 5 dakika
            json.dumps(top_recommendations, default=str)
        )
        
        logger = get_logger_with_context(request_id=request_id, student_id=student_id)
        logger.info("ensemble_recommendations_generated", 
                   recommendations=[r['question_id'] for r in top_recommendations],
                   ensemble_scores=[r['ensemble_score'] for r in top_recommendations])
        
        return top_recommendations

    def process_student_response(self, db: Session, student_id: int, question_id: int,
                               answer: str, is_correct: bool, response_time: float, feedback: str = None, request_id: str = None):
        subject = None
        try:
            question = db.query(Question).filter(Question.id == question_id).first()
            subject = getattr(question, 'subject_id', 'unknown')
        except Exception:
            subject = 'unknown'
        env = 'prod'  # veya settings.ENVIRONMENT
        if getattr(settings, 'USE_PROMETHEUS_HISTOGRAM', True):
            with model_update_duration.labels(subject=str(subject), environment=env).time():
                self._process_student_response_inner(db, student_id, question_id, answer, is_correct, response_time, feedback, request_id)
        else:
            self._process_student_response_inner(db, student_id, question_id, answer, is_correct, response_time, feedback, request_id)

    def _process_student_response_inner(self, db, student_id, question_id, answer, is_correct, response_time, feedback, request_id):
        question = db.query(Question).filter(Question.id == question_id).first()
        if not question: return

        student_features = self.get_student_features(db, student_id)
        question_features = self.get_question_features(db, question_id)
        if self.model_type == 'linucb':
            self.update_linucb_model(student_features, question_features, question_id, 1 if is_correct else 0)
        else:
            self.update_river_model(student_features, question_features, is_correct, response_time)

        # Feedback entegrasyonu örneği (dummy):
        if feedback:
            # Burada feedback'i loglayabilir veya modele ekleyebilirsiniz
            print(f"Feedback for student {student_id}, question {question_id}: {feedback}")

        try:
            update_student_level(db, student_id, question.difficulty_level, is_correct)
        except Exception as e:
            print(f"Seviye güncellenirken hata: {e}")

        # Öğrenci özellikleri değiştiği için cache'i temizle
        self.redis_client.delete(f"recommendations:{student_id}")
        model_update_counter.inc()
        logger = get_logger_with_context(request_id=request_id, student_id=student_id, question_id=question_id)
        logger.info("student_response_processed", is_correct=is_correct, response_time=response_time, feedback=feedback)

    async def _adjust_question_difficulty_runtime(self, question, student_level: int, student_features: Dict[str, Any]) -> int:
        """Runtime'da soru zorluğunu ayarla"""
        try:
            from app.services.llm_service import llm_service
            
            # Öğrenci performans verilerini hazırla
            performance_data = {
                'recent_accuracy': student_features.get('last20_accuracy', 0.5),
                'avg_response_time': student_features.get('avg_response_time', 0),
                'weak_topics': student_features.get('weak_topics', []),
                'strong_topics': student_features.get('strong_topics', [])
            }
            
            # LLM ile dinamik zorluk ayarı
            adjustment = await llm_service.adjust_difficulty_runtime(
                question.content,
                student_level,
                question.difficulty_level,
                performance_data
            )
            
            return adjustment.get('adjusted_difficulty', question.difficulty_level)
            
        except Exception as e:
            # Fallback: Basit mantık
            accuracy = student_features.get('last20_accuracy', 0.5)
            if accuracy > 0.8 and question.difficulty_level < 5:
                return question.difficulty_level + 1
            elif accuracy < 0.4 and question.difficulty_level > 1:
                return question.difficulty_level - 1
            else:
                return question.difficulty_level
    
    async def get_adaptive_hint(self, question_id: int, student_id: int, db: Session) -> str:
        """Öğrenci durumuna göre adaptif ipucu al"""
        try:
            from app.services.llm_service import llm_service
            
            # Öğrenci ve soru bilgilerini al
            question = db.query(Question).filter(Question.id == question_id).first()
            student_features = self.get_student_features(db, student_id)
            
            if not question:
                return "Soru bulunamadı."
            
            # Öğrenci durumunu analiz et
            student_level = student_features.get('level', 1)
            recent_accuracy = student_features.get('last20_accuracy', 0.5)
            avg_response_time = student_features.get('avg_response_time', 0)
            
            # Öğrenci zorlanıyor mu?
            student_struggling = recent_accuracy < 0.4 or avg_response_time > 60000  # 60 saniye
            
            # Adaptif ipucu üret
            adaptive_hint = await llm_service.generate_adaptive_hint(
                question.content,
                student_level,
                student_features.get('hints_used', 0),
                student_struggling
            )
            
            return adaptive_hint
            
        except Exception as e:
            # Fallback: Veritabanındaki hazır ipucu
            question = db.query(Question).filter(Question.id == question_id).first()
            return question.hint if question and question.hint else "İpucu bulunamadı."
    
    async def get_contextual_explanation(self, question_id: int, student_id: int, 
                                       student_answer: str, db: Session) -> str:
        """Öğrencinin cevabına göre bağlamsal açıklama al"""
        try:
            from app.services.llm_service import llm_service
            
            # Öğrenci ve soru bilgilerini al
            question = db.query(Question).filter(Question.id == question_id).first()
            student_features = self.get_student_features(db, student_id)
            
            if not question:
                return "Soru bulunamadı."
            
            # Bağlamsal açıklama üret
            contextual_explanation = await llm_service.generate_contextual_explanation(
                question.content,
                question.correct_answer,
                student_answer,
                student_features.get('level', 1)
            )
            
            return contextual_explanation
            
        except Exception as e:
            # Fallback: Veritabanındaki hazır açıklama
            question = db.query(Question).filter(Question.id == question_id).first()
            return question.explanation if question and question.explanation else "Açıklama bulunamadı."

# Global instance
recommendation_service = RecommendationService() 