import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import redis
import structlog
from app.core.config import settings
from app.db.models import Question, QuestionSkill, StudentResponse
# from app.ml.bandits import LinUCBBandit  # Removed - ml module deleted
# from app.services.ensemble_service import (
#     adjust_weights_dynamically,
#     calculate_enhanced_ensemble_score,
#     filter_questions_by_thresholds,
# )  # Removed - ensemble_service deleted
# from app.services.level_service import update_student_level  # Removed - level_service deleted
from app.services.enhanced_embedding_service import enhanced_embedding_service
from app.services.real_models import cf_model, bandit_model, online_model
from app.services.advanced_models import (
    create_advanced_models, 
    assign_variant, 
    mix_advanced_scores,
    AdvancedCFModel,
    AdvancedBanditModel, 
    AdvancedOnlineModel,
    QuestionReranker,
    QuestionDTO
)
from prometheus_client import Counter, Histogram
from river import compose, linear_model, preprocessing
from sqlalchemy.orm import Session
import threading

# Global logger instance
logger = structlog.get_logger()

# Prometheus metrics
model_update_counter = Counter("model_update_total", "Model güncelleme sayısı")
model_update_duration = Histogram(
    "model_update_duration_seconds", "Model update süresi", ["subject", "environment"]
)

# Lazy import to avoid heavy transformers loading at startup
# from sentence_transformers import SentenceTransformer

# Modeli thread-safe şekilde cache'le
_model = None
_model_lock = threading.Lock()


def get_sbert_model():
    """Lazy load SentenceTransformer to avoid startup delays"""
    global _model
    with _model_lock:
        if _model is None:
            try:
                # Import here to avoid startup delay
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
            except ImportError:
                # Fallback for when transformers are not available
                return None
        return _model


# --- River pipeline ---
def build_river_model():
    return compose.Pipeline(
        # HATA DÜZELTİLDİ: OneHotEncoder 'preprocessing' modülünden çağrılıyor
        preprocessing.OneHotEncoder(),
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(),
    )


def get_skill_centrality(student_id: int) -> Dict[int, int]:
    # from app.services.neo4j_service import neo4j_service  # Removed - using async_neo4j_service instead

    # Using async_neo4j_service instead
    try:
        # This would need to be converted to async call
        logger.warning("skill_centrality_not_implemented", student_id=student_id)
        return {}
    except Exception as e:
        logger.error("get_skill_centrality_error", student_id=student_id, error=str(e))
        return {}


def diversity_filter(
    candidates: List[tuple], question_embeddings: Dict[int, List[float]], top_n: int = 5
) -> List[tuple]:
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
            sim = np.dot(ref_emb, emb) / (
                np.linalg.norm(ref_emb) * np.linalg.norm(emb) + 1e-8
            )
        else:
            sim = 0
        sims.append((qid, sim))
    # En benzer top_n soruyu bul
    sims = sorted(sims, key=lambda x: -x[1])
    filtered = [c for c in candidates if c[0] not in [qid for qid, _ in sims[:top_n]]]
    return filtered


def add_question_to_neo4j(question_id: Any, skill_ids: List[int]) -> None:
    # from app.services.neo4j_service import neo4j_service  # Removed - using async_neo4j_service instead

    # Using async_neo4j_service instead
    logger.warning("add_question_to_neo4j_not_implemented", question_id=question_id)


# Embedding hesaplama artık embedding_service'den geliyor


def get_logger_with_context(**context: Any) -> structlog.BoundLogger:
    return structlog.get_logger().bind(**context)


class RecommendationService:
    def __init__(self, model_type: str = "river"):
        self.redis_client = redis.Redis.from_url(settings.redis_url)
        self.model_type = model_type
        self.river_model = build_river_model()
        # self.linucb_bandit = LinUCBBandit(self.redis_client, alpha=1.0, feature_dim=10)  # Removed - ml module deleted
        self.model_cache_dir = settings.MODEL_CACHE_DIR
        os.makedirs(self.model_cache_dir, exist_ok=True)
        self._load_model()
        
        # Gerçek modeller için initialization
        self.cf_model = cf_model
        self.bandit_model = bandit_model
        self.online_model = online_model
        
        # Advanced models (optional)
        try:
            if self.redis_client is not None:
                self.advanced_models = create_advanced_models(self.redis_client)
                self.use_advanced_models = getattr(settings, "USE_ADVANCED_MODELS", False)
                logger.info("advanced_models_initialized", use_advanced=self.use_advanced_models)
            else:
                self.advanced_models = None
                self.use_advanced_models = False
                logger.warning("redis_client_not_available_for_advanced_models")
        except Exception as e:
            logger.warning("advanced_models_not_available", error=str(e))
            self.advanced_models = None
            self.use_advanced_models = False

    def _load_model(self):
        river_path = os.path.join(self.model_cache_dir, "river_model.bin")
        linucb_path = os.path.join(self.model_cache_dir, "linucb_model.npz")
        if os.path.exists(river_path):
            with open(river_path, "rb") as f:
                try:
                    # Try to use from_bytes if available
                    model = build_river_model()
                    if hasattr(model, "from_bytes"):
                        # River Pipeline may not have from_bytes in all versions
                        try:
                            self.river_model = model.from_bytes(f.read())  # type: ignore
                        except (AttributeError, TypeError):
                            logger.warning("river_model_from_bytes_not_available")
                            self.river_model = build_river_model()
                    else:
                        logger.warning("river_model_from_bytes_not_available")
                        self.river_model = build_river_model()
                except Exception as e:
                    logger.warning("river_model_load_error", error=str(e))
                    self.river_model = build_river_model()
        if os.path.exists(linucb_path):
            # Try to load as JSON first (new format)
            json_path = linucb_path.replace(".npz", ".json")
            if os.path.exists(json_path):
                import json

                with open(json_path, "r") as f:
                    data = json.load(f)
                            # self.linucb_bandit.A = data.get("A", {})  # Removed - ml module deleted
            # self.linucb_bandit.b = data.get("b", {})  # Removed - ml module deleted
            else:
                # Fallback to old npz format
                data = np.load(linucb_path, allow_pickle=True)
                A_data = data["A"].item()
                b_data = data["b"].item()
                # LinUCBBandit uses dictionaries for A and b  # Removed - ml module deleted
                if isinstance(A_data, dict):
                    self.linucb_bandit.A = A_data
                else:
                    self.linucb_bandit.A = {}
                if isinstance(b_data, dict):
                    self.linucb_bandit.b = b_data
                else:
                    self.linucb_bandit.b = {}

    def _save_model(self):
        river_path = os.path.join(self.model_cache_dir, "river_model.bin")
        linucb_path = os.path.join(self.model_cache_dir, "linucb_model.npz")
        try:
            with open(river_path, "wb") as f:
                if hasattr(self.river_model, "to_bytes"):
                    # River Pipeline may not have to_bytes in all versions
                    try:
                        f.write(self.river_model.to_bytes())  # type: ignore
                    except (AttributeError, TypeError):
                        logger.warning("river_model_to_bytes_not_available")
                else:
                    logger.warning("river_model_to_bytes_not_available")
        except Exception as e:
            logger.warning("river_model_save_error", error=str(e))
        # Save LinUCB model state as JSON since it uses dictionaries
        import json

        linucb_state = {"A": self.linucb_bandit.A, "b": self.linucb_bandit.b}
        with open(linucb_path.replace(".npz", ".json"), "w") as f:
            json.dump(linucb_state, f, default=str)

    def get_student_features(self, db: Session, student_id: int) -> Dict[str, Any]:
        """
        Öğrenci özelliklerini hesapla ve döndür.
        
        Returns:
            Dict[str, Any]: Öğrenci özellikleri
                - level: Öğrenci seviyesi (1-5)
                - total_questions: Toplam çözülen soru sayısı
                - correct_answers: Doğru cevap sayısı
                - avg_response_time: Ortalama cevap süresi (ms)
                - last20_accuracy: Son 20 sorudaki doğruluk oranı
                - strong_topics: Güçlü olduğu konular listesi
                - weak_topics: Zayıf olduğu konular listesi
                - avg_confidence: Ortalama güven skoru
                - recent_accuracy: Son dönem doğruluk oranı
                - difficulty_preference: Tercih ettiği zorluk seviyesi
                - speed_factor: Hız faktörü
                - consistency_score: Tutarlılık skoru
                - engagement_level: Katılım seviyesi
                - learning_style: Öğrenme stili
                - motivation_score: Motivasyon skoru
        """
        # HATA DÜZELTİLDİ: 'timestamp' yerine 'created_at' kullanılıyor
        responses = (
            db.query(StudentResponse)
            .filter(getattr(StudentResponse, "student_id") == student_id)
            .order_by(getattr(StudentResponse, "created_at").desc())
            .all()
        )
        features = {
            "total_questions": len(responses),
            "correct_answers": sum(
                1 for r in responses if getattr(r, "is_correct", False)
            ),
            "avg_response_time": np.mean(
                [
                    getattr(r, "response_time", 0)
                    for r in responses
                    if getattr(r, "response_time", 0)
                ]
            )
            if responses
            else 0,
            "avg_confidence": np.mean(
                [
                    getattr(r, "confidence_level", 0)
                    for r in responses
                    if getattr(r, "confidence_level", 0)
                ]
            )
            if responses
            else 0,
            "hour_of_day": getattr(responses[0], "created_at").hour if responses else 0,
        }
        # Son 20 soruda başarı oranı (sliding window)
        last_n = 20
        last_n_responses = responses[:last_n]
        features["last20_accuracy"] = (
            np.mean([getattr(r, "is_correct", False) for r in last_n_responses])
            if last_n_responses
            else 0
        )
        # Konu başına doğruluk oranı
        topic_correct = {}
        topic_total = {}
        for r in responses:
            q = (
                db.query(Question)
                .filter(getattr(Question, "id") == getattr(r, "question_id", 0))
                .first()
            )
            if q:
                topic = getattr(q, "subject_id", 0)
                topic_total[topic] = topic_total.get(topic, 0) + 1
                if getattr(r, "is_correct", False):
                    topic_correct[topic] = topic_correct.get(topic, 0) + 1
        for topic in topic_total:
            features[f"topic_{topic}_accuracy"] = (
                topic_correct.get(topic, 0) / topic_total[topic]
            )
        # Skill mastery
        skill_mastery = {}
        for response in responses:
            question_skills = (
                db.query(QuestionSkill)
                .filter(
                    getattr(QuestionSkill, "question_id")
                    == getattr(response, "question_id", 0)
                )
                .all()
            )
            for qs in question_skills:
                skill_id = getattr(qs, "skill_id", 0)
                if skill_id not in skill_mastery:
                    skill_mastery[skill_id] = {"correct": 0, "total": 0}
                skill_mastery[skill_id]["total"] += 1
                if getattr(response, "is_correct", False):
                    skill_mastery[skill_id]["correct"] += 1
        for skill_id, mastery in skill_mastery.items():
            features[f"skill_{skill_id}_mastery"] = (
                mastery["correct"] / mastery["total"] if mastery["total"] > 0 else 0
            )
        # --- Neo4j skill centrality feature ---
        try:
            centrality = get_skill_centrality(student_id)
            for skill_id, freq in centrality.items():
                features[f"skill_{skill_id}_centrality"] = freq
        except Exception:
            pass
        return features

    def get_question_features(self, db: Session, question_id: Any) -> Dict[str, Any]:
        """
        Soru özelliklerini hesapla ve döndür.
        
        Args:
            db: Database session
            question_id: Soru ID'si
            
        Returns:
            Dict[str, Any]: Soru özellikleri
                - difficulty_level: Zorluk seviyesi (1-5)
                - subject_id: Konu ID'si
                - topic_id: Alt konu ID'si
                - question_type: Soru tipi (1-5)
                - complexity_score: Karmaşıklık skoru (0-1)
                - prerequisite_count: Ön koşul sayısı
                - estimated_time: Tahmini çözüm süresi (saniye)
                - skill_requirements: Gerekli beceriler listesi
                - popularity_score: Popülerlik skoru (0-1)
                - quality_score: Kalite skoru (0-1)
                - embedding_vector: Embedding vektörü (varsa)
                - skill_count: Beceri sayısı
                - question_length: Soru uzunluğu (kelime sayısı)
                - avg_skill_difficulty: Ortalama beceri zorluğu
                - num_tags: Etiket sayısı
        """
        question = (
            db.query(Question).filter(getattr(Question, "id") == question_id).first()
        )
        if not question:
            return {}
        question_skills = (
            db.query(QuestionSkill)
            .filter(getattr(QuestionSkill, "question_id") == question_id)
            .all()
        )
        features = {
            "difficulty_level": getattr(question, "difficulty_level", 1),
            "question_type": getattr(question, "question_type", "multiple_choice"),
            "subject_id": getattr(question, "subject_id", 0),
            "skill_count": len(question_skills),
            # HATA DÜZELTİLDİ: 'text' yerine 'content' kullanılıyor
            "question_length": len(getattr(question, "content", "").split())
            if hasattr(question, "content")
            else 0,
            "avg_skill_difficulty": np.mean(
                [getattr(qs.skill, "difficulty_level", 1) for qs in question_skills]
            )
            if question_skills
            else 0,
            "num_tags": len(getattr(question, "tags", []))
            if hasattr(question, "tags")
            else 0,
        }
        for qs in question_skills:
            features[f"skill_{getattr(qs, 'skill_id', 0)}_weight"] = getattr(
                qs, "weight", 1.0
            )
        # Gömme benzerliği (varsa)
        if hasattr(question, "bert_sim"):
            features["bert_sim"] = question.bert_sim
        return features

    def update_river_model(
        self,
        student_features: Dict[str, Any],
        question_features: Dict[str, Any],
        is_correct: bool,
        response_time: float,
    ):
        combined_features = {**student_features, **question_features}
        x = combined_features
        y = 1 if is_correct else 0
        self.river_model.learn_one(x, y)
        self._save_model()

    def update_linucb_model(
        self,
        student_features: Dict[str, Any],
        question_features: Dict[str, Any],
        question_id: int,
        reward: float,
    ):
        self.linucb_bandit.update(
            question_id, student_features, question_features, reward
        )
        self._save_model()

    async def recommend_with_embedding(
        self, 
        db: Session,
        student_id: int, 
        past_questions: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Past_questions embed vector'larını kullanarak Faiss index ile öneri üret.
        """
        try:
            if not past_questions or not any("embedding" in q for q in past_questions):
                logger.warning("no_embedding_data_available", student_id=student_id)
                return []
            
            # Ortalama embedding hesapla
            embeddings = [q["embedding"] for q in past_questions if "embedding" in q]
            if not embeddings:
                logger.warning("no_valid_embeddings", student_id=student_id)
                return []
                
            avg_embedding = np.mean(embeddings, axis=0).tolist()
            
            # Enhanced embedding service ile benzer soruları bul
            similar_questions = await enhanced_embedding_service.semantic_search_vector_db(
                query_text="",  # We'll use the embedding directly
                k=top_k,
                similarity_threshold=0.7,
                filters=None
            )
            
            # Eğer vector store kullanılamıyorsa, fallback olarak semantic search kullan
            if not similar_questions:
                # Fallback: Question pool'dan semantic search
                from app.crud.question import get_questions
                question_pool = get_questions(db=db, limit=100)  # Get recent questions
                
                # Convert Question objects to dict format
                question_dicts = []
                for question in question_pool:
                    question_dicts.append({
                        "id": question.id,
                        "content": question.content,
                        "difficulty_level": question.difficulty_level,
                        "subject_id": question.subject_id,
                        "topic_id": question.topic_id
                    })
                
                similar_questions = await enhanced_embedding_service.semantic_search(
                    query="",  # We'll use embedding similarity
                    question_pool=question_dicts,
                    top_k=top_k,
                    similarity_threshold=0.6
                )
            
            # Sonuçları formatla
            formatted_results = []
            for q in similar_questions:
                result = {
                    "question_id": q.get("id", 0),
                    "method": "embedding",
                    "similarity_score": q.get("semantic_similarity", 0.0),
                    "content": q.get("content", ""),
                    "difficulty_level": q.get("difficulty_level", 1),
                    "subject_id": q.get("subject_id", None),
                    "topic_id": q.get("topic_id", None)
                }
                formatted_results.append(result)
            
            # Metrik güncelle
            model_update_counter.labels(model_type="embedding", subject="all", environment=settings.ENVIRONMENT).inc()
            
            logger.info(
                "embedding_recommendation_generated", 
                student_id=student_id, 
                count=len(formatted_results),
                avg_similarity=np.mean([r["similarity_score"] for r in formatted_results]) if formatted_results else 0.0
            )
            
            return formatted_results
            
        except Exception as e:
            logger.error("embedding_recommendation_error", student_id=student_id, error=str(e))
            return []

    async def get_cf_recommendations(
        self, 
        db: Session, 
        student_id: int, 
        top_k: int = 50
    ) -> List[QuestionDTO]:
        """Collaborative Filtering önerileri"""
        try:
            recommendations = await self.cf_model.score(student_id, db, top_k)
            logger.info("cf_recommendations_generated", student_id=student_id, count=len(recommendations))
            return recommendations
        except Exception as e:
            logger.error("cf_recommendations_error", student_id=student_id, error=str(e))
            return []

    async def get_bandit_recommendations(
        self,
        db: Session,
        student_id: int,
        user_features: Dict[str, Any],
        top_k: int = 50
    ) -> List[Dict[str, Any]]:
        """Bandit model önerileri"""
        try:
            # Tüm soruları al
            questions = db.query(Question).filter(Question.is_active == True).all()
            
            # Her soru için bandit skoru hesapla
            scored_questions = []
            for question in questions:
                # Question ID'yi güvenli şekilde dönüştür
                qid = int(getattr(question, "id", 0))
                question_features = self.get_question_features(db, qid)
                bandit_score = self.bandit_model.predict(user_features, question_features, qid)
                
                scored_questions.append({
                    "question_id": question.id,
                    "bandit_score": bandit_score,
                    "content": question.content,
                    "difficulty_level": question.difficulty_level,
                    "subject_id": question.subject_id,
                    "topic_id": question.topic_id
                })
            
            # Skora göre sırala ve top_k al
            scored_questions.sort(key=lambda x: x["bandit_score"], reverse=True)
            recommendations = scored_questions[:top_k]
            
            logger.info("bandit_recommendations_generated", student_id=student_id, count=len(recommendations))
            return recommendations
            
        except Exception as e:
            logger.error("bandit_recommendations_error", student_id=student_id, error=str(e))
            return []

    async def get_online_recommendations(
        self,
        db: Session,
        student_id: int,
        user_features: Dict[str, Any],
        top_k: int = 50
    ) -> List[Dict[str, Any]]:
        """Online learning model önerileri"""
        try:
            # Tüm soruları al
            questions = db.query(Question).filter(Question.is_active == True).all()
            
            # Her soru için online model skoru hesapla
            scored_questions = []
            for question in questions:
                # Question ID'yi güvenli şekilde dönüştür
                qid = int(getattr(question, "id", 0))
                question_features = self.get_question_features(db, qid)
                online_score = self.online_model.predict(student_id, user_features, question_features)
                
                scored_questions.append({
                    "question_id": question.id,
                    "online_score": online_score,
                    "content": question.content,
                    "difficulty_level": question.difficulty_level,
                    "subject_id": question.subject_id,
                    "topic_id": question.topic_id
                })
            
            # Skora göre sırala ve top_k al
            scored_questions.sort(key=lambda x: x["online_score"], reverse=True)
            recommendations = scored_questions[:top_k]
            
            logger.info("online_recommendations_generated", student_id=student_id, count=len(recommendations))
            return recommendations
            
        except Exception as e:
            logger.error("online_recommendations_error", student_id=student_id, error=str(e))
            return []

    async def get_recommendations(
        self,
        db: Session,
        student_id: int,
        n_recommendations: int = 10,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Öğrenci için soru önerileri üret (Ensemble yaklaşım)"""
        cache_key = f"recommendations:{student_id}:{self.model_type}:ensemble"
        if self.redis_client is not None:
            cached = self.redis_client.get(cache_key)
            if cached is not None:
                return json.loads(cached.decode("utf-8"))

        student_features = self.get_student_features(db, student_id)
        student_level = student_features.get("level", 1)
        student_recent_performance = student_features.get("last20_accuracy", 0.5)

        # Öğrencinin son sorularını al
        recent_responses = (
            db.query(StudentResponse)
            .filter(StudentResponse.student_id == student_id)
            .order_by(StudentResponse.created_at.desc())
            .limit(10)
            .all()
        )

        student_recent_questions = []
        student_recent_question_ids = []
        for response in recent_responses:
            question = (
                db.query(Question).filter(Question.id == response.question_id).first()
            )
            if question:
                student_recent_questions.append(question.content)
                student_recent_question_ids.append(question.id)

        # Ağırlıkları dinamik olarak ayarla
        adjust_weights_dynamically(student_recent_performance, len(recent_responses))

        # Model stratejilerine göre öneri üret
        if self.model_type == "cf":
            # Collaborative Filtering
            cf_recommendations = await self.get_cf_recommendations(db, student_id, n_recommendations)
            
            # Cache'e kaydet
            if self.redis_client is not None:
                cache_data = json.dumps([{"question_id": q.id, "cf_score": q.features.get("cf_similarity", 0.0) if q.features else 0.0} for q in cf_recommendations])
                self.redis_client.setex(cache_key, 300, cache_data)
            
            return [{"question_id": q.id, "method": "cf", "score": q.features.get("cf_similarity", 0.0) if q.features else 0.0} for q in cf_recommendations]
            
        elif self.model_type == "bandit":
            # Contextual Bandit
            bandit_recommendations = await self.get_bandit_recommendations(db, student_id, student_features, n_recommendations)
            
            # Cache'e kaydet
            if self.redis_client is not None:
                cache_data = json.dumps(bandit_recommendations)
                self.redis_client.setex(cache_key, 300, cache_data)
            
            return bandit_recommendations
            
        elif self.model_type == "online":
            # Online Learning
            online_recommendations = await self.get_online_recommendations(db, student_id, student_features, n_recommendations)
            
            # Cache'e kaydet
            if self.redis_client is not None:
                cache_data = json.dumps(online_recommendations)
                self.redis_client.setex(cache_key, 300, cache_data)
            
            return online_recommendations
            
        elif self.model_type == "advanced":
            # Advanced retriever/reranker pipeline
            return await self._get_advanced_recommendations(
                db=db,
                student_id=student_id,
                student_features=student_features,
                n_recommendations=n_recommendations
            )
            
        elif self.model_type == "embedding":
            # Embedding tabanlı strateji
            student_recent_questions_with_embedding = []
            for response in recent_responses:
                question = (
                    db.query(Question).filter(Question.id == response.question_id).first()
                )
                if question and hasattr(question, 'embedding_vector') and getattr(question, 'embedding_vector', None):
                    student_recent_questions_with_embedding.append({
                        "id": question.id,
                        "content": question.content,
                        "embedding": question.embedding_vector
                    })
            
            embedding_recommendations = await self.recommend_with_embedding(
                db=db,
                student_id=student_id,
                past_questions=student_recent_questions_with_embedding,
                top_k=n_recommendations
            )
            
            # Cache'e kaydet
            if self.redis_client is not None:
                cache_data = json.dumps(embedding_recommendations)
                self.redis_client.setex(cache_key, 300, cache_data)
            
            return embedding_recommendations
            
        elif self.model_type == "advanced":
            # Advanced retriever/reranker pipeline
            return await self._get_advanced_recommendations(
                db=db,
                student_id=student_id,
                student_features=student_features,
                n_recommendations=n_recommendations
            )

        # Ensemble strateji için embedding önerilerini hazırla
        embedding_recommendations = []
        if self.model_type == "ensemble":
            # Öğrencinin geçmiş sorularını embedding formatında hazırla
            student_recent_questions_with_embedding = []
            for response in recent_responses:
                question = (
                    db.query(Question).filter(Question.id == response.question_id).first()
                )
                if question and hasattr(question, 'embedding_vector') and getattr(question, 'embedding_vector', None):
                    student_recent_questions_with_embedding.append({
                        "id": question.id,
                        "content": question.content,
                        "embedding": question.embedding_vector
                    })
            
            # Embedding tabanlı öneri üret (ensemble için daha fazla öneri)
            embedding_recommendations = await self.recommend_with_embedding(
                db=db,
                student_id=student_id,
                past_questions=student_recent_questions_with_embedding,
                top_k=n_recommendations * 2  # Ensemble için daha fazla öneri
            )

        questions = db.query(Question).filter(Question.is_active == True).all()
        recommendations = []

        for question in questions:
            # Daha önce cevaplanmış soruları atla
            answered = (
                db.query(StudentResponse)
                .filter(
                    StudentResponse.student_id == student_id,
                    StudentResponse.question_id == question.id,
                )
                .first()
            )
            if answered:
                continue

            # Runtime: Dinamik zorluk ayarı
            adjusted_difficulty = await self._adjust_question_difficulty_runtime(
                question, student_level, student_features
            )

            question_features = self.get_question_features(db, question.id)
            question_features["adjusted_difficulty"] = adjusted_difficulty
            combined_features = {**student_features, **question_features}

            # River model skoru
            river_score = self.river_model.predict_proba_one(combined_features).get(
                1, 0.5
            )

            # Ensemble skor hesapla (async enhanced embedding similarity ile)
            ensemble_scores = await calculate_enhanced_ensemble_score(
                river_score=river_score,
                question_content=str(question.content),
                question_id=question.id,
                question_difficulty=question.difficulty_level,
                student_id=student_id,
                student_level=student_level,
                student_recent_performance=student_recent_performance,
                student_recent_questions=student_recent_questions,
                student_recent_question_ids=student_recent_question_ids,
            )

            # Embedding similarity skorunu hesapla (ensemble için)
            embedding_similarity_score = 0.0
            if embedding_recommendations:
                # Bu soru embedding önerilerinde var mı kontrol et
                for emb_rec in embedding_recommendations:
                    if emb_rec.get("question_id") == question.id:
                        embedding_similarity_score = emb_rec.get("similarity_score", 0.0)
                        break

            recommendation = {
                "question_id": question.id,
                "ensemble_score": ensemble_scores["ensemble_score"],
                "river_score": ensemble_scores["river_score"],
                "embedding_similarity": ensemble_scores["embedding_similarity"],
                "embedding_similarity_boost": embedding_similarity_score,  # Yeni embedding skoru
                "skill_mastery": ensemble_scores["skill_mastery"],
                "difficulty_match": ensemble_scores["difficulty_match"],
                "neo4j_similarity": ensemble_scores["neo4j_similarity"],
                "adjusted_difficulty": adjusted_difficulty,
                "original_difficulty": question.difficulty_level,
                "question": {
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
                    "updated_at": str(question.updated_at),
                },
            }

            recommendations.append(recommendation)

        # Threshold'lara göre filtrele
        recommendations = filter_questions_by_thresholds(
            recommendations, student_level, student_recent_performance
        )

        # Ensemble skora göre sırala
        recommendations.sort(key=lambda x: x["ensemble_score"], reverse=True)
        top_recommendations = recommendations[:n_recommendations]

        # Cache'e kaydet
        if self.redis_client is not None:
            self.redis_client.setex(
                cache_key,
                300,
                json.dumps(top_recommendations, default=str),  # 5 dakika
            )

        # Metrik güncelle (embedding dahil)
        model_update_counter.labels(
            model_type=self.model_type, 
            subject="all", 
            environment=settings.ENVIRONMENT
        ).inc()

        logger = get_logger_with_context(request_id=request_id, student_id=student_id)
        logger.info(
            "ensemble_recommendations_generated",
            recommendations=[r["question_id"] for r in top_recommendations],
            ensemble_scores=[r["ensemble_score"] for r in top_recommendations],
            model_type=self.model_type,
            embedding_recommendations_count=len(embedding_recommendations) if 'embedding_recommendations' in locals() else 0,
        )

        return top_recommendations

    async def _get_advanced_recommendations(
        self,
        db: Session,
        student_id: int,
        student_features: Dict[str, Any],
        n_recommendations: int = 10
    ) -> List[Dict[str, Any]]:
        """Advanced retriever/reranker pipeline"""
        try:
            if not self.use_advanced_models or not self.advanced_models:
                logger.warning("advanced_models_not_available_fallback")
                return []
            
            # A/B test variant assignment
            variant = assign_variant(str(student_id))
            logger.info("advanced_variant_assigned", student_id=student_id, variant=variant)
            
            # Get candidates from different retrievers
            cf_candidates = []
            bandit_candidates = []
            online_candidates = []
            
            if variant in ["cf-only", "hybrid"]:
                cf_candidates = await self.advanced_models["cf"].score(
                    str(student_id), db, top_k=50
                )
            
            if variant in ["bandit-only", "hybrid"]:
                # Get questions for bandit scoring
                questions = db.query(Question).filter(Question.is_active == True).limit(100).all()
                for question in questions:
                    question_features = self.get_question_features(db, question.id)
                    score = await self.advanced_models["bandit"].predict(
                        student_features, question_features, question.id
                    )
                    bandit_candidates.append({
                        "question_id": question.id,
                        "content": question.content,
                        "difficulty_level": question.difficulty_level,
                        "subject_id": question.subject_id,
                        "topic_id": question.topic_id,
                        "score": score,
                        "method": "advanced_bandit"
                    })
                bandit_candidates.sort(key=lambda x: x["score"], reverse=True)
                bandit_candidates = bandit_candidates[:50]
            
            if variant in ["online-only", "hybrid"]:
                # Get questions for online scoring
                questions = db.query(Question).filter(Question.is_active == True).limit(100).all()
                for question in questions:
                    question_features = self.get_question_features(db, question.id)
                    score = await self.advanced_models["online"].predict(
                        student_id, student_features, question_features
                    )
                    online_candidates.append({
                        "question_id": question.id,
                        "content": question.content,
                        "difficulty_level": question.difficulty_level,
                        "subject_id": question.subject_id,
                        "topic_id": question.topic_id,
                        "score": score,
                        "method": "advanced_online"
                    })
                online_candidates.sort(key=lambda x: x["score"], reverse=True)
                online_candidates = online_candidates[:50]
            
            # Mix candidates based on variant
            if variant == "cf-only":
                candidates = cf_candidates
            elif variant == "bandit-only":
                candidates = bandit_candidates
            elif variant == "online-only":
                candidates = online_candidates
            else:  # hybrid
                # Convert to QuestionDTO format for mixing
                cf_dtos = [QuestionDTO(
                    id=int(c["question_id"]),
                    content=c["content"],
                    difficulty_level=c["difficulty_level"],
                    subject_id=c["subject_id"],
                    topic_id=c["topic_id"],
                    features={"cf_score": c["score"], "method": c["method"]}
                ) for c in cf_candidates]
                
                bandit_dtos = [QuestionDTO(
                    id=int(c["question_id"]),
                    content=c["content"],
                    difficulty_level=c["difficulty_level"],
                    subject_id=c["subject_id"],
                    topic_id=c["topic_id"],
                    features={"bandit_score": c["score"], "method": c["method"]}
                ) for c in bandit_candidates]
                
                online_dtos = [QuestionDTO(
                    id=int(c["question_id"]),
                    content=c["content"],
                    difficulty_level=c["difficulty_level"],
                    subject_id=c["subject_id"],
                    topic_id=c["topic_id"],
                    features={"online_score": c["score"], "method": c["method"]}
                ) for c in online_candidates]
                
                candidates = mix_advanced_scores(cf_dtos, bandit_dtos, online_dtos)
            
            # Rerank candidates if available
            if self.advanced_models["reranker"] and candidates:
                # Create user context from recent responses
                recent_responses = (
                    db.query(StudentResponse)
                    .filter(StudentResponse.student_id == student_id)
                    .order_by(StudentResponse.created_at.desc())
                    .limit(10)
                    .all()
                )
                
                user_context = " ".join([
                    f"Question {r.question_id}" for r in recent_responses
                ])
                
                # Rerank top candidates
                reranked_candidates = await self.advanced_models["reranker"].rerank(
                    user_context, candidates[:100], top_n=n_recommendations
                )
                
                # Convert back to dict format
                final_recommendations = []
                for candidate in reranked_candidates:
                    final_recommendations.append({
                        "question_id": int(candidate.question_id),
                        "content": candidate.content,
                        "difficulty_level": candidate.difficulty_level,
                        "subject_id": candidate.subject_id,
                        "topic_id": candidate.topic_id,
                        "score": candidate.score,
                        "method": candidate.method,
                        "features": candidate.features or {}
                    })
                
                return final_recommendations
            else:
                # Return top candidates without reranking
                return candidates[:n_recommendations]
                
        except Exception as e:
            logger.error("advanced_recommendations_error", 
                        student_id=student_id, 
                        error=str(e))
            return []

    async def process_student_response(
        self,
        db: Session,
        student_id: int,
        question_id: int,
        answer: str,
        is_correct: bool,
        response_time: float,
        feedback: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        subject = None
        try:
            question = db.query(Question).filter(Question.id == question_id).first()
            subject = getattr(question, "subject_id", "unknown")
        except Exception:
            subject = "unknown"
        env = "prod"  # veya settings.ENVIRONMENT
        if getattr(settings, "USE_PROMETHEUS_HISTOGRAM", True):
            with model_update_duration.labels(
                subject=str(subject), 
                environment=env,
                model_type=self.model_type
            ).time():
                await self._process_student_response_inner(
                    db,
                    student_id,
                    question_id,
                    answer,
                    is_correct,
                    response_time,
                    feedback,
                    request_id,
                )
        else:
            await self._process_student_response_inner(
                db,
                student_id,
                question_id,
                answer,
                is_correct,
                response_time,
                feedback,
                request_id,
            )

    async def _process_student_response_inner(
        self,
        db,
        student_id,
        question_id,
        answer,
        is_correct,
        response_time,
        feedback,
        request_id,
    ):
        """Gerçek modellerin feedback loop'u"""
        try:
            # Question ID'yi güvenli şekilde dönüştür
            qid = int(getattr(question_id, "id", question_id) if hasattr(question_id, "id") else question_id)
            
            # Reward hesapla
            reward = 1.0 if is_correct else 0.0
            
            # Kullanıcı ve soru özelliklerini al
            user_features = self.get_student_features(db, student_id)
            question_features = self.get_question_features(db, qid)
            
            # Prometheus metrics ile model güncellemeleri
            env = getattr(settings, "ENVIRONMENT", "development")
            subject = question_features.get("subject_id", "unknown")
            
            # Gerçek modelleri güncelle
            if self.model_type == "bandit":
                with model_update_duration.labels(subject=str(subject), environment=env).time():
                    self.bandit_model.update(user_features, question_features, qid, reward)
                model_update_counter.labels(model_type="bandit", subject=str(subject), environment=env).inc()
                
            elif self.model_type == "online":
                with model_update_duration.labels(subject=str(subject), environment=env).time():
                    self.online_model.update(student_id, user_features, question_features, reward)
                model_update_counter.labels(model_type="online", subject=str(subject), environment=env).inc()
            
            # River model güncelle (her zaman)
            self.update_river_model(user_features, question_features, is_correct, response_time)
            
            # LinUCB model güncelle (her zaman)
            self.update_linucb_model(user_features, question_features, qid, reward)
            
            # Student level güncelle
            update_student_level(db, student_id, is_correct, response_time)
            
            structlog.get_logger().info(
                "student_response_processed",
                student_id=student_id,
                question_id=qid,
                is_correct=is_correct,
                reward=reward,
                model_type=self.model_type
            )
            
        except Exception as e:
            structlog.get_logger().error("process_student_response_error", 
                        student_id=student_id, 
                        question_id=question_id, 
                        error=str(e))
            # Fallback: Sadece temel güncellemeleri yap
            try:
                update_student_level(db, student_id, is_correct, response_time)
            except Exception as fallback_error:
                structlog.get_logger().error("fallback_update_error", error=str(fallback_error))
        question = db.query(Question).filter(Question.id == question_id).first()
        if not question:
            return

        student_features = self.get_student_features(db, student_id)
        question_features = self.get_question_features(db, question_id)
        if self.model_type == "linucb":
            self.update_linucb_model(
                student_features, question_features, question_id, 1 if is_correct else 0
            )
        else:
            self.update_river_model(
                student_features, question_features, is_correct, response_time
            )

        # Feedback entegrasyonu örneği (dummy):
        if feedback:
            # Burada feedback'i loglayabilir veya modele ekleyebilirsiniz
            print(
                f"Feedback for student {student_id}, question {question_id}: {feedback}"
            )

        try:
            update_student_level(db, student_id, question.difficulty_level, is_correct)
        except Exception as e:
            print(f"Seviye güncellenirken hata: {e}")

        # Öğrenci özellikleri değiştiği için cache'i temizle
        if self.redis_client is not None:
            self.redis_client.delete(f"recommendations:{student_id}")
        model_update_counter.inc()
        logger = get_logger_with_context(
            request_id=request_id, student_id=student_id, question_id=question_id
        )
        logger.info(
            "student_response_processed",
            is_correct=is_correct,
            response_time=response_time,
            feedback=feedback,
        )

    async def _adjust_question_difficulty_runtime(
        self, question, student_level: int, student_features: Dict[str, Any]
    ) -> int:
        """Runtime'da soru zorluğunu ayarla"""
        try:
            from app.services.llm_service import llm_service

            # Öğrenci performans verilerini hazırla
            performance_data = {
                "recent_accuracy": student_features.get("last20_accuracy", 0.5),
                "avg_response_time": student_features.get("avg_response_time", 0),
                "weak_topics": student_features.get("weak_topics", []),
                "strong_topics": student_features.get("strong_topics", []),
            }

            # LLM ile dinamik zorluk ayarı
            adjustment = await llm_service.adjust_difficulty_runtime(
                question.content,
                student_level,
                question.difficulty_level,
                performance_data,
            )

            return adjustment.get("adjusted_difficulty", question.difficulty_level)

        except Exception:
            # Fallback: Basit mantık
            accuracy = student_features.get("last20_accuracy", 0.5)
            if accuracy > 0.8 and question.difficulty_level < 5:
                return question.difficulty_level + 1
            elif accuracy < 0.4 and question.difficulty_level > 1:
                return question.difficulty_level - 1
            else:
                return question.difficulty_level

    async def get_adaptive_hint(
        self, question_id: int, student_id: int, db: Session
    ) -> str:
        """Öğrenci durumuna göre adaptif ipucu al"""
        try:
            from app.services.llm_service import llm_service

            # Öğrenci ve soru bilgilerini al
            question = db.query(Question).filter(Question.id == question_id).first()
            student_features = self.get_student_features(db, student_id)

            if not question:
                return "Soru bulunamadı."

            # Öğrenci durumunu analiz et
            student_level = student_features.get("level", 1)

            # Adaptif ipucu üret
            adaptive_hint = await llm_service.generate_adaptive_hint(
                question=str(question.content),
                hint_level=student_level,
                hint_style="guided",
                student_context=student_features,
                previous_attempts=[],
            )

            return adaptive_hint

        except Exception:
            # Fallback: Veritabanındaki hazır ipucu
            question = db.query(Question).filter(Question.id == question_id).first()
            return "İpucu bulunamadı."

    async def get_contextual_explanation(
        self, question_id: int, student_id: int, student_answer: str, db: Session
    ) -> str:
        """Öğrencinin cevabına göre bağlamsal açıklama al"""
        try:
            from app.services.llm_service import llm_service

            # Öğrenci ve soru bilgilerini al
            question = db.query(Question).filter(Question.id == question_id).first()
            student_features = self.get_student_features(db, student_id)

            if not question:
                return "Soru bulunamadı."

            # Bağlamsal açıklama üret
            contextual_explanation_result = (
                await llm_service.generate_contextual_explanation(
                    question=str(question.content),
                    student_answer=student_answer,
                    correct_answer=str(question.correct_answer),
                    context=student_features,
                    depth="medium",
                )
            )

            return contextual_explanation_result.get(
                "explanation", "Açıklama bulunamadı."
            )

        except Exception:
            # Fallback: Veritabanındaki hazır açıklama
            question = db.query(Question).filter(Question.id == question_id).first()
            if question is not None and question.explanation is not None:
                return str(question.explanation)
            else:
                return "Açıklama bulunamadı."

    async def rerank(
        self,
        candidates: List[Dict[str, Any]],
        user_profile: Dict[str, Any],
        difficulty: Optional[int] = None,
        db: Session = None
    ) -> List[Dict[str, Any]]:
        """Candidates'ı user profile ve difficulty'a göre yeniden sırala"""
        try:
            if not candidates:
                return []
            
            # Collaborative filtering score
            cf_scores = {}
            try:
                cf_recommendations = await self.get_cf_recommendations(
                    db, user_profile.get("user_id", 0), top_k=len(candidates)
                )
                for rec in cf_recommendations:
                    cf_scores[rec.question_id] = rec.score
            except Exception as e:
                logger.warning("cf_scoring_failed", error=str(e))
            
            # Bandit score
            bandit_scores = {}
            try:
                bandit_recommendations = await self.get_bandit_recommendations(
                    db, user_profile.get("user_id", 0), user_profile, top_k=len(candidates)
                )
                for rec in bandit_recommendations:
                    bandit_scores[rec["question_id"]] = rec.get("score", 0.0)
            except Exception as e:
                logger.warning("bandit_scoring_failed", error=str(e))
            
            # Online learning score
            online_scores = {}
            try:
                online_recommendations = await self.get_online_recommendations(
                    db, user_profile.get("user_id", 0), user_profile, top_k=len(candidates)
                )
                for rec in online_recommendations:
                    online_scores[rec["question_id"]] = rec.get("score", 0.0)
            except Exception as e:
                logger.warning("online_scoring_failed", error=str(e))
            
            # Rerank candidates with ensemble scoring
            reranked_candidates = []
            for candidate in candidates:
                question_id = candidate.get("question_id")
                
                # Get scores from different models
                cf_score = cf_scores.get(question_id, 0.0)
                bandit_score = bandit_scores.get(question_id, 0.0)
                online_score = online_scores.get(question_id, 0.0)
                
                # Ensemble scoring (weighted average)
                ensemble_score = (
                    0.4 * cf_score +
                    0.3 * bandit_score +
                    0.3 * online_score
                )
                
                # Difficulty adjustment
                if difficulty and candidate.get("difficulty_level"):
                    difficulty_diff = abs(candidate["difficulty_level"] - difficulty)
                    difficulty_penalty = max(0, difficulty_diff - 1) * 0.1
                    ensemble_score -= difficulty_penalty
                
                # User level adjustment
                user_level = user_profile.get("level", 3.0)
                if candidate.get("difficulty_level"):
                    level_diff = abs(candidate["difficulty_level"] - user_level)
                    level_penalty = max(0, level_diff - 1) * 0.05
                    ensemble_score -= level_penalty
                
                reranked_candidates.append({
                    **candidate,
                    "ensemble_score": ensemble_score,
                    "cf_score": cf_score,
                    "bandit_score": bandit_score,
                    "online_score": online_score
                })
            
            # Sort by ensemble score
            reranked_candidates.sort(key=lambda x: x["ensemble_score"], reverse=True)
            
            logger.info("rerank_completed", 
                       candidates_count=len(candidates),
                       reranked_count=len(reranked_candidates))
            
            return reranked_candidates
            
        except Exception as e:
            logger.error("rerank_error", error=str(e))
            # Fallback to original candidates
            return candidates


# Global instance
recommendation_service = RecommendationService()
