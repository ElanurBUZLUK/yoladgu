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

# --- River pipeline ---
def build_river_model():
    return compose.Pipeline(
        # HATA DÜZELTİLDİ: OneHotEncoder 'preprocessing' modülünden çağrılıyor
        preprocessing.OneHotEncoder(),
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression()
    )

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

    def get_recommendations(self, db: Session, student_id: int, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        cache_key = f"recommendations:{student_id}:{self.model_type}"
        cached = self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached.decode('utf-8'))
        student_features = self.get_student_features(db, student_id)
        questions = db.query(Question).filter(Question.is_active == True).all()
        recommendations = []
        if self.model_type == 'linucb':
            candidate_questions = []
            for question in questions:
                answered = db.query(StudentResponse).filter(
                    StudentResponse.student_id == student_id,
                    StudentResponse.question_id == question.id
                ).first()
                if answered:
                    continue
                question_features = self.get_question_features(db, question.id)
                q_dict = {'id': question.id, **question_features}
                candidate_questions.append(q_dict)
            if not candidate_questions:
                return []
            selected_id = self.linucb_bandit.select_question(student_features, candidate_questions)
            selected_question = next(q for q in questions if q.id == selected_id)
            recommendations.append({
                'question_id': selected_question.id,
                'score': 1.0,
                'question': {
                    "id": selected_question.id,
                    "content": selected_question.content,
                    "question_type": selected_question.question_type,
                    "difficulty_level": selected_question.difficulty_level,
                    "subject_id": selected_question.subject_id,
                    "options": selected_question.options,
                    "correct_answer": selected_question.correct_answer,
                    "explanation": selected_question.explanation,
                    "tags": selected_question.tags,
                    "created_by": selected_question.created_by,
                    "is_active": selected_question.is_active,
                    "created_at": str(selected_question.created_at),
                    "updated_at": str(selected_question.updated_at)
                }
            })
        else:
            for question in questions:
                # Öğrencinin daha önce cevapladığı soruları tekrar önerme (isteğe bağlı)
                answered = db.query(StudentResponse).filter(
                    StudentResponse.student_id == student_id,
                    StudentResponse.question_id == question.id
                ).first()
                if answered:
                    continue

                question_features = self.get_question_features(db, question.id)
                combined_features = {**student_features, **question_features}

                # predict_one'a gönderilen x'in sadece nümerik ve one-hot encode edilebilir olması lazım
                # Modelin beklediği formatta özellik seti oluşturmak önemlidir
                # Bu kısım modelin eğitimine göre daha detaylı ele alınabilir
                river_score = self.river_model.predict_proba_one(combined_features).get(1, 0.5) # Olasılığı al

                # Kural tabanlı ek skorlar
                student_avg_difficulty = student_features.get('avg_difficulty', 2.5)
                difficulty_match = 1 - abs(question.difficulty_level - student_avg_difficulty) / 5
                skill_match = 0
                question_skills = db.query(QuestionSkill).filter(
                    QuestionSkill.question_id == question.id
                ).all()
                if question_skills:
                    total_weight = sum(qs.weight for qs in question_skills)
                    for qs in question_skills:
                        skill_mastery = student_features.get(f'skill_{qs.skill_id}_mastery', 0.5) # Bilinmeyen için 0.5 varsayalım
                        skill_match += (1 - skill_mastery) * qs.weight # Ustalığı düşük olan skilli öner
                    skill_match /= total_weight if total_weight > 0 else 1

                # Final skor (ağırlıklar ayarlanabilir)
                final_score = (river_score * 0.4) + (difficulty_match * 0.3) + (skill_match * 0.3)

                # Pydantic modelleri JSON serileştirmesi için uygun değil, bu yüzden elle dict oluşturuyoruz
                recommendations.append({
                    'question_id': question.id,
                    'score': final_score,
                    'river_score': river_score,
                    'difficulty_match': difficulty_match,
                    'skill_match': skill_match,
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
                        "created_at": question.created_at,
                        "updated_at": question.updated_at
                    }
                })

        recommendations.sort(key=lambda x: x['score'], reverse=True)
        top_recommendations = recommendations[:n_recommendations]

        # Redis'e kaydederken Pydantic modellerini değil, serileştirilebilir dict'i kullan
        # datetime objeleri için default converter ekle
        self.redis_client.setex(
            cache_key,
            300,
            json.dumps(top_recommendations, default=str)
        )
        return top_recommendations

    def process_student_response(self, db: Session, student_id: int, question_id: int,
                               answer: str, is_correct: bool, response_time: float):
        question = db.query(Question).filter(Question.id == question_id).first()
        if not question: return

        student_features = self.get_student_features(db, student_id)
        question_features = self.get_question_features(db, question_id)
        if self.model_type == 'linucb':
            self.update_linucb_model(student_features, question_features, question_id, 1 if is_correct else 0)
        else:
            self.update_river_model(student_features, question_features, is_correct, response_time)

        try:
            update_student_level(db, student_id, question.difficulty_level, is_correct)
        except Exception as e:
            print(f"Seviye güncellenirken hata: {e}")

        # Öğrenci özellikleri değiştiği için cache'i temizle
        self.redis_client.delete(f"recommendations:{student_id}")

# Global instance
recommendation_service = RecommendationService() 