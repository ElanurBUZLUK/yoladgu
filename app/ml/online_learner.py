import json
import pickle
from typing import Optional

import structlog
from river import compose, ensemble, linear_model, naive_bayes, preprocessing, tree

logger = structlog.get_logger()


class EnhancedOnlineLearner:
    """Gelişmiş River tabanlı online learning sistemi"""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.model_key = "ml:user_success_model"
        self.performance_key = "ml:model_performance"
        self.learning_rate = 0.01

        # Modeli Redis'ten yükle veya yeni oluştur
        try:
            model_bytes = self.redis.get(self.model_key)
            if model_bytes:
                self.model = pickle.loads(model_bytes)
                logger.info("online_model_loaded_from_redis")
            else:
                raise Exception("No cached model found")
        except Exception as e:
            logger.info("creating_new_online_model", error=str(e))
            self.model = self._create_ensemble_model()
            self._save_model()

    def _create_ensemble_model(self):
        """Ensemble model oluştur - farklı algoritmaları birleştir"""
        try:
            return ensemble.VotingClassifier(
                [
                    (
                        "logistic",
                        compose.Pipeline(
                            preprocessing.StandardScaler(),
                            linear_model.LogisticRegression(),
                        ),
                    ),
                    ("naive_bayes", naive_bayes.GaussianNB()),
                    (
                        "tree",
                        tree.HoeffdingTreeClassifier(
                            grace_period=200,
                            split_confidence=0.0001,
                            tie_threshold=0.05,
                        ),
                    ),
                ]
            )
        except Exception as e:
            logger.warning("ensemble_creation_failed_using_simple_model", error=str(e))
            # Fallback to simple logistic regression
            return compose.Pipeline(
                preprocessing.StandardScaler(), linear_model.LogisticRegression()
            )

    def predict_success_probability(
        self, user_features: dict, question_features: dict
    ) -> float:
        """Başarı olasılığını tahmin et"""
        try:
            combined_features = self._combine_features(user_features, question_features)
            prob_dict = self.model.predict_proba_one(combined_features)

            # True sınıfının olasılığını al, yoksa neutral değer döndür
            probability = prob_dict.get(True, 0.5)

            logger.debug(
                "success_probability_predicted",
                probability=probability,
                user_id=user_features.get("user_id"),
                question_id=question_features.get("question_id"),
            )

            return float(probability)

        except Exception as e:
            logger.error("prediction_error", error=str(e))
            return 0.5  # Neutral probability as fallback

    def learn_from_answer(
        self,
        user_features: dict,
        question_features: dict,
        is_correct: bool,
        response_time: int,
        confidence_score: Optional[float] = None,
    ):
        """Öğrenci cevabından öğren"""
        try:
            combined_features = self._combine_features(
                user_features, question_features, response_time
            )

            # Confidence score varsa özellik olarak ekle
            if confidence_score is not None:
                combined_features["confidence_score"] = confidence_score

            # Model güncelle
            self.model.learn_one(combined_features, is_correct)

            # Performance metriklerini güncelle
            self._update_performance_metrics(is_correct, response_time)

            # Modeli kaydet (her 10 güncellemede bir)
            update_count = self.redis.incr("ml:update_count")
            if update_count % 10 == 0:
                self._save_model()

            logger.info(
                "model_updated",
                is_correct=is_correct,
                response_time=response_time,
                update_count=update_count,
            )

        except Exception as e:
            logger.error("learning_error", error=str(e))

    def _combine_features(
        self,
        user_features: dict,
        question_features: dict,
        response_time: Optional[int] = None,
    ) -> dict:
        """Özellikleri birleştir ve normalize et"""
        features = {}

        # User features
        features.update(
            {
                "user_accuracy": user_features.get("accuracy_rate_overall", 0.5),
                "user_recent_accuracy": user_features.get("accuracy_rate_last_10", 0.5),
                "user_avg_time": min(
                    user_features.get("avg_response_time", 5000) / 10000, 1.0
                ),
                "user_topic_mastery": user_features.get("topic_mastery_math", 0.5),
                "user_consecutive": min(
                    user_features.get("consecutive_correct", 0) / 10, 1.0
                ),
                "user_session_count": min(
                    user_features.get("session_question_count", 0) / 20, 1.0
                ),
            }
        )

        # Question features
        features.update(
            {
                "question_difficulty": question_features.get("difficulty_level", 3) / 5,
                "question_avg_success": question_features.get("avg_success_rate", 0.5),
                "question_avg_time": min(
                    question_features.get("avg_response_time", 5000) / 10000, 1.0
                ),
                "question_attempts": min(
                    question_features.get("attempt_count", 0) / 100, 1.0
                ),
            }
        )

        # Response time (if provided)
        if response_time is not None:
            features["current_response_time"] = min(response_time / 10000, 1.0)

            # Response time vs user average (relative speed)
            user_avg = user_features.get("avg_response_time", 5000)
            if user_avg > 0:
                features["relative_speed"] = min(response_time / user_avg, 2.0)

        return features

    def _update_performance_metrics(self, is_correct: bool, response_time: int):
        """Model performance metriklerini güncelle"""
        try:
            metrics_data = self.redis.get(self.performance_key)
            if metrics_data:
                metrics = json.loads(metrics_data)
            else:
                metrics = {
                    "total_predictions": 0,
                    "correct_predictions": 0,
                    "total_response_time": 0,
                    "prediction_count": 0,
                }

            metrics["total_predictions"] += 1
            metrics["total_response_time"] += response_time

            if is_correct:
                metrics["correct_predictions"] += 1

            # Accuracy hesapla
            if metrics["total_predictions"] > 0:
                metrics["accuracy"] = (
                    metrics["correct_predictions"] / metrics["total_predictions"]
                )
                metrics["avg_response_time"] = (
                    metrics["total_response_time"] / metrics["total_predictions"]
                )

            self.redis.setex(self.performance_key, 3600, json.dumps(metrics))

        except Exception as e:
            logger.error("metrics_update_error", error=str(e))

    def _save_model(self):
        """Modeli Redis'e kaydet"""
        try:
            model_bytes = pickle.dumps(self.model)
            self.redis.setex(self.model_key, 86400, model_bytes)  # 24 saat
            logger.info("online_model_saved_to_redis")
        except Exception as e:
            logger.error("model_save_error", error=str(e))

    def get_performance_metrics(self) -> dict:
        """Model performance metriklerini al"""
        try:
            metrics_data = self.redis.get(self.performance_key)
            if metrics_data:
                return json.loads(metrics_data)
            return {}
        except Exception as e:
            logger.error("metrics_get_error", error=str(e))
            return {}

    def reset_model(self):
        """Modeli sıfırla"""
        try:
            self.model = self._create_ensemble_model()
            self._save_model()
            self.redis.delete(self.performance_key)
            self.redis.delete("ml:update_count")
            logger.info("online_model_reset")
        except Exception as e:
            logger.error("model_reset_error", error=str(e))


# Backward compatibility
OnlineLearner = EnhancedOnlineLearner
