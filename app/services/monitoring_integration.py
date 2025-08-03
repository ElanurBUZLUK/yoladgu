"""
Monitoring integration for ML models and services
Hooks into existing services to provide comprehensive metrics
"""

import functools
import time
from typing import Callable

import structlog
from app.core.monitoring import prometheus_monitoring

logger = structlog.get_logger()


def monitor_ml_operation(operation_type: str, model_type: str = "unknown"):
    """Decorator to monitor ML operations"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)

                # Record successful operation
                duration = time.time() - start_time

                if operation_type == "prediction":
                    prometheus_monitoring.update_ml_metrics(
                        model_type, "river_prediction_time", duration
                    )
                elif operation_type == "update":
                    prometheus_monitoring.update_ml_metrics(
                        model_type, "river_update", 1.0
                    )

                return result

            except Exception as e:
                logger.error(
                    "ml_operation_error",
                    operation=operation_type,
                    model=model_type,
                    error=str(e),
                )
                raise

        return wrapper

    return decorator


def monitor_embedding_operation(operation_type: str):
    """Decorator to monitor embedding operations"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)

                # Determine cache status if applicable
                cache_status = "unknown"
                if hasattr(kwargs, "cache_hit"):
                    cache_status = "hit" if kwargs["cache_hit"] else "miss"

                # Record metrics
                if operation_type == "computation":
                    prometheus_monitoring.update_embedding_metrics(
                        "computation",
                        labels={
                            "model_type": getattr(
                                args[0], "current_model_key", "unknown"
                            ),
                            "cache_status": cache_status,
                        },
                    )
                elif operation_type == "similarity":
                    prometheus_monitoring.update_embedding_metrics("similarity")

                return result

            except Exception as e:
                logger.error(
                    "embedding_operation_error", operation=operation_type, error=str(e)
                )
                raise

        return wrapper

    return decorator


def monitor_database_operation(query_type: str, table: str = "unknown"):
    """Decorator to monitor database operations"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)

                # Record successful query
                duration = time.time() - start_time
                prometheus_monitoring.update_database_metrics(
                    "query_duration",
                    duration,
                    labels={"query_type": query_type, "table": table},
                )

                return result

            except Exception as e:
                # Record database error
                prometheus_monitoring.update_database_metrics(
                    "error",
                    labels={
                        "database_type": "postgresql",
                        "error_type": type(e).__name__,
                    },
                )
                logger.error(
                    "database_operation_error",
                    query_type=query_type,
                    table=table,
                    error=str(e),
                )
                raise

        return wrapper

    return decorator


class StudentResponseMonitor:
    """Monitor student responses for business metrics"""

    @staticmethod
    def record_response(
        student_id: int,
        question_id: int,
        is_correct: bool,
        difficulty_level: int,
        response_time: float,
    ):
        """Record a student response"""
        try:
            # Update business metrics
            prometheus_monitoring.update_business_metrics(
                "student_response",
                labels={"is_correct": is_correct, "difficulty_level": difficulty_level},
            )

            # Update session duration if applicable
            prometheus_monitoring.update_business_metrics(
                "session_duration",
                response_time,
                labels={"session_type": "question_solving"},
            )

            logger.info(
                "student_response_recorded",
                student_id=student_id,
                question_id=question_id,
                is_correct=is_correct,
                difficulty=difficulty_level,
            )

        except Exception as e:
            logger.error("student_response_monitoring_error", error=str(e))


class RecommendationMonitor:
    """Monitor recommendation system performance"""

    @staticmethod
    def record_recommendation_served(
        question_id: int, difficulty_level: int, subject: str, algorithm_type: str
    ):
        """Record a recommendation being served"""
        try:
            prometheus_monitoring.update_business_metrics(
                "questions_served",
                labels={"difficulty_level": str(difficulty_level), "subject": subject},
            )

            logger.info(
                "recommendation_served",
                question_id=question_id,
                algorithm=algorithm_type,
            )

        except Exception as e:
            logger.error("recommendation_monitoring_error", error=str(e))

    @staticmethod
    def update_algorithm_accuracy(algorithm_type: str, accuracy: float):
        """Update recommendation algorithm accuracy"""
        try:
            prometheus_monitoring.update_business_metrics(
                "recommendation_accuracy",
                accuracy,
                labels={"algorithm_type": algorithm_type},
            )

        except Exception as e:
            logger.error("accuracy_monitoring_error", error=str(e))


# Global monitoring instances
student_monitor = StudentResponseMonitor()
recommendation_monitor = RecommendationMonitor()
