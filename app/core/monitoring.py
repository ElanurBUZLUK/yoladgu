"""
Enhanced Prometheus Monitoring Setup
Comprehensive metrics for the Yoladgu recommendation system
"""

from typing import Dict, Optional

import psutil
import structlog
from prometheus_client import Counter, Enum, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

logger = structlog.get_logger()

# === Core Application Metrics ===

# Request metrics (handled by instrumentator)
REQUEST_COUNT = Counter(
    "yoladgu_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_DURATION = Histogram(
    "yoladgu_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

# === ML Model Metrics ===

# River online learning
RIVER_MODEL_UPDATES = Counter(
    "yoladgu_river_model_updates_total",
    "Total number of River model updates",
    ["model_type"],
)

RIVER_PREDICTION_TIME = Histogram(
    "yoladgu_river_prediction_seconds",
    "Time taken for River model predictions",
    ["model_type"],
)

RIVER_MODEL_ACCURACY = Gauge(
    "yoladgu_river_model_accuracy", "Current accuracy of River models", ["model_type"]
)

# LinUCB Bandit
BANDIT_SELECTIONS = Counter(
    "yoladgu_bandit_selections_total",
    "Total number of bandit arm selections",
    ["exploration_type"],
)

BANDIT_REWARDS = Histogram(
    "yoladgu_bandit_rewards", "Distribution of bandit rewards", ["question_difficulty"]
)

BANDIT_EXPLORATION_RATIO = Gauge(
    "yoladgu_bandit_exploration_ratio", "Current exploration vs exploitation ratio"
)

# Collaborative Filtering
COLLABORATIVE_FILTER_TRAINING = Counter(
    "yoladgu_collaborative_training_total",
    "Total number of collaborative filter training sessions",
)

COLLABORATIVE_FILTER_RECOMMENDATIONS = Counter(
    "yoladgu_collaborative_recommendations_total",
    "Total number of collaborative recommendations generated",
    ["method_type"],
)

# === Embedding Service Metrics ===

EMBEDDING_COMPUTATIONS = Counter(
    "yoladgu_embedding_computations_total",
    "Total number of embedding computations",
    ["model_type", "cache_status"],
)

EMBEDDING_SIMILARITY_COMPUTATIONS = Counter(
    "yoladgu_embedding_similarities_total", "Total number of similarity computations"
)

EMBEDDING_CACHE_HIT_RATE = Gauge(
    "yoladgu_embedding_cache_hit_rate", "Embedding cache hit rate percentage"
)

EMBEDDING_MODEL_LOAD_TIME = Histogram(
    "yoladgu_embedding_model_load_seconds",
    "Time taken to load embedding models",
    ["model_name"],
)

# === Database Metrics ===

DATABASE_CONNECTIONS = Gauge(
    "yoladgu_database_connections_active",
    "Number of active database connections",
    ["database_type"],
)

DATABASE_QUERY_DURATION = Histogram(
    "yoladgu_database_query_seconds",
    "Database query execution time",
    ["query_type", "table"],
)

DATABASE_ERRORS = Counter(
    "yoladgu_database_errors_total",
    "Total number of database errors",
    ["database_type", "error_type"],
)

# === Business Metrics ===

QUESTIONS_SERVED = Counter(
    "yoladgu_questions_served_total",
    "Total number of questions served to students",
    ["difficulty_level", "subject"],
)

STUDENT_RESPONSES = Counter(
    "yoladgu_student_responses_total",
    "Total number of student responses",
    ["is_correct", "difficulty_level"],
)

RECOMMENDATION_ACCURACY = Gauge(
    "yoladgu_recommendation_accuracy",
    "Overall recommendation accuracy",
    ["algorithm_type"],
)

SESSION_DURATION = Histogram(
    "yoladgu_session_duration_seconds", "Student session duration", ["session_type"]
)

# === System Metrics ===

SYSTEM_MEMORY_USAGE = Gauge(
    "yoladgu_system_memory_usage_bytes", "System memory usage in bytes", ["memory_type"]
)

REDIS_OPERATIONS = Counter(
    "yoladgu_redis_operations_total",
    "Total number of Redis operations",
    ["operation_type", "key_type"],
)

NEO4J_QUERIES = Counter(
    "yoladgu_neo4j_queries_total", "Total number of Neo4j queries", ["query_type"]
)

# === Service Health Metrics ===

SERVICE_HEALTH = Enum(
    "yoladgu_service_health",
    "Health status of various services",
    ["service_name"],
    states=["healthy", "unhealthy", "unknown"],
)

HEALTH_CHECK_DURATION = Histogram(
    "yoladgu_health_check_duration_seconds",
    "Time taken for health checks",
    ["service_name"],
)


class PrometheusMonitoring:
    """Enhanced Prometheus monitoring class"""

    def __init__(self):
        self.instrumentator = None
        self.initialized = False

    def initialize_instrumentator(self, app):
        """Initialize FastAPI instrumentator with custom metrics"""
        try:
            # Create instrumentator with custom configuration
            self.instrumentator = Instrumentator(
                should_group_status_codes=True,
                should_ignore_untemplated=True,
                should_respect_env_var=True,
                should_instrument_requests_inprogress=True,
                excluded_handlers=["/health", "/metrics"],
                env_var_name="ENABLE_METRICS",
                inprogress_name="yoladgu_inprogress",
                inprogress_labels=True,
            )

            # Add default metrics
            self.instrumentator.instrument(app)

            # Custom metrics will be handled separately

            # Expose metrics endpoint
            self.instrumentator.expose(
                app, endpoint="/metrics", include_in_schema=False
            )

            self.initialized = True
            logger.info("prometheus_instrumentator_initialized")

        except Exception as e:
            logger.error("prometheus_instrumentator_init_error", error=str(e))
            # Fallback to basic setup
            try:
                self.instrumentator = Instrumentator()
                self.instrumentator.instrument(app)
                self.instrumentator.expose(app, endpoint="/metrics")
                self.initialized = True
                logger.info("prometheus_instrumentator_fallback_initialized")
            except Exception as fallback_error:
                logger.error(
                    "prometheus_instrumentator_fallback_error",
                    error=str(fallback_error),
                )
                # Last resort - manual metrics endpoint
                self._setup_manual_metrics_endpoint(app)

    def _setup_custom_metrics(self):
        """Setup custom metrics - simpler approach without instrumentator hooks"""
        # Initialize all custom metrics by calling them once
        try:
            # Initialize counters
            REQUEST_COUNT.labels(method="GET", endpoint="/health", status="200")
            EMBEDDING_COMPUTATIONS.labels(model_type="unknown", cache_status="unknown")
            STUDENT_RESPONSES.labels(is_correct="true", difficulty_level="1")

            # Initialize gauges
            RIVER_MODEL_ACCURACY.labels(model_type="unknown").set(0)
            EMBEDDING_CACHE_HIT_RATE.set(0)
            SERVICE_HEALTH.labels(service_name="unknown").state("unknown")

            logger.info("custom_metrics_initialized")

        except Exception as e:
            logger.error("custom_metrics_setup_error", error=str(e))

    def _setup_manual_metrics_endpoint(self, app):
        """Manual metrics endpoint setup as fallback"""
        try:
            from fastapi import Response
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

            @app.get("/metrics")
            def _metrics_endpoint():
                return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

            self.initialized = True
            logger.info("manual_metrics_endpoint_setup")

        except Exception as e:
            logger.error("manual_metrics_setup_error", error=str(e))

    def update_ml_metrics(
        self,
        model_type: str,
        metric_type: str,
        value: float,
        labels: Optional[Dict] = None,
    ):
        """Update ML model metrics"""
        try:
            if metric_type == "river_update":
                RIVER_MODEL_UPDATES.labels(model_type=model_type).inc()
            elif metric_type == "river_prediction_time":
                RIVER_PREDICTION_TIME.labels(model_type=model_type).observe(value)
            elif metric_type == "river_accuracy":
                RIVER_MODEL_ACCURACY.labels(model_type=model_type).set(value)
            elif metric_type == "bandit_selection":
                exploration_type = (
                    labels.get("exploration_type", "unknown") if labels else "unknown"
                )
                BANDIT_SELECTIONS.labels(exploration_type=exploration_type).inc()
            elif metric_type == "bandit_reward":
                difficulty = (
                    labels.get("difficulty", "unknown") if labels else "unknown"
                )
                BANDIT_REWARDS.labels(question_difficulty=difficulty).observe(value)
            elif metric_type == "bandit_exploration_ratio":
                BANDIT_EXPLORATION_RATIO.set(value)

        except Exception as e:
            logger.error("ml_metrics_update_error", error=str(e))

    def update_embedding_metrics(
        self, metric_type: str, value: float = 1.0, labels: Optional[Dict] = None
    ):
        """Update embedding service metrics"""
        try:
            if metric_type == "computation":
                model_type = (
                    labels.get("model_type", "unknown") if labels else "unknown"
                )
                cache_status = (
                    labels.get("cache_status", "unknown") if labels else "unknown"
                )
                EMBEDDING_COMPUTATIONS.labels(
                    model_type=model_type, cache_status=cache_status
                ).inc()
            elif metric_type == "similarity":
                EMBEDDING_SIMILARITY_COMPUTATIONS.inc()
            elif metric_type == "cache_hit_rate":
                EMBEDDING_CACHE_HIT_RATE.set(value)
            elif metric_type == "model_load_time":
                model_name = (
                    labels.get("model_name", "unknown") if labels else "unknown"
                )
                EMBEDDING_MODEL_LOAD_TIME.labels(model_name=model_name).observe(value)

        except Exception as e:
            logger.error("embedding_metrics_update_error", error=str(e))

    def update_database_metrics(
        self, metric_type: str, value: float = 1.0, labels: Optional[Dict] = None
    ):
        """Update database metrics"""
        try:
            if metric_type == "query_duration":
                query_type = (
                    labels.get("query_type", "unknown") if labels else "unknown"
                )
                table = labels.get("table", "unknown") if labels else "unknown"
                DATABASE_QUERY_DURATION.labels(
                    query_type=query_type, table=table
                ).observe(value)
            elif metric_type == "error":
                db_type = (
                    labels.get("database_type", "unknown") if labels else "unknown"
                )
                error_type = (
                    labels.get("error_type", "unknown") if labels else "unknown"
                )
                DATABASE_ERRORS.labels(
                    database_type=db_type, error_type=error_type
                ).inc()

        except Exception as e:
            logger.error("database_metrics_update_error", error=str(e))

    def update_business_metrics(
        self, metric_type: str, value: float = 1.0, labels: Optional[Dict] = None
    ):
        """Update business metrics"""
        try:
            if metric_type == "student_response":
                is_correct = (
                    labels.get("is_correct", "unknown") if labels else "unknown"
                )
                difficulty = (
                    labels.get("difficulty_level", "unknown") if labels else "unknown"
                )
                STUDENT_RESPONSES.labels(
                    is_correct=str(is_correct), difficulty_level=str(difficulty)
                ).inc()
            elif metric_type == "recommendation_accuracy":
                algorithm = (
                    labels.get("algorithm_type", "unknown") if labels else "unknown"
                )
                RECOMMENDATION_ACCURACY.labels(algorithm_type=algorithm).set(value)
            elif metric_type == "session_duration":
                session_type = (
                    labels.get("session_type", "unknown") if labels else "unknown"
                )
                SESSION_DURATION.labels(session_type=session_type).observe(value)

        except Exception as e:
            logger.error("business_metrics_update_error", error=str(e))

    def update_system_metrics(self):
        """Update system-level metrics"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            SYSTEM_MEMORY_USAGE.labels(memory_type="used").set(memory.used)
            SYSTEM_MEMORY_USAGE.labels(memory_type="available").set(memory.available)
            SYSTEM_MEMORY_USAGE.labels(memory_type="total").set(memory.total)

            # Service health (will be updated by health checks)
            # This is handled separately in health check functions

        except Exception as e:
            logger.error("system_metrics_update_error", error=str(e))

    def update_service_health(self, service_name: str, status: str):
        """Update service health metrics"""
        try:
            SERVICE_HEALTH.labels(service_name=service_name).state(status)
        except Exception as e:
            logger.error("service_health_update_error", error=str(e))

    def record_health_check_duration(self, service_name: str, duration: float):
        """Record health check duration"""
        try:
            HEALTH_CHECK_DURATION.labels(service_name=service_name).observe(duration)
        except Exception as e:
            logger.error("health_check_duration_error", error=str(e))


# Global monitoring instance
prometheus_monitoring = PrometheusMonitoring()
