from typing import Optional, Dict, Any
import structlog

# Re-export existing services
from app.core.deps import (
    get_linucb_service,  # LinUCBService
    get_ftrl_service,    # FTRLService
)
from functools import lru_cache
from app.services.cf import CFModel
from app.core.config import settings

log = structlog.get_logger()


class _FeatureStore:
    def get_user_features(self, user_id: int) -> Dict[str, float]:
        return {}

    def get_question_features(self, question_id: int) -> Dict[str, float]:
        return {}


class _EventLogger:
    def log_exposure(self, user_id: int, variant: str, payload: Dict[str, Any]):
        log.info("exposure", user_id=user_id, variant=variant, payload=payload)

    def log_outcome(self, user_id: int, variant: str, payload: Dict[str, Any]):
        log.info("outcome", user_id=user_id, variant=variant, payload=payload)


def get_feature_store() -> _FeatureStore:
    return _FeatureStore()


def get_event_logger() -> _EventLogger:
    return _EventLogger()


def get_retriever_service():
    # Optional placeholder for a retriever; return None to fallback
    return None


@lru_cache(maxsize=1)
def get_cf_model() -> CFModel:
    model = CFModel()
    path = getattr(settings, "CF_MODEL_PATH", None) or "backend/app/ml/models/cf.npz"
    try:
        model.load(path)
    except Exception:
        pass
    return model


