from typing import Dict, Any, List, Optional, Tuple
from app.core.dependencies import (
    get_retriever_service, get_linucb_service, get_ftrl_service, get_feature_store
)
from app.core.config import settings
from app.services.content.questions_service import QuestionsService


def _safe_features(fs, user_id: int, question_id: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    user_feats = fs.get_user_features(user_id) if fs else {}
    q_feats = fs.get_question_features(question_id) if fs else {}
    return user_feats or {}, q_feats or {}


def pick_next_question(
    user_id: int,
    topic_id: Optional[int],
    asked_ids: List[int],
    k_pool: int = 50,
) -> Optional[int]:
    qsvc = QuestionsService(settings.REDIS_URL)
    candidate_ids: List[int] = []
    try:
        retriever = get_retriever_service()
        candidate_ids = []  # placeholder
    except Exception:
        candidate_ids = []

    if not candidate_ids:
        candidate_ids = qsvc.random_by_topic(topic_id, k=k_pool)

    candidate_ids = [qid for qid in candidate_ids if qid not in asked_ids]
    if not candidate_ids:
        return None

    fs = get_feature_store()
    linucb = get_linucb_service()
    ftrl = get_ftrl_service()

    best_qid, best_score = None, -1e9
    for qid in candidate_ids:
        uf, qf = _safe_features(fs, user_id, qid)
        try:
            s_bandit = linucb.predict(uf, qf, qid)
        except Exception:
            s_bandit = 0.0
        try:
            s_online = ftrl.predict(user_id, uf, qf)
        except Exception:
            s_online = 0.0
        score = 0.6 * s_bandit + 0.4 * s_online
        if score > best_score:
            best_qid, best_score = qid, score
    return best_qid


