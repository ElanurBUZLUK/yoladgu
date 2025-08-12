from typing import Dict, Any, List, Optional, Tuple
from app.core.dependencies import (
    get_linucb_service, get_ftrl_service, get_feature_store, get_cf_model, get_peer_service
)
from app.core.config import settings
from app.services.content.questions_service import QuestionsService
from app.services.advanced_rag import AdvancedRAGService
from app.services.policy.policy_manager import get_policy_manager
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import User
from app.services.candidates.context_generator import ContextAwareCandidateGenerator


def _safe_features(fs, user_id: int, question_id: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    user_feats = fs.get_user_features(user_id) if fs else {}
    q_feats = fs.get_question_features(question_id) if fs else {}
    return user_feats or {}, q_feats or {}


async def pick_next_question(
    user: User,
    db: AsyncSession,
    topic_id: Optional[int],
    asked_ids: List[int],
    k_pool: int = 50,
) -> Optional[int]:
    qsvc = QuestionsService(settings.REDIS_URL)
    candidate_ids: List[int] = []
    # Context-aware candidate generation (vector-only, keyword-free)
    try:
        gen = ContextAwareCandidateGenerator(k_recent=5)
        candidate_ids = gen.generate_candidates(user.id, topic_id, k=k_pool)
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
    # Load dynamic policy parameters for the user (DB-backed with cache inside)
    try:
        pm = get_policy_manager()
        params = await pm.get_params_for_user(db, user)
    except Exception:
        params = {}
    w_b = float(params.get("W_BANDIT", getattr(settings, "W_BANDIT", 0.6)))
    w_o = float(params.get("W_ONLINE", getattr(settings, "W_ONLINE", 0.4)))
    w_cf = float(params.get("W_CF", getattr(settings, "W_CF", 0.0)))
    w_p = float(params.get("W_PEER", getattr(settings, "W_PEER", 0.0)))
    peer = None
    try:
        peer = get_peer_service()
    except Exception:
        peer = None
    for qid in candidate_ids:
        uf, qf = _safe_features(fs, user.id, qid)
        try:
            s_bandit = linucb.predict(uf, qf, qid)
        except Exception:
            s_bandit = 0.0
        try:
            s_online = ftrl.predict(user.id, uf, qf)
        except Exception:
            s_online = 0.0
        try:
            s_cf = float(get_cf_model().score(user.id, qid)) if w_cf > 0.0 else 0.0
        except Exception:
            s_cf = 0.0
        try:
            s_peer = float(peer.peer_score_for(user.id, qid)) if (peer is not None and w_p > 0.0) else 0.0
        except Exception:
            s_peer = 0.0
        score = w_b * s_bandit + w_o * s_online + w_cf * s_cf + w_p * s_peer
        if score > best_score:
            best_qid, best_score = qid, score
    return best_qid


