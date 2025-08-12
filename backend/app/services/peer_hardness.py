from __future__ import annotations

from typing import Dict, List, Tuple, Iterable
from collections import defaultdict
from dataclasses import dataclass
from app.core.config import settings


@dataclass
class PeerParams:
    half_life_days: float = float(settings.PEER_HALF_LIFE_DAYS)
    min_neighbors: int = int(settings.PEER_MIN_NEIGHBORS)
    k_neighbors: int = int(settings.PEER_K_NEIGHBORS)
    lambda_easy: float = float(settings.PEER_LAMBDA_EASY)
    lookback_days: int = int(settings.PEER_LOOKBACK_DAYS)


def jaccard_weighted(A: Dict[int, float], B: Dict[int, float]) -> float:
    if not A and not B:
        return 0.0
    qset = set(A.keys()) | set(B.keys())
    inter = sum(min(A.get(q, 0.0), B.get(q, 0.0)) for q in qset)
    union = sum(max(A.get(q, 0.0), B.get(q, 0.0)) for q in qset)
    return (inter / union) if union > 0.0 else 0.0


class PeerStore:
    # Interface expected by PeerHardnessService; concrete impl can use Redis/PG/materialized views
    def wrong_set(self, student_id: int) -> Dict[int, float]:  # question_id -> weight
        return {}

    def right_set(self, student_id: int) -> Dict[int, float]:  # question_id -> weight
        return {}

    def has_attempt(self, student_id: int, question_id: int) -> bool:
        return False

    def sim_index_neighbors(self, student_id: int) -> Iterable[int]:
        # Optionally return precomputed neighbor candidates; fallback could be all students
        return []


class PeerHardnessService:
    def __init__(self, store: PeerStore, params: PeerParams | None = None):
        self.store = store
        self.params = params or PeerParams()

    def peer_candidates(self, student_id: int) -> List[int]:
        target_wrong = self.store.wrong_set(student_id)
        sims: List[Tuple[int, float]] = []
        for u in self.store.sim_index_neighbors(student_id):
            if u == student_id:
                continue
            w = jaccard_weighted(target_wrong, self.store.wrong_set(u))
            if w > 0.0:
                sims.append((u, w))
        sims.sort(key=lambda x: x[1], reverse=True)
        sims = sims[: self.params.k_neighbors]
        if len(sims) < self.params.min_neighbors:
            return []

        score: Dict[int, float] = defaultdict(float)
        for u, w in sims:
            for q, wu in self.store.wrong_set(u).items():
                if not self.store.has_attempt(student_id, q):
                    score[q] += w * float(wu)
            for q, wu in self.store.right_set(u).items():
                score[q] -= self.params.lambda_easy * w * float(wu)

        ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
        candidates = [q for q, s in ranked if s > 0.0]
        return candidates

    def peer_score_for(self, student_id: int, question_id: int) -> float:
        # Compute on-the-fly from neighbors; a production impl would cache per student
        target_wrong = self.store.wrong_set(student_id)
        sims: List[Tuple[int, float]] = []
        for u in self.store.sim_index_neighbors(student_id):
            if u == student_id:
                continue
            w = jaccard_weighted(target_wrong, self.store.wrong_set(u))
            if w > 0.0:
                sims.append((u, w))
        sims.sort(key=lambda x: x[1], reverse=True)
        sims = sims[: self.params.k_neighbors]
        if len(sims) < self.params.min_neighbors:
            return 0.0
        total = 0.0
        for u, w in sims:
            hard = float(self.store.wrong_set(u).get(question_id, 0.0))
            easy = float(self.store.right_set(u).get(question_id, 0.0))
            total += w * (hard - self.params.lambda_easy * easy)
        return max(0.0, total)


