from __future__ import annotations

import time
from typing import Any, Dict, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import PolicyVariant, User
from app.core.config import settings


DEFAULT_PARAMS: dict[str, float] = {
    "W_CF": float(getattr(settings, "W_CF", 0.0)),
    "W_BANDIT": float(getattr(settings, "W_BANDIT", 0.6)),
    "W_ONLINE": float(getattr(settings, "W_ONLINE", 0.4)),
    "W_PEER": float(getattr(settings, "W_PEER", 0.0)),
    "EXPLORE_RATIO": float(getattr(settings, "EXPLORE_RATIO", 0.2)),
}


class PolicyManager:
    def __init__(self) -> None:
        self._cache: dict[str, PolicyVariant] = {}
        self._last_loaded: float = 0.0
        self._ttl_s: float = 300.0

    async def _load_active(self, db: AsyncSession) -> None:
        now = time.time()
        if now - self._last_loaded < self._ttl_s and self._cache:
            return
        res = await db.execute(select(PolicyVariant).where(PolicyVariant.is_active == True))
        items = res.scalars().all()
        self._cache = {pv.name: pv for pv in items}
        self._last_loaded = now

    async def get_params_for_user(self, db: AsyncSession, user: User) -> Dict[str, Any]:
        try:
            await self._load_active(db)
            # 1) cohort match
            if getattr(user, "experiment_cohort", None):
                variant = self._cache.get(str(user.experiment_cohort))
                if variant is not None:
                    return dict(variant.parameters or {})
            # 2) rule match
            for pv in self._cache.values():
                if self._matches_rule(user, pv.assignment_rule or {}):
                    return dict(pv.parameters or {})
            # 3) default by name
            if "default" in self._cache:
                return dict(self._cache["default"].parameters or {})
        except Exception:
            pass
        return DEFAULT_PARAMS.copy()

    def _matches_rule(self, user: User, rule: Dict[str, Any]) -> bool:
        if not rule:
            return False
        # Example rules: {"min_skill": 0.5, "cohort": ["A","B"]}
        try:
            if "min_skill" in rule:
                if float(getattr(user, "skill_rating", 0.0)) < float(rule["min_skill"]):
                    return False
            if "cohort" in rule:
                cohort = str(getattr(user, "experiment_cohort", ""))
                if cohort not in [str(x) for x in (rule.get("cohort") or [])]:
                    return False
            return True
        except Exception:
            return False


_singleton: Optional[PolicyManager] = None


def get_policy_manager() -> PolicyManager:
    global _singleton
    if _singleton is None:
        _singleton = PolicyManager()
    return _singleton


