from __future__ import annotations

from typing import Dict, Optional
from dataclasses import dataclass
import json
import random
from functools import lru_cache
from sqlalchemy import select, update, insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import EnsembleWeights, ABAssignment


DEFAULT_WEIGHTS: dict[str, float] = {
    "cf": 0.25,
    "bandit": 0.35,
    "online": 0.40,
    "retr": 0.0,
    "peer": 0.0,
}


@dataclass
class Variant:
    name: str
    weights: dict[str, float]
    is_active: bool = False


class EnsembleWeightsService:
    """Manages ensemble weight variants, user assignments, and reward logging.

    MVP: sticky uniform random assignment across active variants.
    Future: replace assignment with Thompson Sampling / UCB over variant rewards.
    """

    async def list_variants(self, db: AsyncSession) -> list[Variant]:
        rows = (await db.execute(select(EnsembleWeights))).scalars().all()
        return [Variant(name=r.variant, weights=dict(r.weights or {}), is_active=bool(r.is_active)) for r in rows]

    async def upsert_variant(self, db: AsyncSession, variant: str, weights: dict[str, float], is_active: Optional[bool] = None) -> None:
        row = (await db.execute(select(EnsembleWeights).where(EnsembleWeights.variant == variant))).scalar_one_or_none()
        if row is None:
            await db.execute(insert(EnsembleWeights).values(variant=variant, weights=weights, is_active=bool(is_active)))
        else:
            vals = {"weights": weights}
            if is_active is not None:
                vals["is_active"] = bool(is_active)
            await db.execute(update(EnsembleWeights).where(EnsembleWeights.variant == variant).values(**vals))
        await db.commit()

    async def set_active(self, db: AsyncSession, variant: str) -> None:
        # deactivate all, activate one
        await db.execute(update(EnsembleWeights).values(is_active=False))
        await db.execute(update(EnsembleWeights).where(EnsembleWeights.variant == variant).values(is_active=True))
        await db.commit()

    async def _get_or_assign_variant(self, db: AsyncSession, user_id: int) -> str:
        cur = (await db.execute(select(ABAssignment).where(ABAssignment.user_id == user_id))).scalar_one_or_none()
        if cur is not None:
            return cur.variant
        # assign among active variants; fallback to any; finally default
        rows = (await db.execute(select(EnsembleWeights))).scalars().all()
        actives = [r.variant for r in rows if r.is_active]
        pool = actives or [r.variant for r in rows]
        chosen = (random.choice(pool) if pool else "control")
        if not pool:
            # ensure default exists in db to avoid FK issues
            await db.execute(insert(EnsembleWeights).values(variant="control", weights=DEFAULT_WEIGHTS, is_active=True))
        await db.execute(insert(ABAssignment).values(user_id=int(user_id), variant=chosen))
        await db.commit()
        return chosen

    async def get_effective_weights(self, db: AsyncSession, user_id: int, override: Optional[dict[str, float]] = None) -> tuple[str, dict[str, float]]:
        if override:
            return ("manual", override)
        variant = await self._get_or_assign_variant(db, user_id)
        row = (await db.execute(select(EnsembleWeights).where(EnsembleWeights.variant == variant))).scalar_one_or_none()
        if row is None:
            return (variant, DEFAULT_WEIGHTS)
        return (variant, dict(row.weights or {}))

    async def get_assignment(self, db: AsyncSession, user_id: int) -> Optional[str]:
        cur = (await db.execute(select(ABAssignment).where(ABAssignment.user_id == user_id))).scalar_one_or_none()
        return cur.variant if cur else None


@lru_cache(maxsize=1)
def get_ensemble_weights_service() -> EnsembleWeightsService:
    return EnsembleWeightsService()


