import random
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.adaptive import repo


class SelectorService:
    def __init__(self, cfg):
        self.tol = cfg.TOLERANCE
        self.explore = cfg.EXPLORE_RATIO

    async def serve(self, db: AsyncSession, student, k: int = 1):
        near = await repo.find_questions_by_diff_near(db, student.skill_rating, tol=self.tol, limit=50)
        pool = list(near)
        if random.random() < self.explore:
            extra = await repo.random_questions(db, limit=20)
            pool.extend(extra)
        # Unique and sort by proximity
        seen = set()
        unique = []
        for q in pool:
            if q.id not in seen:
                seen.add(q.id)
                unique.append(q)
        unique.sort(key=lambda q: abs(q.difficulty_rating - student.skill_rating))
        return unique[:k]


