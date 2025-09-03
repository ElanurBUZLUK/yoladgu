"""
Item repositories for math and English questions.
"""

from typing import Optional, List, Dict, Any
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_, or_, func

from app.db.models import MathItem, EnglishItem
from app.db.repositories.base import BaseRepository


class MathItemRepository(BaseRepository[MathItem]):
    """Math item repository with search and filtering capabilities."""
    
    def __init__(self):
        super().__init__(MathItem)
    
    async def search_by_skills(
        self,
        session: AsyncSession,
        skills: List[str],
        tenant_id: str,
        lang: str = "tr",
        difficulty_range: Optional[tuple] = None,
        limit: int = 200
    ) -> List[MathItem]:
        """Search math items by skills."""
        statement = select(MathItem).where(
            and_(
                MathItem.tenant_id == tenant_id,
                MathItem.lang == lang,
                MathItem.status == "active"
            )
        )
        
        # Filter by skills (items that contain any of the target skills)
        if skills:
            skill_conditions = []
            for skill in skills:
                # Using JSON contains operator for PostgreSQL
                skill_conditions.append(MathItem.skills.contains([skill]))
            statement = statement.where(or_(*skill_conditions))
        
        # Filter by difficulty range
        if difficulty_range:
            min_diff, max_diff = difficulty_range
            statement = statement.where(
                MathItem.difficulty_b.between(min_diff, max_diff)
            )
        
        statement = statement.limit(limit)
        result = await session.exec(statement)
        return result.all()
    
    async def get_by_difficulty_range(
        self,
        session: AsyncSession,
        min_difficulty: float,
        max_difficulty: float,
        tenant_id: str,
        lang: str = "tr",
        limit: int = 100
    ) -> List[MathItem]:
        """Get math items within difficulty range."""
        statement = select(MathItem).where(
            and_(
                MathItem.tenant_id == tenant_id,
                MathItem.lang == lang,
                MathItem.status == "active",
                MathItem.difficulty_b.between(min_difficulty, max_difficulty)
            )
        ).limit(limit)
        
        result = await session.exec(statement)
        return result.all()
    
    async def get_by_bloom_level(
        self,
        session: AsyncSession,
        bloom_level: str,
        tenant_id: str,
        lang: str = "tr",
        limit: int = 100
    ) -> List[MathItem]:
        """Get math items by Bloom taxonomy level."""
        statement = select(MathItem).where(
            and_(
                MathItem.tenant_id == tenant_id,
                MathItem.lang == lang,
                MathItem.status == "active",
                MathItem.bloom_level == bloom_level
            )
        ).limit(limit)
        
        result = await session.exec(statement)
        return result.all()
    
    async def update_difficulty_parameters(
        self,
        session: AsyncSession,
        item_id: str,
        difficulty_a: Optional[float] = None,
        difficulty_b: Optional[float] = None
    ) -> Optional[MathItem]:
        """Update IRT difficulty parameters."""
        item = await self.get(session, item_id)
        if not item:
            return None
        
        updates = {}
        if difficulty_a is not None:
            updates["difficulty_a"] = difficulty_a
        if difficulty_b is not None:
            updates["difficulty_b"] = difficulty_b
        
        if updates:
            return await self.update(session, db_obj=item, obj_in=updates)
        return item
    
    async def get_items_for_calibration(
        self,
        session: AsyncSession,
        tenant_id: str,
        min_attempts: int = 10
    ) -> List[MathItem]:
        """Get items that need difficulty parameter calibration."""
        # This would typically join with attempts table to find items with enough data
        # For now, return items with default parameters
        statement = select(MathItem).where(
            and_(
                MathItem.tenant_id == tenant_id,
                MathItem.status == "active",
                or_(
                    MathItem.difficulty_a == 1.0,
                    MathItem.difficulty_b == 0.0
                )
            )
        )
        
        result = await session.exec(statement)
        return result.all()


class EnglishItemRepository(BaseRepository[EnglishItem]):
    """English item repository with CEFR and error tag filtering."""
    
    def __init__(self):
        super().__init__(EnglishItem)
    
    async def search_by_error_tags(
        self,
        session: AsyncSession,
        error_tags: List[str],
        tenant_id: str,
        level_cefr: Optional[str] = None,
        topic: Optional[str] = None,
        limit: int = 200
    ) -> List[EnglishItem]:
        """Search English items by error tags."""
        statement = select(EnglishItem).where(
            and_(
                EnglishItem.tenant_id == tenant_id,
                EnglishItem.lang == "en",
                EnglishItem.status == "active",
                EnglishItem.ambiguity_flag == False
            )
        )
        
        # Filter by error tags
        if error_tags:
            tag_conditions = []
            for tag in error_tags:
                tag_conditions.append(EnglishItem.error_tags.contains([tag]))
            statement = statement.where(or_(*tag_conditions))
        
        # Filter by CEFR level
        if level_cefr:
            statement = statement.where(EnglishItem.level_cefr == level_cefr)
        
        # Filter by topic
        if topic:
            statement = statement.where(EnglishItem.topic == topic)
        
        statement = statement.limit(limit)
        result = await session.exec(statement)
        return result.all()
    
    async def get_by_cefr_level(
        self,
        session: AsyncSession,
        level_cefr: str,
        tenant_id: str,
        limit: int = 100
    ) -> List[EnglishItem]:
        """Get English items by CEFR level."""
        statement = select(EnglishItem).where(
            and_(
                EnglishItem.tenant_id == tenant_id,
                EnglishItem.lang == "en",
                EnglishItem.status == "active",
                EnglishItem.level_cefr == level_cefr,
                EnglishItem.ambiguity_flag == False
            )
        ).limit(limit)
        
        result = await session.exec(statement)
        return result.all()
    
    async def get_by_topic(
        self,
        session: AsyncSession,
        topic: str,
        tenant_id: str,
        level_cefr: Optional[str] = None,
        limit: int = 100
    ) -> List[EnglishItem]:
        """Get English items by topic."""
        conditions = [
            EnglishItem.tenant_id == tenant_id,
            EnglishItem.lang == "en",
            EnglishItem.status == "active",
            EnglishItem.topic == topic,
            EnglishItem.ambiguity_flag == False
        ]
        
        if level_cefr:
            conditions.append(EnglishItem.level_cefr == level_cefr)
        
        statement = select(EnglishItem).where(and_(*conditions)).limit(limit)
        result = await session.exec(statement)
        return result.all()
    
    async def mark_ambiguous(
        self,
        session: AsyncSession,
        item_id: str,
        ambiguous: bool = True
    ) -> Optional[EnglishItem]:
        """Mark item as ambiguous or clear ambiguity flag."""
        item = await self.get(session, item_id)
        if not item:
            return None
        
        return await self.update(
            session,
            db_obj=item,
            obj_in={"ambiguity_flag": ambiguous}
        )
    
    async def get_items_needing_review(
        self,
        session: AsyncSession,
        tenant_id: str
    ) -> List[EnglishItem]:
        """Get items that need human review."""
        statement = select(EnglishItem).where(
            and_(
                EnglishItem.tenant_id == tenant_id,
                EnglishItem.status == "draft",
                or_(
                    EnglishItem.review_status == "pending",
                    EnglishItem.review_status.is_(None)
                )
            )
        )
        
        result = await session.exec(statement)
        return result.all()


# Create repository instances
math_item_repository = MathItemRepository()
english_item_repository = EnglishItemRepository()