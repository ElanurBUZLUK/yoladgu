"""
Base repository with common CRUD operations.
"""

from typing import Generic, TypeVar, Type, Optional, List, Dict, Any
from sqlmodel import SQLModel, select, Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, and_, or_

ModelType = TypeVar("ModelType", bound=SQLModel)


class BaseRepository(Generic[ModelType]):
    """Base repository with common CRUD operations."""
    
    def __init__(self, model: Type[ModelType]):
        self.model = model
    
    async def create(self, session: AsyncSession, *, obj_in: Dict[str, Any]) -> ModelType:
        """Create a new record."""
        db_obj = self.model(**obj_in)
        session.add(db_obj)
        await session.commit()
        await session.refresh(db_obj)
        return db_obj
    
    async def get(self, session: AsyncSession, id: str) -> Optional[ModelType]:
        """Get record by ID."""
        statement = select(self.model).where(self.model.id == id)
        result = await session.exec(statement)
        return result.first()
    
    async def get_multi(
        self, 
        session: AsyncSession, 
        *, 
        skip: int = 0, 
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ModelType]:
        """Get multiple records with pagination and filtering."""
        statement = select(self.model)
        
        # Apply filters
        if filters:
            conditions = []
            for key, value in filters.items():
                if hasattr(self.model, key):
                    if isinstance(value, list):
                        conditions.append(getattr(self.model, key).in_(value))
                    else:
                        conditions.append(getattr(self.model, key) == value)
            if conditions:
                statement = statement.where(and_(*conditions))
        
        statement = statement.offset(skip).limit(limit)
        result = await session.exec(statement)
        return result.all()
    
    async def update(
        self, 
        session: AsyncSession, 
        *, 
        db_obj: ModelType, 
        obj_in: Dict[str, Any]
    ) -> ModelType:
        """Update a record."""
        for field, value in obj_in.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        
        session.add(db_obj)
        await session.commit()
        await session.refresh(db_obj)
        return db_obj
    
    async def delete(self, session: AsyncSession, *, id: str) -> Optional[ModelType]:
        """Delete a record by ID."""
        db_obj = await self.get(session, id)
        if db_obj:
            await session.delete(db_obj)
            await session.commit()
        return db_obj
    
    async def count(
        self, 
        session: AsyncSession, 
        *, 
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Count records with optional filtering."""
        statement = select(func.count(self.model.id))
        
        if filters:
            conditions = []
            for key, value in filters.items():
                if hasattr(self.model, key):
                    if isinstance(value, list):
                        conditions.append(getattr(self.model, key).in_(value))
                    else:
                        conditions.append(getattr(self.model, key) == value)
            if conditions:
                statement = statement.where(and_(*conditions))
        
        result = await session.exec(statement)
        return result.one()
    
    async def exists(self, session: AsyncSession, *, id: str) -> bool:
        """Check if record exists."""
        statement = select(self.model.id).where(self.model.id == id)
        result = await session.exec(statement)
        return result.first() is not None