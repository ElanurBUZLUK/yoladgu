from typing import TypeVar, Generic, Type, Optional, List, Any, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import DeclarativeBase
from pydantic import BaseModel

from app.database import database_manager

ModelType = TypeVar("ModelType", bound=DeclarativeBase)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Base repository class - Tüm repository'ler için temel CRUD operasyonları"""
    
    def __init__(self, model: Type[ModelType]):
        self.model = model
    
    async def get(self, db: AsyncSession, id: Any) -> Optional[ModelType]:
        """ID ile tek kayıt getir"""
        result = await db.execute(select(self.model).where(self.model.id == id))
        return result.scalar_one_or_none()
    
    async def get_multi(
        self, 
        db: AsyncSession, 
        skip: int = 0, 
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ModelType]:
        """Çoklu kayıt getir"""
        query = select(self.model)
        
        # Filtreleri uygula
        if filters:
            for field, value in filters.items():
                if hasattr(self.model, field):
                    query = query.where(getattr(self.model, field) == value)
        
        query = query.offset(skip).limit(limit)
        result = await db.execute(query)
        return result.scalars().all()
    
    async def create(self, db: AsyncSession, obj_in: CreateSchemaType) -> ModelType:
        """Yeni kayıt oluştur"""
        obj_data = obj_in.dict()
        db_obj = self.model(**obj_data)
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj
    
    async def update(
        self, 
        db: AsyncSession, 
        db_obj: ModelType, 
        obj_in: UpdateSchemaType
    ) -> ModelType:
        """Kayıt güncelle"""
        obj_data = obj_in.dict(exclude_unset=True)
        
        for field, value in obj_data.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        
        await db.commit()
        await db.refresh(db_obj)
        return db_obj
    
    async def delete(self, db: AsyncSession, id: Any) -> bool:
        """Kayıt sil"""
        result = await db.execute(delete(self.model).where(self.model.id == id))
        await db.commit()
        return result.rowcount > 0
    
    async def exists(self, db: AsyncSession, id: Any) -> bool:
        """Kayıt var mı kontrol et"""
        result = await db.execute(select(self.model).where(self.model.id == id))
        return result.scalar_one_or_none() is not None
    
    async def count(self, db: AsyncSession, filters: Optional[Dict[str, Any]] = None) -> int:
        """Kayıt sayısını getir"""
        from sqlalchemy import func
        
        query = select(func.count(self.model.id))
        
        if filters:
            for field, value in filters.items():
                if hasattr(self.model, field):
                    query = query.where(getattr(self.model, field) == value)
        
        result = await db.execute(query)
        return result.scalar()
    
    async def get_by_field(
        self, 
        db: AsyncSession, 
        field: str, 
        value: Any
    ) -> Optional[ModelType]:
        """Belirli bir alan değeri ile kayıt getir"""
        if not hasattr(self.model, field):
            raise ValueError(f"Field '{field}' does not exist in model")
        
        result = await db.execute(
            select(self.model).where(getattr(self.model, field) == value)
        )
        return result.scalar_one_or_none()
    
    async def get_multi_by_field(
        self, 
        db: AsyncSession, 
        field: str, 
        value: Any,
        skip: int = 0,
        limit: int = 100
    ) -> List[ModelType]:
        """Belirli bir alan değeri ile çoklu kayıt getir"""
        if not hasattr(self.model, field):
            raise ValueError(f"Field '{field}' does not exist in model")
        
        result = await db.execute(
            select(self.model)
            .where(getattr(self.model, field) == value)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def bulk_create(
        self, 
        db: AsyncSession, 
        objs_in: List[CreateSchemaType]
    ) -> List[ModelType]:
        """Toplu kayıt oluştur"""
        db_objs = []
        for obj_in in objs_in:
            obj_data = obj_in.dict()
            db_obj = self.model(**obj_data)
            db_objs.append(db_obj)
        
        db.add_all(db_objs)
        await db.commit()
        
        # Refresh all objects
        for db_obj in db_objs:
            await db.refresh(db_obj)
        
        return db_objs
    
    async def bulk_update(
        self, 
        db: AsyncSession, 
        ids: List[Any], 
        update_data: Dict[str, Any]
    ) -> int:
        """Toplu güncelleme"""
        result = await db.execute(
            update(self.model)
            .where(self.model.id.in_(ids))
            .values(**update_data)
        )
        await db.commit()
        return result.rowcount
    
    async def bulk_delete(self, db: AsyncSession, ids: List[Any]) -> int:
        """Toplu silme"""
        result = await db.execute(delete(self.model).where(self.model.id.in_(ids)))
        await db.commit()
        return result.rowcount
    
    async def search(
        self, 
        db: AsyncSession, 
        search_term: str,
        search_fields: List[str],
        skip: int = 0,
        limit: int = 100
    ) -> List[ModelType]:
        """Metin arama"""
        from sqlalchemy import or_
        
        # Arama koşullarını oluştur
        search_conditions = []
        for field in search_fields:
            if hasattr(self.model, field):
                search_conditions.append(
                    getattr(self.model, field).ilike(f"%{search_term}%")
                )
        
        if not search_conditions:
            return []
        
        result = await db.execute(
            select(self.model)
            .where(or_(*search_conditions))
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_with_relations(
        self, 
        db: AsyncSession, 
        id: Any,
        relations: List[str]
    ) -> Optional[ModelType]:
        """İlişkili kayıtları da getir"""
        from sqlalchemy.orm import joinedload
        
        query = select(self.model)
        
        # İlişkileri yükle
        for relation in relations:
            if hasattr(self.model, relation):
                query = query.options(joinedload(getattr(self.model, relation)))
        
        query = query.where(self.model.id == id)
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_paginated(
        self,
        db: AsyncSession,
        page: int = 1,
        page_size: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        order_desc: bool = False
    ) -> Dict[str, Any]:
        """Sayfalanmış sonuçlar getir"""
        skip = (page - 1) * page_size
        
        # Ana sorgu
        query = select(self.model)
        
        # Filtreleri uygula
        if filters:
            for field, value in filters.items():
                if hasattr(self.model, field):
                    query = query.where(getattr(self.model, field) == value)
        
        # Sıralama
        if order_by and hasattr(self.model, order_by):
            order_field = getattr(self.model, order_by)
            if order_desc:
                query = query.order_by(order_field.desc())
            else:
                query = query.order_by(order_field.asc())
        
        # Toplam sayı
        count_query = select(self.model)
        if filters:
            for field, value in filters.items():
                if hasattr(self.model, field):
                    count_query = count_query.where(getattr(self.model, field) == value)
        
        total_result = await db.execute(count_query)
        total = len(total_result.scalars().all())
        
        # Sayfalanmış sonuçlar
        query = query.offset(skip).limit(page_size)
        result = await db.execute(query)
        items = result.scalars().all()
        
        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": (total + page_size - 1) // page_size,
            "has_next": page * page_size < total,
            "has_prev": page > 1
        }
