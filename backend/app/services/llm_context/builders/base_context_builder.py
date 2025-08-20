from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from ..schemas.base_context import BaseContext


class BaseContextBuilder(ABC):
    """Temel context builder sınıfı - tüm builder'lar için ortak özellikler"""
    
    def __init__(self):
        self.logger = None  # Logger instance'ı sonradan set edilecek
    
    @abstractmethod
    async def build(
        self, 
        session: AsyncSession, 
        **kwargs
    ) -> BaseContext:
        """Context'i oluştur"""
        pass
    
    async def _get_user_context(
        self, 
        session: AsyncSession, 
        user_id: str
    ) -> Dict[str, Any]:
        """Kullanıcı bağlamını al"""
        # Bu metod alt sınıflar tarafından override edilebilir
        return {
            "user_id": user_id,
            "user_level": None,
            "user_preferences": {}
        }
    
    async def _get_knowledge_context(
        self, 
        session: AsyncSession, 
        **kwargs
    ) -> Dict[str, Any]:
        """Bilgi bağlamını al"""
        # Bu metod alt sınıflar tarafından override edilebilir
        return {
            "grammar_rules": [],
            "vocabulary_context": None,
            "topic_context": None
        }
    
    async def _get_output_format_context(
        self, 
        **kwargs
    ) -> Dict[str, Any]:
        """Çıktı format bağlamını al"""
        # Bu metod alt sınıflar tarafından override edilebilir
        return {
            "output_schema": {},
            "format_instructions": ""
        }
