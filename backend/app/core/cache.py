import redis.asyncio as redis
from typing import Optional, Any
import json
import pickle
from app.core.config import settings


class CacheService:
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None

    async def connect(self):
        """Redis bağlantısını başlat"""
        self.redis_client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=False  # Binary data için False
        )
        # Bağlantıyı test et
        await self.redis_client.ping()
        print("✅ Redis connected")

    async def disconnect(self):
        """Redis bağlantısını kapat"""
        if self.redis_client:
            await self.redis_client.close()
            print("❌ Redis disconnected")

    async def get(self, key: str) -> Optional[Any]:
        """Cache'den veri getir"""
        if not self.redis_client:
            return None
        
        try:
            data = await self.redis_client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            print(f"Cache get error: {e}")
            return None

    async def set(self, key: str, value: Any, expire: int = 3600) -> bool:
        """Cache'e veri kaydet"""
        if not self.redis_client:
            return False
        
        try:
            serialized_data = pickle.dumps(value)
            await self.redis_client.setex(key, expire, serialized_data)
            return True
        except Exception as e:
            print(f"Cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Cache'den veri sil"""
        if not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            print(f"Cache delete error: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Key'in varlığını kontrol et"""
        if not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.exists(key)
            return result > 0
        except Exception as e:
            print(f"Cache exists error: {e}")
            return False

    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Counter'ı artır"""
        if not self.redis_client:
            return None
        
        try:
            result = await self.redis_client.incrby(key, amount)
            return result
        except Exception as e:
            print(f"Cache increment error: {e}")
            return None

    async def get_json(self, key: str) -> Optional[dict]:
        """JSON formatında veri getir"""
        if not self.redis_client:
            return None
        
        try:
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data.decode('utf-8'))
            return None
        except Exception as e:
            print(f"Cache get_json error: {e}")
            return None

    async def set_json(self, key: str, value: dict, expire: int = 3600) -> bool:
        """JSON formatında veri kaydet"""
        if not self.redis_client:
            return False
        
        try:
            json_data = json.dumps(value)
            await self.redis_client.setex(key, expire, json_data)
            return True
        except Exception as e:
            print(f"Cache set_json error: {e}")
            return False


# Global cache instance
cache_service = CacheService()