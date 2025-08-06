"""
Redis Service
Cache ve session yönetimi için Redis servisi
"""

import structlog
import json
import redis
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from app.core.config import settings

logger = structlog.get_logger()


class RedisService:
    """Redis cache ve session servisi"""
    
    def __init__(self):
        self.client = None
        self.initialized = False
        
    async def initialize(self):
        """Redis servisini başlat"""
        try:
            # Redis client'ı oluştur
            self.client = redis.Redis.from_url(
                settings.redis_url or "redis://localhost:6379/0",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Bağlantıyı test et
            self.client.ping()
            
            self.initialized = True
            logger.info("redis_service_initialized")
            
        except Exception as e:
            logger.error("redis_service_initialization_error", error=str(e))
            raise
    
    async def set_cache(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Cache'e değer kaydet"""
        try:
            if not self.initialized:
                await self.initialize()
            
            # JSON'a çevir
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = str(value)
            
            result = self.client.setex(key, ttl, serialized_value)
            logger.debug("cache_set", key=key, ttl=ttl)
            return result
            
        except Exception as e:
            logger.error("set_cache_error", error=str(e), key=key)
            return False
    
    async def get_cache(self, key: str) -> Optional[Any]:
        """Cache'den değer al"""
        try:
            if not self.initialized:
                await self.initialize()
            
            value = self.client.get(key)
            
            if value is None:
                return None
            
            # JSON'dan çevir
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
                
        except Exception as e:
            logger.error("get_cache_error", error=str(e), key=key)
            return None
    
    async def delete_cache(self, key: str) -> bool:
        """Cache'den değer sil"""
        try:
            if not self.initialized:
                await self.initialize()
            
            result = self.client.delete(key)
            logger.debug("cache_deleted", key=key)
            return result > 0
            
        except Exception as e:
            logger.error("delete_cache_error", error=str(e), key=key)
            return False
    
    async def set_session(self, session_id: str, user_data: Dict[str, Any], ttl: int = 86400) -> bool:
        """Session kaydet"""
        try:
            if not self.initialized:
                await self.initialize()
            
            session_data = {
                "user_data": user_data,
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat()
            }
            
            key = f"session:{session_id}"
            result = await self.set_cache(key, session_data, ttl)
            
            logger.info("session_set", session_id=session_id, ttl=ttl)
            return result
            
        except Exception as e:
            logger.error("set_session_error", error=str(e), session_id=session_id)
            return False
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Session getir"""
        try:
            if not self.initialized:
                await self.initialize()
            
            key = f"session:{session_id}"
            session_data = await self.get_cache(key)
            
            if session_data:
                # Last activity'yi güncelle
                session_data["last_activity"] = datetime.now().isoformat()
                await self.set_cache(key, session_data, 86400)  # 24 saat
            
            return session_data
            
        except Exception as e:
            logger.error("get_session_error", error=str(e), session_id=session_id)
            return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Session sil"""
        try:
            if not self.initialized:
                await self.initialize()
            
            key = f"session:{session_id}"
            result = await self.delete_cache(key)
            
            logger.info("session_deleted", session_id=session_id)
            return result
            
        except Exception as e:
            logger.error("delete_session_error", error=str(e), session_id=session_id)
            return False
    
    async def set_rate_limit(self, key: str, limit: int, window: int) -> bool:
        """Rate limit ayarla"""
        try:
            if not self.initialized:
                await self.initialize()
            
            current_count = self.client.get(key)
            if current_count is None:
                self.client.setex(key, window, 1)
                return True
            else:
                count = int(current_count)
                if count < limit:
                    self.client.incr(key)
                    return True
                else:
                    return False
                    
        except Exception as e:
            logger.error("set_rate_limit_error", error=str(e), key=key)
            return False
    
    async def get_rate_limit(self, key: str) -> Dict[str, Any]:
        """Rate limit bilgisi getir"""
        try:
            if not self.initialized:
                await self.initialize()
            
            current_count = self.client.get(key)
            ttl = self.client.ttl(key)
            
            return {
                "current_count": int(current_count) if current_count else 0,
                "remaining_ttl": ttl if ttl > 0 else 0
            }
            
        except Exception as e:
            logger.error("get_rate_limit_error", error=str(e), key=key)
            return {"current_count": 0, "remaining_ttl": 0}
    
    async def set_user_activity(self, user_id: int, activity_data: Dict[str, Any]) -> bool:
        """Kullanıcı aktivitesi kaydet"""
        try:
            if not self.initialized:
                await self.initialize()
            
            key = f"user_activity:{user_id}"
            timestamp = datetime.now().isoformat()
            
            # Mevcut aktiviteleri al
            activities = await self.get_cache(key) or []
            activities.append({
                "timestamp": timestamp,
                "data": activity_data
            })
            
            # Son 100 aktiviteyi tut
            if len(activities) > 100:
                activities = activities[-100:]
            
            result = await self.set_cache(key, activities, 86400)  # 24 saat
            
            logger.debug("user_activity_set", user_id=user_id)
            return result
            
        except Exception as e:
            logger.error("set_user_activity_error", error=str(e), user_id=user_id)
            return False
    
    async def get_user_activities(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Kullanıcı aktivitelerini getir"""
        try:
            if not self.initialized:
                await self.initialize()
            
            key = f"user_activity:{user_id}"
            activities = await self.get_cache(key) or []
            
            # Son aktiviteleri döndür
            return activities[-limit:]
            
        except Exception as e:
            logger.error("get_user_activities_error", error=str(e), user_id=user_id)
            return []
    
    async def set_embedding_cache(self, text: str, embedding: List[float], ttl: int = 3600) -> bool:
        """Embedding cache'e kaydet"""
        try:
            if not self.initialized:
                await self.initialize()
            
            # Text hash'ini oluştur
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            key = f"embedding:{text_hash}"
            
            result = await self.set_cache(key, embedding, ttl)
            
            logger.debug("embedding_cache_set", text_hash=text_hash)
            return result
            
        except Exception as e:
            logger.error("set_embedding_cache_error", error=str(e))
            return False
    
    async def get_embedding_cache(self, text: str) -> Optional[List[float]]:
        """Embedding cache'den getir"""
        try:
            if not self.initialized:
                await self.initialize()
            
            # Text hash'ini oluştur
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            key = f"embedding:{text_hash}"
            
            embedding = await self.get_cache(key)
            
            if embedding:
                logger.debug("embedding_cache_hit", text_hash=text_hash)
            else:
                logger.debug("embedding_cache_miss", text_hash=text_hash)
            
            return embedding
            
        except Exception as e:
            logger.error("get_embedding_cache_error", error=str(e))
            return None
    
    async def get_redis_stats(self) -> Dict[str, Any]:
        """Redis istatistiklerini getir"""
        try:
            if not self.initialized:
                await self.initialize()
            
            info = self.client.info()
            
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": info.get("keyspace_hits", 0) / max(info.get("keyspace_misses", 1), 1),
                "initialized": self.initialized
            }
            
        except Exception as e:
            logger.error("get_redis_stats_error", error=str(e))
            return {
                "connected_clients": 0,
                "used_memory_human": "0B",
                "total_commands_processed": 0,
                "keyspace_hits": 0,
                "keyspace_misses": 0,
                "hit_rate": 0,
                "initialized": self.initialized
            }
    
    async def is_healthy(self) -> bool:
        """Redis sağlık kontrolü"""
        try:
            if not self.initialized or not self.client:
                return False
            
            self.client.ping()
            return True
            
        except Exception as e:
            logger.error("redis_health_check_error", error=str(e))
            return False
    
    async def cleanup(self):
        """Redis servisini temizle"""
        try:
            if self.client:
                self.client.close()
                self.initialized = False
            
            logger.info("redis_service_cleanup_completed")
            
        except Exception as e:
            logger.error("redis_cleanup_error", error=str(e))


# Global instance
redis_service = RedisService() 