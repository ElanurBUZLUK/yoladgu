import asyncio
import logging
import time
import json
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
from collections import OrderedDict
import hashlib

logger = logging.getLogger(__name__)


class CacheItem:
    """Cache item with metadata and expiration"""
    
    def __init__(self, key: str, value: Any, ttl_seconds: int = 3600):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0
        self.ttl_seconds = ttl_seconds
        self.size_bytes = self._calculate_size()
    
    def _calculate_size(self) -> int:
        """Calculate approximate size in bytes"""
        try:
            if isinstance(self.value, str):
                return len(self.value.encode('utf-8'))
            elif isinstance(self.value, (dict, list)):
                return len(json.dumps(self.value, default=str).encode('utf-8'))
            elif isinstance(self.value, (int, float)):
                return 8
            else:
                return len(str(self.value).encode('utf-8'))
        except Exception:
            return 100  # Default size
    
    def is_expired(self) -> bool:
        """Check if item is expired"""
        return time.time() - self.created_at > self.ttl_seconds
    
    def access(self):
        """Mark item as accessed"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def get_age(self) -> float:
        """Get item age in seconds"""
        return time.time() - self.created_at
    
    def get_remaining_ttl(self) -> float:
        """Get remaining TTL in seconds"""
        remaining = self.ttl_seconds - self.get_age()
        return max(0, remaining)


class EnhancedCacheService:
    """High-performance cache service with TTL, size management, and analytics"""
    
    def __init__(
        self,
        max_size_mb: int = 100,
        default_ttl_seconds: int = 3600,
        cleanup_interval_seconds: int = 300,
        max_items: int = 10000
    ):
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl_seconds = default_ttl_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.max_items = max_items
        
        # Cache storage
        self.cache: OrderedDict[str, CacheItem] = OrderedDict()
        
        # Performance tracking
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
            "total_size_bytes": 0,
            "created_at": time.time()
        }
        
        # Start cleanup task
        self._start_cleanup_task()
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache"""
        
        try:
            if key in self.cache:
                item = self.cache[key]
                
                # Check expiration
                if item.is_expired():
                    await self.delete(key)
                    self.stats["misses"] += 1
                    return default
                
                # Mark as accessed and move to end (LRU)
                item.access()
                self.cache.move_to_end(key)
                
                self.stats["hits"] += 1
                logger.debug(f"✅ Cache hit for key: {key}")
                return item.value
            else:
                self.stats["misses"] += 1
                logger.debug(f"❌ Cache miss for key: {key}")
                return default
                
        except Exception as e:
            logger.error(f"❌ Error getting from cache: {e}")
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        force: bool = False
    ) -> bool:
        """Set item in cache"""
        
        try:
            # Use default TTL if not specified
            if ttl_seconds is None:
                ttl_seconds = self.default_ttl_seconds
            
            # Create cache item
            item = CacheItem(key, value, ttl_seconds)
            
            # Check if we need to make space
            if not force and not await self._has_space_for_item(item):
                await self._evict_items_for_space(item.size_bytes)
            
            # Add to cache
            if key in self.cache:
                # Update existing item
                old_item = self.cache[key]
                self.stats["total_size_bytes"] -= old_item.size_bytes
                self.cache[key] = item
            else:
                # Add new item
                self.cache[key] = item
            
            # Update size tracking
            self.stats["total_size_bytes"] += item.size_bytes
            self.stats["sets"] += 1
            
            # Move to end (LRU)
            self.cache.move_to_end(key)
            
            logger.debug(f"✅ Cache set for key: {key}, size: {item.size_bytes} bytes")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error setting cache item: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete item from cache"""
        
        try:
            if key in self.cache:
                item = self.cache[key]
                self.stats["total_size_bytes"] -= item.size_bytes
                del self.cache[key]
                self.stats["deletes"] += 1
                logger.debug(f"🗑️ Cache delete for key: {key}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Error deleting cache item: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache items"""
        
        try:
            self.cache.clear()
            self.stats["total_size_bytes"] = 0
            self.stats["deletes"] += len(self.cache)
            logger.info("🧹 Cache cleared")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error clearing cache: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        
        try:
            if key in self.cache:
                item = self.cache[key]
                if not item.is_expired():
                    return True
                else:
                    # Clean up expired item
                    await self.delete(key)
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking cache existence: {e}")
            return False
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple items from cache"""
        
        try:
            results = {}
            for key in keys:
                value = await self.get(key)
                if value is not None:
                    results[key] = value
            
            logger.debug(f"✅ Cache get_many: {len(results)}/{len(keys)} items found")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error getting multiple items: {e}")
            return {}
    
    async def set_many(
        self,
        items: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ) -> Dict[str, bool]:
        """Set multiple items in cache"""
        
        try:
            results = {}
            for key, value in items.items():
                success = await self.set(key, value, ttl_seconds)
                results[key] = success
            
            success_count = sum(results.values())
            logger.debug(f"✅ Cache set_many: {success_count}/{len(items)} items set")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error setting multiple items: {e}")
            return {key: False for key in items.keys()}
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment numeric value in cache"""
        
        try:
            current_value = await self.get(key, 0)
            if isinstance(current_value, (int, float)):
                new_value = current_value + amount
                await self.set(key, new_value)
                return new_value
            return None
            
        except Exception as e:
            logger.error(f"❌ Error incrementing cache value: {e}")
            return None
    
    async def expire(self, key: str, ttl_seconds: int) -> bool:
        """Set expiration for existing key"""
        
        try:
            if key in self.cache:
                item = self.cache[key]
                item.ttl_seconds = ttl_seconds
                item.created_at = time.time()  # Reset creation time
                logger.debug(f"⏰ Set expiration for key: {key}, TTL: {ttl_seconds}s")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Error setting expiration: {e}")
            return False
    
    async def get_ttl(self, key: str) -> Optional[float]:
        """Get remaining TTL for key"""
        
        try:
            if key in self.cache:
                item = self.cache[key]
                return item.get_remaining_ttl()
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting TTL: {e}")
            return None
    
    async def _has_space_for_item(self, item: CacheItem) -> bool:
        """Check if there's space for a new item"""
        
        # Check item count limit
        if len(self.cache) >= self.max_items:
            return False
        
        # Check size limit
        if self.stats["total_size_bytes"] + item.size_bytes > self.max_size_bytes:
            return False
        
        return True
    
    async def _evict_items_for_space(self, required_bytes: int) -> int:
        """Evict items to make space for new item"""
        
        try:
            evicted_count = 0
            evicted_bytes = 0
            
            # Evict expired items first
            expired_keys = [key for key, item in self.cache.items() if item.is_expired()]
            for key in expired_keys:
                await self.delete(key)
                evicted_count += 1
            
            # If still need space, evict by LRU
            while (self.stats["total_size_bytes"] + required_bytes > self.max_size_bytes and 
                   len(self.cache) > 0):
                
                # Remove oldest item (LRU)
                oldest_key = next(iter(self.cache))
                oldest_item = self.cache[oldest_key]
                
                await self.delete(oldest_key)
                evicted_count += 1
                evicted_bytes += oldest_item.size_bytes
                
                # Safety check
                if evicted_count > len(self.cache) * 2:
                    logger.warning("⚠️ Excessive evictions, stopping")
                    break
            
            if evicted_count > 0:
                logger.info(f"🗑️ Evicted {evicted_count} items ({evicted_bytes} bytes) to make space")
                self.stats["evictions"] += evicted_count
            
            return evicted_count
            
        except Exception as e:
            logger.error(f"❌ Error evicting items: {e}")
            return 0
    
    async def _cleanup_expired_items(self):
        """Remove expired items from cache"""
        
        try:
            expired_keys = [key for key, item in self.cache.items() if item.is_expired()]
            
            for key in expired_keys:
                await self.delete(key)
            
            if expired_keys:
                logger.debug(f"🧹 Cleaned up {len(expired_keys)} expired items")
                
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval_seconds)
                    await self._cleanup_expired_items()
                except Exception as e:
                    logger.error(f"❌ Error in cleanup loop: {e}")
        
        # Start cleanup task
        asyncio.create_task(cleanup_loop())
        logger.info(f"🔄 Started cache cleanup task (interval: {self.cleanup_interval_seconds}s)")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        
        try:
            # Calculate hit rate
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
            
            # Calculate memory usage
            memory_usage_mb = self.stats["total_size_bytes"] / (1024 * 1024)
            memory_usage_percent = (memory_usage_mb / self.max_size_mb) * 100
            
            # Get cache item distribution
            item_distribution = self._get_item_distribution()
            
            return {
                "cache_info": {
                    "max_size_mb": self.max_size_mb,
                    "current_size_mb": round(memory_usage_mb, 2),
                    "memory_usage_percent": round(memory_usage_percent, 2),
                    "max_items": self.max_items,
                    "current_items": len(self.cache),
                    "item_usage_percent": (len(self.cache) / self.max_items) * 100
                },
                "performance": {
                    "hits": self.stats["hits"],
                    "misses": self.stats["misses"],
                    "hit_rate": round(hit_rate, 4),
                    "sets": self.stats["sets"],
                    "deletes": self.stats["deletes"],
                    "evictions": self.stats["evictions"]
                },
                "item_distribution": item_distribution,
                "uptime_seconds": time.time() - self.stats["created_at"],
                "last_cleanup": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting cache stats: {e}")
            return {}
    
    def _get_item_distribution(self) -> Dict[str, Any]:
        """Get distribution of cache items by age and size"""
        
        try:
            if not self.cache:
                return {"age_distribution": {}, "size_distribution": {}}
            
            # Age distribution
            ages = [item.get_age() for item in self.cache.values()]
            age_distribution = {
                "0-1h": len([age for age in ages if age < 3600]),
                "1-6h": len([age for age in ages if 3600 <= age < 21600]),
                "6-24h": len([age for age in ages if 21600 <= age < 86400]),
                "24h+": len([age for age in ages if age >= 86400])
            }
            
            # Size distribution
            sizes = [item.size_bytes for item in self.cache.values()]
            size_distribution = {
                "0-1KB": len([size for size in sizes if size < 1024]),
                "1-10KB": len([size for size in sizes if 1024 <= size < 10240]),
                "10-100KB": len([size for size in sizes if 10240 <= size < 102400]),
                "100KB+": len([size for size in sizes if size >= 102400])
            }
            
            return {
                "age_distribution": age_distribution,
                "size_distribution": size_distribution
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating item distribution: {e}")
            return {}
    
    async def optimize(self) -> Dict[str, Any]:
        """Optimize cache performance"""
        
        try:
            optimization_results = {
                "items_removed": 0,
                "memory_freed_mb": 0,
                "recommendations": []
            }
            
            # Remove expired items
            expired_keys = [key for key, item in self.cache.items() if item.is_expired()]
            for key in expired_keys:
                await self.delete(key)
                optimization_results["items_removed"] += 1
            
            # Calculate memory freed
            memory_freed_mb = self.stats["total_size_bytes"] / (1024 * 1024)
            optimization_results["memory_freed_mb"] = round(memory_freed_mb, 2)
            
            # Generate recommendations
            stats = await self.get_stats()
            
            if stats.get("performance", {}).get("hit_rate", 0) < 0.5:
                optimization_results["recommendations"].append("Low hit rate - consider increasing cache size or TTL")
            
            if stats.get("cache_info", {}).get("memory_usage_percent", 0) > 80:
                optimization_results["recommendations"].append("High memory usage - consider reducing cache size or TTL")
            
            if stats.get("performance", {}).get("evictions", 0) > 100:
                optimization_results["recommendations"].append("High eviction rate - consider increasing cache size")
            
            logger.info(f"🔧 Cache optimization completed: {optimization_results['items_removed']} items removed")
            return optimization_results
            
        except Exception as e:
            logger.error(f"❌ Error during cache optimization: {e}")
            return {"error": str(e)}


# Global instance
enhanced_cache_service = EnhancedCacheService()
