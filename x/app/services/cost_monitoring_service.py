from typing import Dict, Any, List, Optional, Union
from enum import Enum
import asyncio
import logging
from datetime import datetime, timedelta
import json

from app.core.config import settings
from app.core.cache import cache_service
from app.models.user import User

logger = logging.getLogger(__name__)


class LimitType(str, Enum):
    """Limit types for cost monitoring"""
    USER_MONTHLY = "user_monthly"
    ORGANIZATION_MONTHLY = "organization_monthly"
    ENDPOINT_DAILY = "endpoint_daily"
    GLOBAL_DAILY = "global_daily"


class DegradationMode(str, Enum):
    """Degradation modes when limits are exceeded"""
    SMALLER_MODEL = "smaller_model"
    RAG_ONLY = "rag_only"
    TEMPLATE_ONLY = "template_only"
    BLOCKED = "blocked"


class CostLimit:
    """Cost limit configuration"""
    
    def __init__(
        self,
        limit_type: LimitType,
        entity_id: str,
        token_limit: int,
        cost_limit: float,
        degradation_mode: DegradationMode = DegradationMode.SMALLER_MODEL,
        reset_period: str = "monthly"  # monthly, daily, weekly
    ):
        self.limit_type = limit_type
        self.entity_id = entity_id
        self.token_limit = token_limit
        self.cost_limit = cost_limit
        self.degradation_mode = degradation_mode
        self.reset_period = reset_period


class CostMonitoringService:
    """Comprehensive cost monitoring and limit management"""
    
    def __init__(self):
        self.cache_prefix = "cost_monitoring"
        self.default_limits = self._initialize_default_limits()
        
    def _initialize_default_limits(self) -> Dict[str, CostLimit]:
        """Initialize default cost limits"""
        return {
            # User monthly limits
            "user_default": CostLimit(
                limit_type=LimitType.USER_MONTHLY,
                entity_id="default",
                token_limit=100000,  # 100K tokens per month
                cost_limit=50.0,     # $50 per month
                degradation_mode=DegradationMode.SMALLER_MODEL
            ),
            
            # Organization monthly limits
            "org_default": CostLimit(
                limit_type=LimitType.ORGANIZATION_MONTHLY,
                entity_id="default",
                token_limit=1000000,  # 1M tokens per month
                cost_limit=500.0,     # $500 per month
                degradation_mode=DegradationMode.RAG_ONLY
            ),
            
            # Endpoint daily limits
            "endpoint_math_rag": CostLimit(
                limit_type=LimitType.ENDPOINT_DAILY,
                entity_id="math_rag",
                token_limit=10000,    # 10K tokens per day
                cost_limit=5.0,       # $5 per day
                degradation_mode=DegradationMode.TEMPLATE_ONLY
            ),
            
            "endpoint_english_rag": CostLimit(
                limit_type=LimitType.ENDPOINT_DAILY,
                entity_id="english_rag",
                token_limit=10000,    # 10K tokens per day
                cost_limit=5.0,       # $5 per day
                degradation_mode=DegradationMode.TEMPLATE_ONLY
            ),
            
            # Global daily limit
            "global_daily": CostLimit(
                limit_type=LimitType.GLOBAL_DAILY,
                entity_id="global",
                token_limit=100000,   # 100K tokens per day
                cost_limit=100.0,     # $100 per day
                degradation_mode=DegradationMode.BLOCKED
            )
        }
    
    async def check_limits(
        self,
        user_id: str,
        organization_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        estimated_tokens: int = 1000,
        estimated_cost: float = 0.01
    ) -> Dict[str, Any]:
        """Check all applicable limits"""
        
        results = {
            "allowed": True,
            "degradation_mode": None,
            "limits_checked": [],
            "warnings": []
        }
        
        # Check user monthly limit
        user_limit = await self._get_user_limit(user_id)
        user_usage = await self._get_user_usage(user_id)
        
        if user_usage["tokens"] + estimated_tokens > user_limit.token_limit:
            results["allowed"] = False
            results["degradation_mode"] = user_limit.degradation_mode.value
            results["warnings"].append(f"User monthly token limit exceeded: {user_usage['tokens']}/{user_limit.token_limit}")
        
        if user_usage["cost"] + estimated_cost > user_limit.cost_limit:
            results["allowed"] = False
            results["degradation_mode"] = user_limit.degradation_mode.value
            results["warnings"].append(f"User monthly cost limit exceeded: {user_usage['cost']:.2f}/{user_limit.cost_limit:.2f}")
        
        results["limits_checked"].append({
            "type": "user_monthly",
            "current_tokens": user_usage["tokens"],
            "current_cost": user_usage["cost"],
            "limit_tokens": user_limit.token_limit,
            "limit_cost": user_limit.cost_limit
        })
        
        # Check organization limit if provided
        if organization_id:
            org_limit = await self._get_organization_limit(organization_id)
            org_usage = await self._get_organization_usage(organization_id)
            
            if org_usage["tokens"] + estimated_tokens > org_limit.token_limit:
                results["allowed"] = False
                results["degradation_mode"] = org_limit.degradation_mode.value
                results["warnings"].append(f"Organization monthly token limit exceeded: {org_usage['tokens']}/{org_limit.token_limit}")
            
            if org_usage["cost"] + estimated_cost > org_limit.cost_limit:
                results["allowed"] = False
                results["degradation_mode"] = org_limit.degradation_mode.value
                results["warnings"].append(f"Organization monthly cost limit exceeded: {org_usage['cost']:.2f}/{org_limit.cost_limit:.2f}")
            
            results["limits_checked"].append({
                "type": "organization_monthly",
                "current_tokens": org_usage["tokens"],
                "current_cost": org_usage["cost"],
                "limit_tokens": org_limit.token_limit,
                "limit_cost": org_limit.cost_limit
            })
        
        # Check endpoint limit if provided
        if endpoint:
            endpoint_limit = await self._get_endpoint_limit(endpoint)
            endpoint_usage = await self._get_endpoint_usage(endpoint)
            
            if endpoint_usage["tokens"] + estimated_tokens > endpoint_limit.token_limit:
                results["allowed"] = False
                results["degradation_mode"] = endpoint_limit.degradation_mode.value
                results["warnings"].append(f"Endpoint daily token limit exceeded: {endpoint_usage['tokens']}/{endpoint_limit.token_limit}")
            
            if endpoint_usage["cost"] + estimated_cost > endpoint_limit.cost_limit:
                results["allowed"] = False
                results["degradation_mode"] = endpoint_limit.degradation_mode.value
                results["warnings"].append(f"Endpoint daily cost limit exceeded: {endpoint_usage['cost']:.2f}/{endpoint_limit.cost_limit:.2f}")
            
            results["limits_checked"].append({
                "type": "endpoint_daily",
                "current_tokens": endpoint_usage["tokens"],
                "current_cost": endpoint_usage["cost"],
                "limit_tokens": endpoint_limit.token_limit,
                "limit_cost": endpoint_limit.cost_limit
            })
        
        # Check global daily limit
        global_limit = await self._get_global_limit()
        global_usage = await self._get_global_usage()
        
        if global_usage["tokens"] + estimated_tokens > global_limit.token_limit:
            results["allowed"] = False
            results["degradation_mode"] = global_limit.degradation_mode.value
            results["warnings"].append(f"Global daily token limit exceeded: {global_usage['tokens']}/{global_limit.token_limit}")
        
        if global_usage["cost"] + estimated_cost > global_limit.cost_limit:
            results["allowed"] = False
            results["degradation_mode"] = global_limit.degradation_mode.value
            results["warnings"].append(f"Global daily cost limit exceeded: {global_usage['cost']:.2f}/{global_limit.cost_limit:.2f}")
        
        results["limits_checked"].append({
            "type": "global_daily",
            "current_tokens": global_usage["tokens"],
            "current_cost": global_usage["cost"],
            "limit_tokens": global_limit.token_limit,
            "limit_cost": global_limit.cost_limit
        })
        
        return results
    
    async def record_usage(
        self,
        user_id: str,
        tokens_used: int,
        cost: float,
        endpoint: Optional[str] = None,
        organization_id: Optional[str] = None
    ):
        """Record usage for all applicable entities"""
        
        # Record user usage
        await self._add_user_usage(user_id, tokens_used, cost)
        
        # Record organization usage if provided
        if organization_id:
            await self._add_organization_usage(organization_id, tokens_used, cost)
        
        # Record endpoint usage if provided
        if endpoint:
            await self._add_endpoint_usage(endpoint, tokens_used, cost)
        
        # Record global usage
        await self._add_global_usage(tokens_used, cost)
        
        logger.info(f"Recorded usage: user={user_id}, tokens={tokens_used}, cost=${cost:.4f}")
    
    async def _get_user_limit(self, user_id: str) -> CostLimit:
        """Get user-specific limit or default"""
        # In a real implementation, you'd fetch from database
        # For now, return default
        return self.default_limits["user_default"]
    
    async def _get_organization_limit(self, organization_id: str) -> CostLimit:
        """Get organization-specific limit or default"""
        return self.default_limits["org_default"]
    
    async def _get_endpoint_limit(self, endpoint: str) -> CostLimit:
        """Get endpoint-specific limit or default"""
        return self.default_limits.get(f"endpoint_{endpoint}", self.default_limits["endpoint_math_rag"])
    
    async def _get_global_limit(self) -> CostLimit:
        """Get global limit"""
        return self.default_limits["global_daily"]
    
    async def _get_user_usage(self, user_id: str) -> Dict[str, Union[int, float]]:
        """Get current user usage for the month"""
        current_month = datetime.now().strftime("%Y-%m")
        cache_key = f"{self.cache_prefix}:user:{user_id}:{current_month}"
        
        usage_data = await cache_service.get(cache_key)
        if usage_data:
            return json.loads(usage_data)
        
        return {"tokens": 0, "cost": 0.0}
    
    async def _get_organization_usage(self, organization_id: str) -> Dict[str, Union[int, float]]:
        """Get current organization usage for the month"""
        current_month = datetime.now().strftime("%Y-%m")
        cache_key = f"{self.cache_prefix}:org:{organization_id}:{current_month}"
        
        usage_data = await cache_service.get(cache_key)
        if usage_data:
            return json.loads(usage_data)
        
        return {"tokens": 0, "cost": 0.0}
    
    async def _get_endpoint_usage(self, endpoint: str) -> Dict[str, Union[int, float]]:
        """Get current endpoint usage for the day"""
        current_day = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"{self.cache_prefix}:endpoint:{endpoint}:{current_day}"
        
        usage_data = await cache_service.get(cache_key)
        if usage_data:
            return json.loads(usage_data)
        
        return {"tokens": 0, "cost": 0.0}
    
    async def _get_global_usage(self) -> Dict[str, Union[int, float]]:
        """Get current global usage for the day"""
        current_day = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"{self.cache_prefix}:global:{current_day}"
        
        usage_data = await cache_service.get(cache_key)
        if usage_data:
            return json.loads(usage_data)
        
        return {"tokens": 0, "cost": 0.0}
    
    async def _add_user_usage(self, user_id: str, tokens: int, cost: float):
        """Add usage to user monthly total"""
        current_month = datetime.now().strftime("%Y-%m")
        cache_key = f"{self.cache_prefix}:user:{user_id}:{current_month}"
        
        current_usage = await self._get_user_usage(user_id)
        new_usage = {
            "tokens": current_usage["tokens"] + tokens,
            "cost": current_usage["cost"] + cost
        }
        
        # Cache until end of month
        days_until_month_end = (datetime.now().replace(day=28) + timedelta(days=4)).replace(day=1) - datetime.now()
        expire_seconds = int(days_until_month_end.total_seconds())
        
        await cache_service.set(cache_key, json.dumps(new_usage), expire=max(expire_seconds, 86400))
    
    async def _add_organization_usage(self, organization_id: str, tokens: int, cost: float):
        """Add usage to organization monthly total"""
        current_month = datetime.now().strftime("%Y-%m")
        cache_key = f"{self.cache_prefix}:org:{organization_id}:{current_month}"
        
        current_usage = await self._get_organization_usage(organization_id)
        new_usage = {
            "tokens": current_usage["tokens"] + tokens,
            "cost": current_usage["cost"] + cost
        }
        
        # Cache until end of month
        days_until_month_end = (datetime.now().replace(day=28) + timedelta(days=4)).replace(day=1) - datetime.now()
        expire_seconds = int(days_until_month_end.total_seconds())
        
        await cache_service.set(cache_key, json.dumps(new_usage), expire=max(expire_seconds, 86400))
    
    async def _add_endpoint_usage(self, endpoint: str, tokens: int, cost: float):
        """Add usage to endpoint daily total"""
        current_day = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"{self.cache_prefix}:endpoint:{endpoint}:{current_day}"
        
        current_usage = await self._get_endpoint_usage(endpoint)
        new_usage = {
            "tokens": current_usage["tokens"] + tokens,
            "cost": current_usage["cost"] + cost
        }
        
        # Cache until end of day
        tomorrow = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        expire_seconds = int((tomorrow - datetime.now()).total_seconds())
        
        await cache_service.set(cache_key, json.dumps(new_usage), expire=max(expire_seconds, 86400))
    
    async def _add_global_usage(self, tokens: int, cost: float):
        """Add usage to global daily total"""
        current_day = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"{self.cache_prefix}:global:{current_day}"
        
        current_usage = await self._get_global_usage()
        new_usage = {
            "tokens": current_usage["tokens"] + tokens,
            "cost": current_usage["cost"] + cost
        }
        
        # Cache until end of day
        tomorrow = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        expire_seconds = int((tomorrow - datetime.now()).total_seconds())
        
        await cache_service.set(cache_key, json.dumps(new_usage), expire=max(expire_seconds, 86400))
    
    async def get_usage_report(
        self,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        endpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive usage report"""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "period": "current_month",
            "usage": {}
        }
        
        if user_id:
            report["usage"]["user"] = await self._get_user_usage(user_id)
            report["usage"]["user"]["limit"] = await self._get_user_limit(user_id).__dict__
        
        if organization_id:
            report["usage"]["organization"] = await self._get_organization_usage(organization_id)
            report["usage"]["organization"]["limit"] = await self._get_organization_limit(organization_id).__dict__
        
        if endpoint:
            report["usage"]["endpoint"] = await self._get_endpoint_usage(endpoint)
            report["usage"]["endpoint"]["limit"] = await self._get_endpoint_limit(endpoint).__dict__
        
        report["usage"]["global"] = await self._get_global_usage()
        report["usage"]["global"]["limit"] = await self._get_global_limit().__dict__
        
        return report


# Global instance
cost_monitoring_service = CostMonitoringService()
