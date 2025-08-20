from typing import Dict, List, Optional, Set, Callable, Any
from enum import Enum
from functools import wraps
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog
from ..core.config import settings
from ..core.security import get_current_user, verify_token
from ..models.user import User

logger = structlog.get_logger()

# Security scheme for OpenAPI docs
security_scheme = HTTPBearer()


class Permission(str, Enum):
    """System permissions"""
    # User management
    USER_READ = "user:read"
    USER_CREATE = "user:create"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    
    # Question management
    QUESTION_READ = "question:read"
    QUESTION_CREATE = "question:create"
    QUESTION_UPDATE = "question:update"
    QUESTION_DELETE = "question:delete"
    
    # Student attempts
    ATTEMPT_READ = "attempt:read"
    ATTEMPT_CREATE = "attempt:create"
    ATTEMPT_UPDATE = "attempt:update"
    ATTEMPT_DELETE = "attempt:delete"
    
    # Analytics
    ANALYTICS_READ = "analytics:read"
    ANALYTICS_CREATE = "analytics:create"
    ANALYTICS_UPDATE = "analytics:update"
    ANALYTICS_DELETE = "analytics:delete"
    
    # System management
    SYSTEM_READ = "system:read"
    SYSTEM_UPDATE = "system:update"
    SYSTEM_DELETE = "system:delete"
    
    # LLM management
    LLM_READ = "llm:read"
    LLM_UPDATE = "llm:update"
    LLM_DELETE = "llm:delete"
    
    # Vector management
    VECTOR_READ = "vector:read"
    VECTOR_UPDATE = "vector:update"
    VECTOR_DELETE = "vector:delete"
    
    # Monitoring
    MONITORING_READ = "monitoring:read"
    MONITORING_UPDATE = "monitoring:update"
    
    # Content moderation
    MODERATION_READ = "moderation:read"
    MODERATION_UPDATE = "moderation:update"
    
    # Cost monitoring
    COST_READ = "cost:read"
    COST_UPDATE = "cost:update"


class Role(str, Enum):
    """User roles"""
    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"
    SYSTEM = "system"


class SecurityService:
    """Enhanced security service with role-based access control"""
    
    def __init__(self):
        self._role_permissions: Dict[Role, Set[Permission]] = self._initialize_role_permissions()
        self._endpoint_permissions: Dict[str, Set[Permission]] = self._initialize_endpoint_permissions()
        logger.info("Security service initialized with role-based access control")
    
    def _initialize_role_permissions(self) -> Dict[Role, Set[Permission]]:
        """Initialize role-permission mappings"""
        return {
            Role.STUDENT: {
                Permission.QUESTION_READ,
                Permission.ATTEMPT_READ,
                Permission.ATTEMPT_CREATE,
                Permission.ATTEMPT_UPDATE,
                Permission.ANALYTICS_READ,  # Own analytics only
            },
            Role.TEACHER: {
                Permission.USER_READ,
                Permission.QUESTION_READ,
                Permission.QUESTION_CREATE,
                Permission.QUESTION_UPDATE,
                Permission.ATTEMPT_READ,
                Permission.ATTEMPT_UPDATE,
                Permission.ANALYTICS_READ,
                Permission.ANALYTICS_CREATE,
                Permission.ANALYTICS_UPDATE,
                Permission.MODERATION_READ,
                Permission.MODERATION_UPDATE,
            },
            Role.ADMIN: {
                Permission.USER_READ,
                Permission.USER_CREATE,
                Permission.USER_UPDATE,
                Permission.USER_DELETE,
                Permission.QUESTION_READ,
                Permission.QUESTION_CREATE,
                Permission.QUESTION_UPDATE,
                Permission.QUESTION_DELETE,
                Permission.ATTEMPT_READ,
                Permission.ATTEMPT_CREATE,
                Permission.ATTEMPT_UPDATE,
                Permission.ATTEMPT_DELETE,
                Permission.ANALYTICS_READ,
                Permission.ANALYTICS_CREATE,
                Permission.ANALYTICS_UPDATE,
                Permission.ANALYTICS_DELETE,
                Permission.SYSTEM_READ,
                Permission.SYSTEM_UPDATE,
                Permission.LLM_READ,
                Permission.LLM_UPDATE,
                Permission.VECTOR_READ,
                Permission.VECTOR_UPDATE,
                Permission.MONITORING_READ,
                Permission.MONITORING_UPDATE,
                Permission.MODERATION_READ,
                Permission.MODERATION_UPDATE,
                Permission.COST_READ,
                Permission.COST_UPDATE,
            },
            Role.SYSTEM: {
                # System role has all permissions
                *[perm for perm in Permission]
            }
        }
    
    def _initialize_endpoint_permissions(self) -> Dict[str, Set[Permission]]:
        """Initialize endpoint-permission mappings"""
        return {
            # User endpoints
            "/api/v1/users/": {Permission.USER_READ},
            "/api/v1/users/me": {Permission.USER_READ},
            "/api/v1/users/{user_id}": {Permission.USER_READ},
            "POST:/api/v1/users/": {Permission.USER_CREATE},
            "PUT:/api/v1/users/{user_id}": {Permission.USER_UPDATE},
            "DELETE:/api/v1/users/{user_id}": {Permission.USER_DELETE},
            
            # Question endpoints
            "/api/v1/questions/": {Permission.QUESTION_READ},
            "/api/v1/questions/{question_id}": {Permission.QUESTION_READ},
            "POST:/api/v1/questions/": {Permission.QUESTION_CREATE},
            "PUT:/api/v1/questions/{question_id}": {Permission.QUESTION_UPDATE},
            "DELETE:/api/v1/questions/{question_id}": {Permission.QUESTION_DELETE},
            
            # Math RAG endpoints
            "/api/v1/math/rag/": {Permission.QUESTION_READ},
            "POST:/api/v1/math/rag/next-question": {Permission.ATTEMPT_CREATE},
            "POST:/api/v1/math/rag/submit-answer": {Permission.ATTEMPT_CREATE},
            "GET:/api/v1/math/rag/profile": {Permission.ATTEMPT_READ},
            "GET:/api/v1/math/rag/analytics": {Permission.ANALYTICS_READ},
            
            # English RAG endpoints
            "/api/v1/english/rag/": {Permission.QUESTION_READ},
            "POST:/api/v1/english/rag/next-question": {Permission.ATTEMPT_CREATE},
            
            # Dashboard endpoints
            "/api/v1/dashboard/": {Permission.ANALYTICS_READ},
            "/api/v1/dashboard/overview": {Permission.ANALYTICS_READ},
            "/api/v1/dashboard/subject-progress": {Permission.ANALYTICS_READ},
            
            # LLM management endpoints
            "/api/v1/llm/": {Permission.LLM_READ},
            "POST:/api/v1/llm/policies": {Permission.LLM_UPDATE},
            "POST:/api/v1/llm/select-policy": {Permission.LLM_UPDATE},
            
            # Vector management endpoints
            "/api/v1/vector/": {Permission.VECTOR_READ},
            "POST:/api/v1/vector/batch-upsert": {Permission.VECTOR_UPDATE},
            "POST:/api/v1/vector/rebuild-index": {Permission.VECTOR_UPDATE},
            
            # Monitoring endpoints
            "/api/v1/monitoring/": {Permission.MONITORING_READ},
            "/api/v1/monitoring/metrics": {Permission.MONITORING_READ},
            "/api/v1/monitoring/health": {Permission.MONITORING_READ},
            
            # Content moderation endpoints
            "/api/v1/moderation/": {Permission.MODERATION_READ},
            "POST:/api/v1/moderation/moderate": {Permission.MODERATION_UPDATE},
            
            # Cost monitoring endpoints
            "/api/v1/cost/": {Permission.COST_READ},
            "POST:/api/v1/cost/limits": {Permission.COST_UPDATE},
        }
    
    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission"""
        if not user or not user.is_active:
            return False
        
        user_role = Role(user.role)
        user_permissions = self._role_permissions.get(user_role, set())
        
        return permission in user_permissions
    
    def has_any_permission(self, user: User, permissions: List[Permission]) -> bool:
        """Check if user has any of the specified permissions"""
        return any(self.has_permission(user, perm) for perm in permissions)
    
    def has_all_permissions(self, user: User, permissions: List[Permission]) -> bool:
        """Check if user has all of the specified permissions"""
        return all(self.has_permission(user, perm) for perm in permissions)
    
    def get_user_permissions(self, user: User) -> Set[Permission]:
        """Get all permissions for a user"""
        if not user or not user.is_active:
            return set()
        
        user_role = Role(user.role)
        return self._role_permissions.get(user_role, set())
    
    def get_endpoint_permissions(self, endpoint: str, method: str = "GET") -> Set[Permission]:
        """Get required permissions for an endpoint"""
        # Try method-specific endpoint first
        method_endpoint = f"{method}:{endpoint}"
        if method_endpoint in self._endpoint_permissions:
            return self._endpoint_permissions[method_endpoint]
        
        # Fall back to general endpoint
        return self._endpoint_permissions.get(endpoint, set())
    
    def can_access_endpoint(self, user: User, endpoint: str, method: str = "GET") -> bool:
        """Check if user can access specific endpoint"""
        required_permissions = self.get_endpoint_permissions(endpoint, method)
        
        if not required_permissions:
            # No permissions required for this endpoint
            return True
        
        return self.has_all_permissions(user, list(required_permissions))
    
    def require_permission(self, permission: Permission):
        """Decorator to require specific permission"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract user from kwargs or request
                user = kwargs.get('current_user')
                if not user:
                    # Try to get from request
                    request = kwargs.get('request')
                    if request:
                        user = getattr(request.state, 'user', None)
                
                if not user:
                    raise HTTPException(status_code=401, detail="Authentication required")
                
                if not self.has_permission(user, permission):
                    logger.warning("Permission denied", 
                                 user_id=user.id, 
                                 permission=permission,
                                 endpoint=func.__name__)
                    raise HTTPException(
                        status_code=403, 
                        detail=f"Permission denied: {permission}"
                    )
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def require_any_permission(self, permissions: List[Permission]):
        """Decorator to require any of the specified permissions"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                user = kwargs.get('current_user')
                if not user:
                    request = kwargs.get('request')
                    if request:
                        user = getattr(request.state, 'user', None)
                
                if not user:
                    raise HTTPException(status_code=401, detail="Authentication required")
                
                if not self.has_any_permission(user, permissions):
                    logger.warning("Permission denied", 
                                 user_id=user.id, 
                                 permissions=permissions,
                                 endpoint=func.__name__)
                    raise HTTPException(
                        status_code=403, 
                        detail=f"Permission denied: requires one of {permissions}"
                    )
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def require_role(self, role: Role):
        """Decorator to require specific role"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                user = kwargs.get('current_user')
                if not user:
                    request = kwargs.get('request')
                    if request:
                        user = getattr(request.state, 'user', None)
                
                if not user:
                    raise HTTPException(status_code=401, detail="Authentication required")
                
                if user.role != role.value:
                    logger.warning("Role access denied", 
                                 user_id=user.id, 
                                 user_role=user.role,
                                 required_role=role.value,
                                 endpoint=func.__name__)
                    raise HTTPException(
                        status_code=403, 
                        detail=f"Role access denied: requires {role.value}"
                    )
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def require_any_role(self, roles: List[Role]):
        """Decorator to require any of the specified roles"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                user = kwargs.get('current_user')
                if not user:
                    request = kwargs.get('request')
                    if request:
                        user = getattr(request.state, 'user', None)
                
                if not user:
                    raise HTTPException(status_code=401, detail="Authentication required")
                
                if user.role not in [role.value for role in roles]:
                    logger.warning("Role access denied", 
                                 user_id=user.id, 
                                 user_role=user.role,
                                 required_roles=[role.value for role in roles],
                                 endpoint=func.__name__)
                    raise HTTPException(
                        status_code=403, 
                        detail=f"Role access denied: requires one of {[role.value for role in roles]}"
                    )
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator


# Global security service instance
security_service = SecurityService()


# Dependency functions for FastAPI
async def require_permission(permission: Permission, current_user: User = Depends(get_current_user)) -> User:
    """FastAPI dependency to require specific permission"""
    if not security_service.has_permission(current_user, permission):
        raise HTTPException(
            status_code=403, 
            detail=f"Permission denied: {permission}"
        )
    return current_user


async def require_any_permission(permissions: List[Permission], current_user: User = Depends(get_current_user)) -> User:
    """FastAPI dependency to require any of the specified permissions"""
    if not security_service.has_any_permission(current_user, permissions):
        raise HTTPException(
            status_code=403, 
            detail=f"Permission denied: requires one of {permissions}"
        )
    return current_user


async def require_role(role: Role, current_user: User = Depends(get_current_user)) -> User:
    """FastAPI dependency to require specific role"""
    if current_user.role != role.value:
        raise HTTPException(
            status_code=403, 
            detail=f"Role access denied: requires {role.value}"
        )
    return current_user


async def require_any_role(roles: List[Role], current_user: User = Depends(get_current_user)) -> User:
    """FastAPI dependency to require any of the specified roles"""
    if current_user.role not in [role.value for role in roles]:
        raise HTTPException(
            status_code=403, 
            detail=f"Role access denied: requires one of {[role.value for role in roles]}"
        )
    return current_user


# Convenience functions for common permission checks
async def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require admin role"""
    return await require_role(Role.ADMIN, current_user)


async def require_teacher_or_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require teacher or admin role"""
    return await require_any_role([Role.TEACHER, Role.ADMIN], current_user)


async def require_student_or_teacher_or_admin(current_user: User = Depends(get_current_user)) -> User:
    """Require student, teacher, or admin role"""
    return await require_any_role([Role.STUDENT, Role.TEACHER, Role.ADMIN], current_user)


# Permission-specific dependencies
async def require_user_management(current_user: User = Depends(get_current_user)) -> User:
    """Require user management permissions"""
    return await require_permission(Permission.USER_READ, current_user)


async def require_question_management(current_user: User = Depends(get_current_user)) -> User:
    """Require question management permissions"""
    return await require_permission(Permission.QUESTION_READ, current_user)


async def require_analytics_access(current_user: User = Depends(get_current_user)) -> User:
    """Require analytics access permissions"""
    return await require_permission(Permission.ANALYTICS_READ, current_user)


async def require_system_management(current_user: User = Depends(get_current_user)) -> User:
    """Require system management permissions"""
    return await require_permission(Permission.SYSTEM_READ, current_user)


async def require_llm_management(current_user: User = Depends(get_current_user)) -> User:
    """Require LLM management permissions"""
    return await require_permission(Permission.LLM_READ, current_user)


async def require_vector_management(current_user: User = Depends(get_current_user)) -> User:
    """Require vector management permissions"""
    return await require_permission(Permission.VECTOR_READ, current_user)


async def require_monitoring_access(current_user: User = Depends(get_current_user)) -> User:
    """Require monitoring access permissions"""
    return await require_permission(Permission.MONITORING_READ, current_user)
