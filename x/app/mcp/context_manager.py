import logging
import json
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.user import User
from app.models.student_attempt import StudentAttempt
from app.models.error_pattern import ErrorPattern
from app.core.mcp_client import mcp_client

logger = logging.getLogger(__name__)


@dataclass
class StudentContext:
    """Student context data for MCP"""
    user_id: str
    current_level: int
    recent_errors: List[str]
    learning_preferences: Dict[str, Any]
    last_activity: datetime
    session_data: Dict[str, Any]
    error_patterns: List[str]
    progress_metrics: Dict[str, Any]


@dataclass
class SessionContext:
    """Session context data for MCP"""
    session_id: str
    user_id: str
    subject: str
    current_topic: str
    difficulty_level: int
    attempted_questions: List[str]
    session_start: datetime
    context_data: Dict[str, Any]


class MCPContextManager:
    """MCP Context Manager for shared memory and context sync"""
    
    def __init__(self):
        self.active_contexts: Dict[str, StudentContext] = {}
        self.session_contexts: Dict[str, SessionContext] = {}
        self.context_cache: Dict[str, Any] = {}
        self.sync_interval = 300  # 5 minutes
        self.last_sync = datetime.utcnow()
    
    async def sync_student_context(self, session: AsyncSession, user_id: str) -> StudentContext:
        """Sync student context to MCP"""
        try:
            # Get user data
            user_result = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = user_result.scalar_one_or_none()
            
            if not user:
                raise ValueError(f"User not found: {user_id}")
            
            # Get recent attempts
            recent_attempts_result = await session.execute(
                select(StudentAttempt)
                .where(StudentAttempt.user_id == user_id)
                .order_by(StudentAttempt.created_at.desc())
                .limit(20)
            )
            recent_attempts = recent_attempts_result.scalars().all()
            
            # Get error patterns
            error_patterns_result = await session.execute(
                select(ErrorPattern)
                .where(ErrorPattern.user_id == user_id)
                .order_by(ErrorPattern.frequency.desc())
                .limit(10)
            )
            error_patterns = error_patterns_result.scalars().all()
            
            # Calculate progress metrics
            progress_metrics = self._calculate_progress_metrics(recent_attempts)
            
            # Create student context
            student_context = StudentContext(
                user_id=user_id,
                current_level=getattr(user, 'current_english_level', 3),
                recent_errors=[attempt.error_analysis.get('patterns', []) for attempt in recent_attempts if attempt.error_analysis],
                learning_preferences=getattr(user, 'preferences', {}),
                last_activity=user.updated_at or user.created_at,
                session_data={},
                error_patterns=[pattern.pattern_type for pattern in error_patterns],
                progress_metrics=progress_metrics
            )
            
            # Cache context
            self.active_contexts[user_id] = student_context
            
            # Sync to MCP
            await self._send_context_to_mcp(student_context)
            
            logger.info(f"Student context synced for user {user_id}")
            return student_context
            
        except Exception as e:
            logger.error(f"Failed to sync student context for {user_id}: {e}")
            raise
    
    async def sync_session_context(self, session_id: str, user_id: str, subject: str, **kwargs) -> SessionContext:
        """Sync session context to MCP"""
        try:
            session_context = SessionContext(
                session_id=session_id,
                user_id=user_id,
                subject=subject,
                current_topic=kwargs.get('current_topic', 'general'),
                difficulty_level=kwargs.get('difficulty_level', 3),
                attempted_questions=kwargs.get('attempted_questions', []),
                session_start=datetime.utcnow(),
                context_data=kwargs.get('context_data', {})
            )
            
            # Cache session context
            self.session_contexts[session_id] = session_context
            
            # Sync to MCP
            await self._send_session_context_to_mcp(session_context)
            
            logger.info(f"Session context synced for session {session_id}")
            return session_context
            
        except Exception as e:
            logger.error(f"Failed to sync session context for {session_id}: {e}")
            raise
    
    async def get_student_context(self, user_id: str) -> Optional[StudentContext]:
        """Get cached student context"""
        return self.active_contexts.get(user_id)
    
    async def get_session_context(self, session_id: str) -> Optional[SessionContext]:
        """Get cached session context"""
        return self.session_contexts.get(session_id)
    
    async def update_context(self, user_id: str, updates: Dict[str, Any]):
        """Update student context"""
        if user_id in self.active_contexts:
            context = self.active_contexts[user_id]
            
            # Update context fields
            for key, value in updates.items():
                if hasattr(context, key):
                    setattr(context, key, value)
            
            # Sync to MCP
            await self._send_context_to_mcp(context)
            
            logger.info(f"Context updated for user {user_id}")
    
    async def _send_context_to_mcp(self, context: StudentContext):
        """Send student context to MCP with full context sync"""
        try:
            if mcp_client.is_connected:
                # Validate context data using Pydantic
                from app.mcp.schema_validator import mcp_schema_validator
                
                context_data = asdict(context)
                
                # Validate context data
                try:
                    validated_data = mcp_schema_validator.validate_input(
                        "context_update", {
                            "context_type": "student",
                            "context_data": context_data,
                            "timestamp": datetime.utcnow().isoformat(),
                            "user_id": context.user_id
                        }
                    )
                except Exception as e:
                    logger.warning(f"Context validation failed: {e}")
                    validated_data = {
                        "context_type": "student",
                        "context_data": context_data,
                        "timestamp": datetime.utcnow().isoformat(),
                        "user_id": context.user_id
                    }
                
                # Send to MCP with full context
                await mcp_client.call_tool(
                    tool_name="update_student_context",
                    arguments=validated_data
                )
                
                logger.info(f"Student context synced to MCP for user {context.user_id}")
        except Exception as e:
            logger.warning(f"Failed to send context to MCP: {e}")
    
    async def _send_session_context_to_mcp(self, context: SessionContext):
        """Send session context to MCP with full context sync"""
        try:
            if mcp_client.is_connected:
                # Validate context data using Pydantic
                from app.mcp.schema_validator import mcp_schema_validator
                
                context_data = asdict(context)
                
                # Validate context data
                try:
                    validated_data = mcp_schema_validator.validate_input(
                        "context_update", {
                            "context_type": "session",
                            "context_data": context_data,
                            "timestamp": datetime.utcnow().isoformat(),
                            "user_id": context.user_id
                        }
                    )
                except Exception as e:
                    logger.warning(f"Session context validation failed: {e}")
                    validated_data = {
                        "context_type": "session",
                        "context_data": context_data,
                        "timestamp": datetime.utcnow().isoformat(),
                        "user_id": context.user_id
                    }
                
                # Send to MCP with full context
                await mcp_client.call_tool(
                    tool_name="update_session_context",
                    arguments=validated_data
                )
                
                logger.info(f"Session context synced to MCP for session {context.session_id}")
        except Exception as e:
            logger.warning(f"Failed to send session context to MCP: {e}")
    
    def _calculate_progress_metrics(self, attempts: List[StudentAttempt]) -> Dict[str, Any]:
        """Calculate progress metrics from attempts"""
        if not attempts:
            return {}
        
        total_attempts = len(attempts)
        correct_attempts = len([a for a in attempts if a.is_correct])
        success_rate = correct_attempts / total_attempts if total_attempts > 0 else 0
        
        # Calculate average time spent
        avg_time = sum(a.time_spent or 0 for a in attempts) / total_attempts if total_attempts > 0 else 0
        
        # Calculate difficulty progression
        difficulties = [a.difficulty_level for a in attempts if a.difficulty_level]
        avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else 3
        
        return {
            "total_attempts": total_attempts,
            "correct_attempts": correct_attempts,
            "success_rate": success_rate,
            "average_time_spent": avg_time,
            "average_difficulty": avg_difficulty,
            "last_activity": max(a.created_at for a in attempts).isoformat() if attempts else None
        }
    
    async def cleanup_old_contexts(self, max_age_hours: int = 24):
        """Cleanup old contexts"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        # Cleanup student contexts
        old_user_ids = [
            user_id for user_id, context in self.active_contexts.items()
            if context.last_activity < cutoff_time
        ]
        for user_id in old_user_ids:
            del self.active_contexts[user_id]
        
        # Cleanup session contexts
        old_session_ids = [
            session_id for session_id, context in self.session_contexts.items()
            if context.session_start < cutoff_time
        ]
        for session_id in old_session_ids:
            del self.session_contexts[session_id]
        
        if old_user_ids or old_session_ids:
            logger.info(f"Cleaned up {len(old_user_ids)} user contexts and {len(old_session_ids)} session contexts")
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get context manager statistics"""
        return {
            "active_student_contexts": len(self.active_contexts),
            "active_session_contexts": len(self.session_contexts),
            "cache_size": len(self.context_cache),
            "last_sync": self.last_sync.isoformat(),
            "sync_interval": self.sync_interval
        }


# Global context manager instance
mcp_context_manager = MCPContextManager()
