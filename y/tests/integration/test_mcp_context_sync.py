import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from app.mcp.context_manager import MCPContextManager, StudentContext, SessionContext


class TestMCPContextSync:
    """MCP Context Sync testleri"""
    
    @pytest.fixture
    def context_manager(self):
        """Context manager instance"""
        return MCPContextManager()
    
    @pytest.fixture
    def mock_session(self):
        """Mock database session"""
        session = AsyncMock()
        
        # Mock user
        mock_user = MagicMock()
        mock_user.id = "test_user_123"
        mock_user.current_english_level = 4
        mock_user.preferences = {"difficulty": "adaptive"}
        mock_user.updated_at = datetime.utcnow()
        mock_user.created_at = datetime.utcnow()
        
        # Mock attempts
        mock_attempts = [
            MagicMock(
                is_correct=True,
                error_analysis={"patterns": ["past_tense"]},
                time_spent=30,
                difficulty_level=3,
                created_at=datetime.utcnow()
            ),
            MagicMock(
                is_correct=False,
                error_analysis={"patterns": ["articles", "past_tense"]},
                time_spent=45,
                difficulty_level=4,
                created_at=datetime.utcnow()
            )
        ]
        
        # Mock error patterns
        mock_patterns = [
            MagicMock(pattern_type="past_tense", frequency=5),
            MagicMock(pattern_type="articles", frequency=3)
        ]
        
        session.execute.return_value.scalar_one_or_none.return_value = mock_user
        session.execute.return_value.scalars.return_value.all.side_effect = [
            mock_attempts,  # For attempts
            mock_patterns   # For error patterns
        ]
        
        return session
    
    async def test_sync_student_context_success(self, context_manager, mock_session):
        """Student context sync başarı testi"""
        with patch('app.mcp.context_manager.mcp_client') as mock_mcp_client:
            mock_mcp_client.is_connected = True
            mock_mcp_client.call_tool.return_value = {"success": True}
            
            student_context = await context_manager.sync_student_context(
                mock_session, "test_user_123"
            )
            
            assert isinstance(student_context, StudentContext)
            assert student_context.user_id == "test_user_123"
            assert student_context.current_level == 4
            assert "past_tense" in student_context.error_patterns
            assert "articles" in student_context.error_patterns
            assert student_context.progress_metrics["total_attempts"] == 2
            assert student_context.progress_metrics["success_rate"] == 0.5
            
            # Check if context was cached
            assert "test_user_123" in context_manager.active_contexts
            
            # Check if MCP was called
            mock_mcp_client.call_tool.assert_called_once()
    
    async def test_sync_session_context_success(self, context_manager):
        """Session context sync başarı testi"""
        with patch('app.mcp.context_manager.mcp_client') as mock_mcp_client:
            mock_mcp_client.is_connected = True
            mock_mcp_client.call_tool.return_value = {"success": True}
            
            session_context = await context_manager.sync_session_context(
                session_id="session_123",
                user_id="test_user_123",
                subject="english",
                current_topic="grammar",
                difficulty_level=4,
                attempted_questions=["q1", "q2"]
            )
            
            assert isinstance(session_context, SessionContext)
            assert session_context.session_id == "session_123"
            assert session_context.user_id == "test_user_123"
            assert session_context.subject == "english"
            assert session_context.current_topic == "grammar"
            assert session_context.difficulty_level == 4
            assert session_context.attempted_questions == ["q1", "q2"]
            
            # Check if context was cached
            assert "session_123" in context_manager.session_contexts
            
            # Check if MCP was called
            mock_mcp_client.call_tool.assert_called_once()
    
    async def test_get_student_context(self, context_manager):
        """Get cached student context testi"""
        # Add a context to cache
        mock_context = StudentContext(
            user_id="test_user_123",
            current_level=4,
            recent_errors=["past_tense"],
            learning_preferences={"difficulty": "adaptive"},
            last_activity=datetime.utcnow(),
            session_data={},
            error_patterns=["past_tense"],
            progress_metrics={"total_attempts": 10}
        )
        context_manager.active_contexts["test_user_123"] = mock_context
        
        # Get context
        retrieved_context = await context_manager.get_student_context("test_user_123")
        
        assert retrieved_context == mock_context
        assert retrieved_context.user_id == "test_user_123"
        assert retrieved_context.current_level == 4
    
    async def test_get_session_context(self, context_manager):
        """Get cached session context testi"""
        # Add a session context to cache
        mock_session_context = SessionContext(
            session_id="session_123",
            user_id="test_user_123",
            subject="english",
            current_topic="grammar",
            difficulty_level=4,
            attempted_questions=["q1"],
            session_start=datetime.utcnow(),
            context_data={}
        )
        context_manager.session_contexts["session_123"] = mock_session_context
        
        # Get context
        retrieved_context = await context_manager.get_session_context("session_123")
        
        assert retrieved_context == mock_session_context
        assert retrieved_context.session_id == "session_123"
        assert retrieved_context.subject == "english"
    
    async def test_update_context(self, context_manager):
        """Context update testi"""
        with patch('app.mcp.context_manager.mcp_client') as mock_mcp_client:
            mock_mcp_client.is_connected = True
            mock_mcp_client.call_tool.return_value = {"success": True}
            
            # Add a context to cache
            mock_context = StudentContext(
                user_id="test_user_123",
                current_level=4,
                recent_errors=["past_tense"],
                learning_preferences={"difficulty": "adaptive"},
                last_activity=datetime.utcnow(),
                session_data={},
                error_patterns=["past_tense"],
                progress_metrics={"total_attempts": 10}
            )
            context_manager.active_contexts["test_user_123"] = mock_context
            
            # Update context
            updates = {
                "current_level": 5,
                "recent_errors": ["past_tense", "articles"]
            }
            
            await context_manager.update_context("test_user_123", updates)
            
            # Check if context was updated
            updated_context = context_manager.active_contexts["test_user_123"]
            assert updated_context.current_level == 5
            assert updated_context.recent_errors == ["past_tense", "articles"]
            
            # Check if MCP was called
            mock_mcp_client.call_tool.assert_called_once()
    
    async def test_cleanup_old_contexts(self, context_manager):
        """Old contexts cleanup testi"""
        from datetime import timedelta
        
        # Add old contexts
        old_time = datetime.utcnow() - timedelta(hours=25)
        
        old_student_context = StudentContext(
            user_id="old_user",
            current_level=3,
            recent_errors=[],
            learning_preferences={},
            last_activity=old_time,
            session_data={},
            error_patterns=[],
            progress_metrics={}
        )
        
        old_session_context = SessionContext(
            session_id="old_session",
            user_id="old_user",
            subject="english",
            current_topic="general",
            difficulty_level=3,
            attempted_questions=[],
            session_start=old_time,
            context_data={}
        )
        
        context_manager.active_contexts["old_user"] = old_student_context
        context_manager.session_contexts["old_session"] = old_session_context
        
        # Add new contexts
        new_time = datetime.utcnow()
        
        new_student_context = StudentContext(
            user_id="new_user",
            current_level=4,
            recent_errors=[],
            learning_preferences={},
            last_activity=new_time,
            session_data={},
            error_patterns=[],
            progress_metrics={}
        )
        
        new_session_context = SessionContext(
            session_id="new_session",
            user_id="new_user",
            subject="english",
            current_topic="general",
            difficulty_level=4,
            attempted_questions=[],
            session_start=new_time,
            context_data={}
        )
        
        context_manager.active_contexts["new_user"] = new_student_context
        context_manager.session_contexts["new_session"] = new_session_context
        
        # Cleanup old contexts
        await context_manager.cleanup_old_contexts(max_age_hours=24)
        
        # Check that old contexts were removed
        assert "old_user" not in context_manager.active_contexts
        assert "old_session" not in context_manager.session_contexts
        
        # Check that new contexts remain
        assert "new_user" in context_manager.active_contexts
        assert "new_session" in context_manager.session_contexts
    
    def test_get_context_stats(self, context_manager):
        """Context stats testi"""
        # Add some contexts
        context_manager.active_contexts["user1"] = MagicMock()
        context_manager.active_contexts["user2"] = MagicMock()
        context_manager.session_contexts["session1"] = MagicMock()
        context_manager.context_cache["cache1"] = "data1"
        
        stats = context_manager.get_context_stats()
        
        assert stats["active_student_contexts"] == 2
        assert stats["active_session_contexts"] == 1
        assert stats["cache_size"] == 1
        assert "last_sync" in stats
        assert stats["sync_interval"] == 300
