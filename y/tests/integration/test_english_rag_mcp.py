import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestEnglishRAGMCPIntegration:
    """English RAG MCP Integration testleri"""
    
    @pytest.fixture
    def mock_mcp_utils(self):
        """Mock MCP utils"""
        with patch('app.core.mcp_utils.mcp_utils') as mock:
            mock.is_initialized = True
            mock.call_tool.return_value = {
                "success": True,
                "data": {
                    "text": "This is a compressed answer based on the context.",
                    "compressed_context": "Compressed context here"
                }
            }
            yield mock
    
    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user"""
        return {
            "id": "test_user_123",
            "email": "test@example.com",
            "current_english_level": 3
        }
    
    def test_answer_endpoint_mcp_integration(self, mock_mcp_utils, mock_user):
        """Test answer endpoint with MCP integration"""
        with patch('app.middleware.auth.get_current_student', return_value=mock_user):
            response = client.post(
                "/api/v1/english-rag/answer",
                json={
                    "query": "What is the past tense of 'go'?",
                    "question": "What is the past tense of 'go'?",
                    "namespace": "grammar",
                    "slot": "active",
                    "k": 3,
                    "compress_context": True,
                    "max_context": 1000
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "sources" in data
            
            # Verify MCP was called
            mock_mcp_utils.call_tool.assert_called()
            call_args = mock_mcp_utils.call_tool.call_args_list
            
            # Should have calls for both compress_context and llm_generate
            assert len(call_args) >= 1
            
            # Check llm_generate call
            llm_call = next((call for call in call_args if call[1]['tool_name'] == 'llm_generate'), None)
            assert llm_call is not None
            assert llm_call[1]['arguments']['output_type'] == 'text'
            assert 'user_id' in llm_call[1]['arguments']
    
    def test_answer_endpoint_mcp_fallback(self, mock_user):
        """Test answer endpoint with MCP fallback"""
        with patch('app.core.mcp_utils.mcp_utils') as mock_mcp:
            mock_mcp.is_initialized = False
            
            with patch('app.services.llm_gateway.llm_gateway') as mock_llm:
                mock_llm.generate_text.return_value = {
                    "success": True,
                    "text": "Fallback answer from direct LLM"
                }
                
                with patch('app.middleware.auth.get_current_student', return_value=mock_user):
                    response = client.post(
                        "/api/v1/english-rag/answer",
                        json={
                            "query": "What is the past tense of 'go'?",
                            "question": "What is the past tense of 'go'?",
                            "namespace": "grammar",
                            "slot": "active",
                            "k": 3
                        }
                    )
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert "answer" in data
                    
                    # Verify direct LLM was called (fallback)
                    mock_llm.generate_text.assert_called()
    
    def test_context_compression_mcp(self, mock_mcp_utils, mock_user):
        """Test context compression via MCP"""
        with patch('app.middleware.auth.get_current_student', return_value=mock_user):
            # Mock the compression response
            mock_mcp_utils.call_tool.side_effect = [
                {
                    "success": True,
                    "data": {
                        "compressed_context": "Compressed context here",
                        "original_length": 1000,
                        "compressed_length": 500,
                        "compression_ratio": 0.5
                    }
                },
                {
                    "success": True,
                    "data": {
                        "text": "Answer based on compressed context"
                    }
                }
            ]
            
            response = client.post(
                "/api/v1/english-rag/answer",
                json={
                    "query": "Test query",
                    "question": "Test question",
                    "namespace": "test",
                    "slot": "active",
                    "k": 3,
                    "compress_context": True,
                    "max_context": 800
                }
            )
            
            assert response.status_code == 200
            
            # Verify compress_context was called
            compress_call = next(
                (call for call in mock_mcp_utils.call_tool.call_args_list 
                 if call[1]['tool_name'] == 'compress_context'), 
                None
            )
            assert compress_call is not None
            assert compress_call[1]['arguments']['max_length'] == 800
    
    def test_mcp_error_handling(self, mock_user):
        """Test MCP error handling and fallback"""
        with patch('app.core.mcp_utils.mcp_utils') as mock_mcp:
            mock_mcp.is_initialized = True
            mock_mcp.call_tool.side_effect = Exception("MCP connection failed")
            
            with patch('app.services.llm_gateway.llm_gateway') as mock_llm:
                mock_llm.generate_text.return_value = {
                    "success": True,
                    "text": "Fallback answer after MCP error"
                }
                
                with patch('app.middleware.auth.get_current_student', return_value=mock_user):
                    response = client.post(
                        "/api/v1/english-rag/answer",
                        json={
                            "query": "Test query",
                            "question": "Test question",
                            "namespace": "test",
                            "slot": "active",
                            "k": 3
                        }
                    )
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert "answer" in data
                    
                    # Verify fallback was used
                    mock_llm.generate_text.assert_called()
    
    def test_cache_integration_with_mcp(self, mock_mcp_utils, mock_user):
        """Test cache integration with MCP responses"""
        with patch('app.middleware.auth.get_current_student', return_value=mock_user):
            # First call - should use MCP
            response1 = client.post(
                "/api/v1/english-rag/answer",
                json={
                    "query": "Cached query",
                    "question": "Cached question",
                    "namespace": "cache_test",
                    "slot": "active",
                    "k": 3
                }
            )
            
            assert response1.status_code == 200
            
            # Second call with same parameters - should use cache
            response2 = client.post(
                "/api/v1/english-rag/answer",
                json={
                    "query": "Cached query",
                    "question": "Cached question",
                    "namespace": "cache_test",
                    "slot": "active",
                    "k": 3
                }
            )
            
            assert response2.status_code == 200
            
            # Both responses should be identical
            assert response1.json() == response2.json()
