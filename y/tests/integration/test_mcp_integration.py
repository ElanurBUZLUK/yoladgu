import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.llm_gateway_mcp_integration import LLMGatewayIntegrationService
from app.core.mcp_client import MCPClient


class TestMCPIntegration:
    """MCP Integration testleri"""
    
    @pytest.fixture
    def integration_service(self):
        """Integration service instance"""
        return LLMGatewayIntegrationService(use_mcp=True)
    
    @pytest.fixture
    def mock_mcp_client(self):
        """Mock MCP client"""
        client = AsyncMock(spec=MCPClient)
        client.is_connected = True
        client.call_tool.return_value = {
            "success": True,
            "question": {"content": "Test question", "correct_answer": "A"}
        }
        return client
    
    async def test_initialize_mcp_success(self, integration_service):
        """MCP başarılı başlatma testi"""
        with patch('app.services.llm_gateway_mcp_integration.llm_gateway_mcp') as mock_mcp:
            mock_mcp.connect.return_value = True
            
            await integration_service.initialize()
            
            assert integration_service.use_mcp is True
            assert integration_service.mcp_connected is True
            mock_mcp.connect.assert_called_once()
    
    async def test_initialize_mcp_failure_fallback(self, integration_service):
        """MCP başarısız başlatma - fallback testi"""
        with patch('app.services.llm_gateway_mcp_integration.llm_gateway_mcp') as mock_mcp:
            mock_mcp.connect.return_value = False
            
            await integration_service.initialize()
            
            assert integration_service.use_mcp is False
            assert integration_service.mcp_connected is False
    
    async def test_generate_english_question_mcp_success(self, integration_service, mock_mcp_client):
        """MCP üzerinden İngilizce soru üretimi başarı testi"""
        integration_service.mcp_connected = True
        
        with patch('app.services.llm_gateway_mcp_integration.llm_gateway_mcp') as mock_mcp:
            mock_mcp.generate_english_question.return_value = {
                "success": True,
                "question": {"content": "Test question"}
            }
            
            result = await integration_service.generate_english_question(
                user_id="test_user",
                error_patterns=["past_tense"],
                difficulty_level=3
            )
            
            assert result["method"] == "mcp"
            assert result["success"] is True
            mock_mcp.generate_english_question.assert_called_once()
    
    async def test_generate_english_question_mcp_failure_fallback(self, integration_service):
        """MCP başarısız - direct fallback testi"""
        integration_service.mcp_connected = True
        
        with patch('app.services.llm_gateway_mcp_integration.llm_gateway_mcp') as mock_mcp:
            mock_mcp.generate_english_question.side_effect = Exception("MCP Error")
            
        with patch('app.services.llm_gateway_mcp_integration.llm_gateway') as mock_direct:
            mock_direct.generate_english_question.return_value = {
                "success": True,
                "question": {"content": "Direct question"}
            }
            
            result = await integration_service.generate_english_question(
                user_id="test_user",
                error_patterns=["past_tense"],
                difficulty_level=3
            )
            
            assert result["method"] == "direct"
            assert result["success"] is True
            mock_direct.generate_english_question.assert_called_once()
    
    async def test_generate_json_mcp_success(self, integration_service):
        """MCP üzerinden JSON üretimi başarı testi"""
        integration_service.mcp_connected = True
        
        with patch('app.services.llm_gateway_mcp_integration.llm_gateway_mcp') as mock_mcp:
            mock_mcp.generate_json.return_value = {
                "success": True,
                "parsed_json": {"key": "value"}
            }
            
            result = await integration_service.generate_json(
                prompt="Test prompt",
                schema={"type": "object"}
            )
            
            assert result["method"] == "mcp"
            assert result["success"] is True
            mock_mcp.generate_json.assert_called_once()
    
    async def test_generate_text_mcp_success(self, integration_service):
        """MCP üzerinden text üretimi başarı testi"""
        integration_service.mcp_connected = True
        
        with patch('app.services.llm_gateway_mcp_integration.llm_gateway_mcp') as mock_mcp:
            mock_mcp.generate_text.return_value = {
                "success": True,
                "text": "Generated text"
            }
            
            result = await integration_service.generate_text(
                prompt="Test prompt"
            )
            
            assert result["method"] == "mcp"
            assert result["success"] is True
            mock_mcp.generate_text.assert_called_once()
    
    async def test_get_status(self, integration_service):
        """Status alma testi"""
        integration_service.mcp_connected = True
        
        status = integration_service.get_status()
        
        assert status["use_mcp"] is True
        assert status["mcp_connected"] is True
        assert "mcp" in status["available_methods"]
        assert "direct" in status["available_methods"]
    
    async def test_cleanup(self, integration_service):
        """Cleanup testi"""
        integration_service.mcp_connected = True
        
        with patch('app.services.llm_gateway_mcp_integration.llm_gateway_mcp') as mock_mcp:
            await integration_service.cleanup()
            
            assert integration_service.mcp_connected is False
            mock_mcp.disconnect.assert_called_once()
