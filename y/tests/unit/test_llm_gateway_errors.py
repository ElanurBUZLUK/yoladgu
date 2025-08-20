import pytest
from unittest.mock import AsyncMock, patch
from app.services.llm_gateway import LLMGatewayService

@pytest.fixture
def llm_gateway_service():
    return LLMGatewayService()

@pytest.mark.asyncio
async def test_generate_text_error_handling(llm_gateway_service):
    with patch('app.services.llm_providers.llm_router.LLMRouter.generate_with_fallback', new_callable=AsyncMock) as mock_generate:
        mock_generate.side_effect = Exception("LLM API error")
        
        result = await llm_gateway_service.generate_text("test prompt")
        
        assert result["success"] is False
        assert "LLM API error" in result["error"]
        assert result["text"] == ""

@pytest.mark.asyncio
async def test_generate_json_error_handling(llm_gateway_service):
    with patch('app.services.llm_providers.llm_router.LLMRouter.generate_structured_with_fallback', new_callable=AsyncMock) as mock_generate:
        mock_generate.side_effect = Exception("JSON generation error")
        
        result = await llm_gateway_service.generate_json("test prompt", schema={})
        
        assert result["success"] is False
        assert "JSON generation error" in result["error"]
        assert result["parsed_json"] == {}

@pytest.mark.asyncio
async def test_embed_query_error_handling(llm_gateway_service):
    with patch('app.services.embedding_service.embedding_service.embed_texts', new_callable=AsyncMock) as mock_embed:
        mock_embed.side_effect = Exception("Embedding service error")
        
        result = await llm_gateway_service.embed_query("test query")
        
        assert result == []

@pytest.mark.asyncio
async def test_compress_context_error_handling(llm_gateway_service):
    with patch('app.services.llm_providers.llm_router.LLMRouter.generate_with_fallback', new_callable=AsyncMock) as mock_compress:
        mock_compress.side_effect = Exception("Compression error")
        
        context = "This is a very long context that needs to be compressed."
        result = await llm_gateway_service.compress_context(context, max_len=10)
        
        assert result == context[:10] # Fallback to simple truncation
