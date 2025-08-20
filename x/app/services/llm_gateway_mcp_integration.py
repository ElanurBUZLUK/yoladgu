import logging
from typing import Dict, Any, List, Optional
from app.services.llm_gateway_mcp import llm_gateway_mcp
from app.services.llm_gateway import llm_gateway

logger = logging.getLogger(__name__)


class LLMGatewayIntegrationService:
    """LLM Gateway Integration Service - MCP ve normal gateway arasında geçiş"""
    
    def __init__(self, use_mcp: bool = True):
        self.use_mcp = use_mcp
        self.mcp_connected = False
    
    async def initialize(self):
        """Initialize the service"""
        if self.use_mcp:
            try:
                self.mcp_connected = await llm_gateway_mcp.connect()
                if self.mcp_connected:
                    logger.info("✅ MCP LLM Gateway connected")
                else:
                    logger.warning("⚠️ MCP connection failed, falling back to direct LLM")
                    self.use_mcp = False
            except Exception as e:
                logger.error(f"❌ MCP initialization failed: {e}")
                self.use_mcp = False
    
    async def cleanup(self):
        """Cleanup the service"""
        if self.mcp_connected:
            await llm_gateway_mcp.disconnect()
            self.mcp_connected = False
    
    async def generate_english_question(
        self,
        user_id: str,
        error_patterns: List[str],
        difficulty_level: int,
        question_type: str = "multiple_choice",
        topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """İngilizce soru üretimi - MCP veya direct"""
        
        if self.use_mcp and self.mcp_connected:
            try:
                result = await llm_gateway_mcp.generate_english_question(
                    user_id=user_id,
                    error_patterns=error_patterns,
                    difficulty_level=difficulty_level,
                    question_type=question_type,
                    topic=topic
                )
                result["method"] = "mcp"
                return result
            except Exception as e:
                logger.error(f"MCP English question generation failed: {e}")
                # Fallback to direct
                self.use_mcp = False
        
        # Direct LLM gateway
        result = await llm_gateway.generate_english_question(
            user_id=user_id,
            error_patterns=error_patterns,
            difficulty_level=difficulty_level,
            question_type=question_type,
            topic=topic
        )
        result["method"] = "direct"
        return result
    
    async def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        schema: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """JSON üretimi - MCP veya direct"""
        
        if self.use_mcp and self.mcp_connected:
            try:
                result = await llm_gateway_mcp.generate_json(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    schema=schema,
                    context=context,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                result["method"] = "mcp"
                return result
            except Exception as e:
                logger.error(f"MCP JSON generation failed: {e}")
                # Fallback to direct
                self.use_mcp = False
        
        # Direct LLM gateway
        result = await llm_gateway.generate_json(
            prompt=prompt,
            system_prompt=system_prompt,
            schema=schema,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens
        )
        result["method"] = "direct"
        return result
    
    async def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Metin üretimi - MCP veya direct"""
        
        if self.use_mcp and self.mcp_connected:
            try:
                result = await llm_gateway_mcp.generate_text(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    context=context,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                result["method"] = "mcp"
                return result
            except Exception as e:
                logger.error(f"MCP text generation failed: {e}")
                # Fallback to direct
                self.use_mcp = False
        
        # Direct LLM gateway
        result = await llm_gateway.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens
        )
        result["method"] = "direct"
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "use_mcp": self.use_mcp,
            "mcp_connected": self.mcp_connected,
            "available_methods": ["mcp", "direct"]
        }


# Global integration service instance
llm_gateway_integration = LLMGatewayIntegrationService(use_mcp=True)
