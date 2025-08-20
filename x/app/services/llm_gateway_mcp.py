import logging
from typing import Dict, Any, List, Optional
from app.core.mcp_client import mcp_client

logger = logging.getLogger(__name__)


class LLMGatewayMCPService:
    """LLM Gateway Service - MCP üzerinden LLM işlemleri"""
    
    def __init__(self):
        self.is_connected = False
    
    async def connect(self) -> bool:
        """MCP client'a bağlan"""
        try:
            self.is_connected = await mcp_client.connect()
            return self.is_connected
        except Exception as e:
            logger.error(f"MCP connection failed: {e}")
            return False
    
    async def disconnect(self):
        """MCP client'tan ayrıl"""
        if self.is_connected:
            await mcp_client.disconnect()
            self.is_connected = False
    
    async def generate_english_question(
        self,
        user_id: str,
        error_patterns: List[str],
        difficulty_level: int,
        question_type: str = "multiple_choice",
        topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """İngilizce soru üretimi - MCP üzerinden"""
        
        if not self.is_connected:
            raise RuntimeError("MCP Client not connected")
        
        try:
            result = await mcp_client.call_tool(
                tool_name="generate_english_cloze",
                arguments={
                    "student_id": user_id,
                    "num_recent_errors": len(error_patterns),
                    "difficulty_level": difficulty_level,
                    "question_type": question_type,
                    "topic": topic,
                    "error_patterns": error_patterns
                }
            )
            
            return {
                "success": True,
                "question": result,
                "method": "mcp"
            }
            
        except Exception as e:
            logger.error(f"MCP English question generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "method": "mcp"
            }
    
    async def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        schema: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """JSON üretimi - MCP üzerinden"""
        
        if not self.is_connected:
            raise RuntimeError("MCP Client not connected")
        
        try:
            # MCP'de JSON generation tool'u yoksa, fallback
            # Bu durumda normal LLM gateway'e yönlendir
            logger.warning("JSON generation not available via MCP, using fallback")
            return await self._fallback_json_generation(
                prompt, system_prompt, schema, context, temperature, max_tokens
            )
            
        except Exception as e:
            logger.error(f"MCP JSON generation failed: {e}")
            return {
                "success": False,
                "parsed_json": {},
                "usage": {},
                "error": str(e),
                "method": "mcp"
            }
    
    async def generate_text(
        self,
        prompt: str,
        system_prompt: str = "",
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Metin üretimi - MCP üzerinden"""
        
        if not self.is_connected:
            raise RuntimeError("MCP Client not connected")
        
        try:
            # MCP'de text generation tool'u yoksa, fallback
            logger.warning("Text generation not available via MCP, using fallback")
            return await self._fallback_text_generation(
                prompt, system_prompt, context, temperature, max_tokens
            )
            
        except Exception as e:
            logger.error(f"MCP text generation failed: {e}")
            return {
                "success": False,
                "text": "",
                "usage": {},
                "error": str(e),
                "method": "mcp"
            }
    
    async def _fallback_json_generation(
        self,
        prompt: str,
        system_prompt: str = "",
        schema: Optional[Dict[str, Any]] = None,
        context: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Fallback JSON generation - normal LLM gateway kullan"""
        
        from .llm_gateway import llm_gateway
        
        return await llm_gateway.generate_json(
            prompt=prompt,
            system_prompt=system_prompt,
            schema=schema,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    async def _fallback_text_generation(
        self,
        prompt: str,
        system_prompt: str = "",
        context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """Fallback text generation - normal LLM gateway kullan"""
        
        from .llm_gateway import llm_gateway
        
        return await llm_gateway.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    async def list_available_tools(self) -> List[Dict[str, Any]]:
        """MCP'de mevcut tool'ları listele"""
        
        if not self.is_connected:
            raise RuntimeError("MCP Client not connected")
        
        try:
            tools = await mcp_client.list_tools()
            return tools
        except Exception as e:
            logger.error(f"Failed to list MCP tools: {e}")
            return []


# Global MCP LLM gateway instance
llm_gateway_mcp = LLMGatewayMCPService()
