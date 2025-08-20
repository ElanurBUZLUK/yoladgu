import logging
from typing import Dict, Any, Optional
from .base import BaseMCPTool
from app.services.llm_gateway import llm_gateway

logger = logging.getLogger(__name__)


class LLMGeneratorTool(BaseMCPTool):
    """LLM Generation Tool for MCP"""
    
    def get_name(self) -> str:
        return "llm_generate"
    
    def get_description(self) -> str:
        return "Generate text or JSON using LLM models"
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The prompt to send to the LLM"
                },
                "system_prompt": {
                    "type": "string",
                    "description": "System prompt for the LLM",
                    "default": ""
                },
                "output_type": {
                    "type": "string",
                    "enum": ["text", "json"],
                    "description": "Type of output to generate",
                    "default": "text"
                },
                "schema": {
                    "type": "object",
                    "description": "JSON schema for structured output (required if output_type is json)"
                },
                "context": {
                    "type": "string",
                    "description": "Additional context for the LLM"
                },
                "temperature": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 2.0,
                    "description": "Temperature for generation",
                    "default": 0.7
                },
                "max_tokens": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 4000,
                    "description": "Maximum tokens to generate",
                    "default": 1000
                }
            },
            "required": ["prompt"]
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute LLM generation"""
        try:
            prompt = arguments["prompt"]
            system_prompt = arguments.get("system_prompt", "")
            output_type = arguments.get("output_type", "text")
            context = arguments.get("context")
            temperature = arguments.get("temperature", 0.7)
            max_tokens = arguments.get("max_tokens", 1000)
            
            if output_type == "json":
                schema = arguments.get("schema")
                if not schema:
                    raise ValueError("Schema is required for JSON output type")
                
                result = await llm_gateway.generate_json(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    schema=schema,
                    context=context,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                return {
                    "success": result.get("success", False),
                    "output": result.get("parsed_json", {}),
                    "usage": result.get("usage", {}),
                    "error": result.get("error"),
                    "method": "llm_gateway"
                }
            
            else:  # text
                result = await llm_gateway.generate_text(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    context=context,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                return {
                    "success": result.get("success", False),
                    "output": result.get("text", ""),
                    "usage": result.get("usage", {}),
                    "error": result.get("error"),
                    "method": "llm_gateway"
                }
                
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {
                "success": False,
                "output": None,
                "usage": {},
                "error": str(e),
                "method": "llm_gateway"
            }
