from typing import Optional
from pydantic import BaseModel, Field


class MCPConfig(BaseModel):
    """MCP Configuration"""
    
    # MCP Server settings
    server_command: list[str] = Field(
        default=["python", "-m", "app.mcp.server"],
        description="Command to start MCP server"
    )
    
    # MCP Client settings
    use_mcp: bool = Field(
        default=True,
        description="Whether to use MCP for LLM calls"
    )
    
    # Connection settings
    connection_timeout: int = Field(
        default=30,
        description="MCP connection timeout in seconds"
    )
    
    # Fallback settings
    enable_fallback: bool = Field(
        default=True,
        description="Enable fallback to direct LLM calls"
    )
    
    # Logging settings
    log_level: str = Field(
        default="INFO",
        description="MCP logging level"
    )
    
    # Tool settings
    available_tools: list[str] = Field(
        default=[
            "recommend_math",
            "generate_english_cloze", 
            "assess_cefr",
            "llm_generate"
        ],
        description="Available MCP tools"
    )


# Global MCP config instance
mcp_config = MCPConfig()
