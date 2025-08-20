from abc import ABC, abstractmethod
from typing import Dict, Any, List
from mcp.types import Tool, CallToolResult, TextContent
import json


class BaseMCPTool(ABC):
    """Base class for all MCP tools"""
    
    def __init__(self):
        self.name = self.get_name()
        self.description = self.get_description()
        self.input_schema = self.get_input_schema()
    
    @abstractmethod
    def get_name(self) -> str:
        """Return tool name"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return tool description"""
        pass
    
    @abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """Return JSON schema for input validation"""
        pass
    
    @abstractmethod
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given arguments"""
        pass
    
    async def call(self, arguments: Dict[str, Any]) -> CallToolResult:
        """MCP call interface"""
        try:
            result = await self.execute(arguments)
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(result, ensure_ascii=False, indent=2)
                    )
                ]
            )
        except Exception as e:
            error_result = {
                "error": str(e),
                "tool": self.name,
                "arguments": arguments
            }
            return CallToolResult(
                content=[
                    TextContent(
                        type="text", 
                        text=json.dumps(error_result, ensure_ascii=False, indent=2)
                    )
                ],
                isError=True
            )
    
    def to_tool_definition(self) -> Tool:
        """Convert to MCP Tool definition"""
        return Tool(
            name=self.name,
            description=self.description,
            inputSchema=self.input_schema
        )