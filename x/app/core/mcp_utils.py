import logging
from typing import Dict, Any, List, Optional, Union
from functools import wraps
from .mcp_client import mcp_client, MCPTransportType

logger = logging.getLogger(__name__)


class MCPUtils:
    """MCP Client Utility Wrapper"""
    
    def __init__(self):
        self.is_initialized = False
    
    async def initialize(self, transport_type: MCPTransportType = MCPTransportType.HTTP) -> bool:
        """Initialize MCP client"""
        try:
            mcp_client.transport_type = transport_type
            self.is_initialized = await mcp_client.connect()
            if self.is_initialized:
                logger.info("✅ MCP Utils initialized successfully")
            return self.is_initialized
        except Exception as e:
            logger.error(f"❌ MCP Utils initialization failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup MCP client"""
        if self.is_initialized:
            await mcp_client.disconnect()
            self.is_initialized = False
            logger.info("MCP Utils cleaned up")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP tool with error handling"""
        if not self.is_initialized:
            raise RuntimeError("MCP Utils not initialized")
        
        try:
            result = await mcp_client.call_tool(tool_name, arguments)
            return {
                "success": True,
                "data": result,
                "method": "mcp"
            }
        except Exception as e:
            logger.error(f"MCP tool call failed for {tool_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "method": "mcp"
            }
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools"""
        if not self.is_initialized:
            raise RuntimeError("MCP Utils not initialized")
        
        try:
            tools = await mcp_client.list_tools()
            return tools
        except Exception as e:
            logger.error(f"Failed to list MCP tools: {e}")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get MCP utils status"""
        return {
            "initialized": self.is_initialized,
            "transport_type": mcp_client.transport_type.value if mcp_client.transport_type else None,
            "connected": mcp_client.is_connected
        }


def mcp_tool_call(tool_name: str, fallback_func=None):
    """Decorator for MCP tool calls with fallback"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Try MCP first
            if mcp_utils.is_initialized:
                try:
                    # Convert function arguments to MCP format
                    mcp_args = {}
                    if args:
                        # Assume first arg is self, second is the main parameter
                        if len(args) > 1:
                            mcp_args["data"] = args[1]
                    mcp_args.update(kwargs)
                    
                    result = await mcp_utils.call_tool(tool_name, mcp_args)
                    if result["success"]:
                        return result["data"]
                except Exception as e:
                    logger.warning(f"MCP tool call failed, using fallback: {e}")
            
            # Fallback to original function or provided fallback
            if fallback_func:
                return await fallback_func(*args, **kwargs)
            else:
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def mcp_required(func):
    """Decorator that requires MCP to be initialized"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not mcp_utils.is_initialized:
            raise RuntimeError(f"Function {func.__name__} requires MCP to be initialized")
        return await func(*args, **kwargs)
    return wrapper


# Global MCP utils instance
mcp_utils = MCPUtils()
