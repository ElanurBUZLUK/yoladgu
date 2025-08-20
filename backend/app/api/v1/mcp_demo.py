from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List
import logging
import os
from app.mcp.server_simple import TOOLS, call_tool, list_tools

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/system", tags=["mcp-demo"])

# MCP Demo flag
USE_MCP_DEMO = os.getenv("USE_MCP_DEMO", "true").lower() == "true"


@router.get("/mcp-tools", status_code=status.HTTP_200_OK)
async def list_mcp_tools():
    """List available MCP tools for demo purposes."""
    if not USE_MCP_DEMO:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MCP demo is disabled"
        )
    
    try:
        # Use the list_tools function from simple server
        return list_tools()
    except Exception as e:
        logger.error(f"Error listing MCP tools: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list MCP tools: {str(e)}"
        )


@router.post("/mcp-call", status_code=status.HTTP_200_OK)
async def demo_mcp_call(tool_name: str, arguments: Dict[str, Any]):
    """Demo MCP tool call (for testing purposes)."""
    if not USE_MCP_DEMO:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MCP demo is disabled"
        )
    
    try:
        # Validate tool exists
        if tool_name not in TOOLS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Tool '{tool_name}' not found. Available tools: {list(TOOLS.keys())}"
            )
        
        # Call the actual tool using simple server
        result = await call_tool(tool_name, arguments)
        
        logger.info(f"MCP_DEMO_CALL tool={tool_name} arguments={arguments}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in MCP demo call: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute MCP demo call: {str(e)}"
        )


@router.get("/mcp-status", status_code=status.HTTP_200_OK)
async def get_mcp_status():
    """Get MCP demo status and configuration."""
    return {
        "success": True,
        "data": {
            "demo_enabled": USE_MCP_DEMO,
            "available_tools": list(TOOLS.keys()),
            "total_tools": len(TOOLS),
            "version": "1.0.0",
            "description": "Educational MCP Demo Server"
        }
    }
