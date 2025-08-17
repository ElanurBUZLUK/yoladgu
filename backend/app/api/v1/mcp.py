from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/mcp", tags=["mcp"])


@router.get("/health")
async def mcp_health():
    """MCP modülü health check"""
    return {"status": "ok", "module": "mcp"}


@router.get("/tools")
async def list_mcp_tools():
    """List available MCP tools"""
    return {
        "tools": [
            "question_generator",
            "answer_evaluator", 
            "pdf_content_reader",
            "question_delivery",
            "analytics"
        ],
        "status": "available"
    }