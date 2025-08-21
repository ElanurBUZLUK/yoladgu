import asyncio
import json
import logging
import sys
import time
from typing import Dict, Any, List, Optional
from mcp.server import FastMCP
from mcp.types import Tool, CallToolResult, TextContent

from app.database import database_manager
from app.services.math_recommend_service import math_recommend_service
from app.services.english_cloze_service import english_cloze_service

logger = logging.getLogger(__name__)

server = FastMCP("education-mcp")

# Demo MCP Tools - Only 2 tools for showcase
TOOLS = {
    "english_cloze.generate": {
        "args_schema": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string", "description": "Student ID"},
                "num_recent_errors": {"type": "integer", "default": 5, "description": "Number of recent errors to consider"},
                "difficulty_level": {"type": "integer", "description": "Target difficulty level (1-5)"},
                "question_type": {"type": "string", "default": "cloze", "description": "Question type"}
            },
            "required": ["student_id"]
        },
        "result_schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "content": {"type": "string"},
                    "question_type": {"type": "string"},
                    "difficulty_level": {"type": "integer"},
                    "correct_answer": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}}
                }
            }
        }
    },
    "math.recommend": {
        "args_schema": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string", "description": "Student ID"},
                "limit": {"type": "integer", "default": 10, "description": "Number of recommendations"},
                "difficulty_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Difficulty range [min, max]"
                }
            },
            "required": ["student_id"]
        },
        "result_schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "content": {"type": "string"},
                    "difficulty_level": {"type": "integer"},
                    "recommendation_score": {"type": "number"}
                }
            }
        }
    }
}


@server.tool()
async def english_cloze_generate(
    student_id: str,
    num_recent_errors: int = 5,
    difficulty_level: Optional[int] = None,
    question_type: str = "cloze"
) -> CallToolResult:
    """Generate English cloze questions based on student's recent errors."""
    start_time = time.time()
    
    try:
        async with database_manager.get_session() as db_session:
            questions = await english_cloze_service.generate_cloze_questions(
                session=db_session,
                user_id=student_id,
                num_questions=1,
                last_n_errors=num_recent_errors
            )
            
            # Convert to serializable format
            result = [q.to_dict() for q in questions]
            
            latency_ms = int((time.time() - start_time) * 1000)
            logger.info(f"MCP_CALL tool=english_cloze_generate latency_ms={latency_ms} ok=true")
            
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
            )
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"MCP_CALL tool=english_cloze_generate latency_ms={latency_ms} ok=false error={str(e)}")
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))],
            isError=True
        )


@server.tool()
async def math_recommend(
    student_id: str,
    limit: int = 10,
    difficulty_range: Optional[List[int]] = None
) -> CallToolResult:
    """Recommend math questions based on student's level and performance."""
    start_time = time.time()
    
    try:
        async with database_manager.get_session() as db_session:
            recommendations = await math_recommend_service.recommend_questions(
                user_id=student_id,
                session=db_session,
                limit=limit
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            logger.info(f"MCP_CALL tool=math_recommend latency_ms={latency_ms} ok=true")
            
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(recommendations, ensure_ascii=False))]
            )
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"MCP_CALL tool=math_recommend latency_ms={latency_ms} ok=false error={str(e)}")
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))],
            isError=True
        )


if __name__ == "__main__":
    # Configure logging for the MCP server
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting MCP server...")
    asyncio.run(server.run_stdio())
    logger.info("MCP server stopped.")
