import json
import logging
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

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


async def english_cloze_generate(
    student_id: str,
    num_recent_errors: int = 5,
    difficulty_level: Optional[int] = None,
    question_type: str = "cloze"
) -> Dict[str, Any]:
    """Generate English cloze questions based on student's recent errors."""
    start_time = time.time()
    
    try:
        # Mock implementation for demo
        mock_question = {
            "id": "demo_cloze_001",
            "content": "The student _____ to school every day.",
            "question_type": "cloze",
            "difficulty_level": difficulty_level or 3,
            "correct_answer": "goes",
            "options": ["go", "goes", "going", "went"]
        }
        
        result = [mock_question]
        
        latency_ms = int((time.time() - start_time) * 1000)
        logger.info(f"MCP_CALL tool=english_cloze_generate latency_ms={latency_ms} ok=true")
        
        return {"success": True, "data": result}
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"MCP_CALL tool=english_cloze_generate latency_ms={latency_ms} ok=false error={str(e)}")
        return {"success": False, "error": str(e)}


async def math_recommend(
    student_id: str,
    limit: int = 10,
    difficulty_range: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Recommend math questions based on student's level and performance."""
    start_time = time.time()
    
    try:
        # Mock implementation for demo
        mock_recommendations = [
            {
                "id": "demo_math_001",
                "content": "What is 2 + 3?",
                "difficulty_level": 2,
                "recommendation_score": 0.85
            },
            {
                "id": "demo_math_002", 
                "content": "Solve: 5x + 3 = 18",
                "difficulty_level": 3,
                "recommendation_score": 0.72
            }
        ]
        
        result = mock_recommendations[:limit]
        
        latency_ms = int((time.time() - start_time) * 1000)
        logger.info(f"MCP_CALL tool=math_recommend latency_ms={latency_ms} ok=true")
        
        return {"success": True, "data": result}
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"MCP_CALL tool=math_recommend latency_ms={latency_ms} ok=false error={str(e)}")
        return {"success": False, "error": str(e)}


def list_tools() -> Dict[str, Any]:
    """List available MCP tools."""
    tools_list = []
    for tool_name, tool_config in TOOLS.items():
        tools_list.append({
            "name": tool_name,
            "description": tool_config.get("description", ""),
            "args_schema": tool_config.get("args_schema", {}),
            "result_schema": tool_config.get("result_schema", {})
        })
    
    return {
        "success": True,
        "data": {
            "tools": tools_list,
            "total": len(tools_list),
            "demo_mode": True
        }
    }


async def call_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Call an MCP tool by name."""
    if tool_name == "english_cloze.generate":
        return await english_cloze_generate(**arguments)
    elif tool_name == "math.recommend":
        return await math_recommend(**arguments)
    else:
        return {"success": False, "error": f"Tool '{tool_name}' not found"}
