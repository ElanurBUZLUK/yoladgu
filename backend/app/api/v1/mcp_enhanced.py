"""
Enhanced MCP API endpoints with context management and performance monitoring
"""
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from datetime import datetime

from app.mcp.enhanced_server import context_manager, ENHANCED_TOOLS
from app.core.error_handling import ErrorHandler, ErrorCode, ErrorSeverity

router = APIRouter(prefix="/api/v1/mcp-enhanced", tags=["MCP Enhanced"])
error_handler = ErrorHandler()


class MCPToolCallRequest(BaseModel):
    tool_name: str = Field(..., description="Name of the MCP tool to call")
    arguments: Dict[str, Any] = Field(..., description="Tool arguments")
    student_id: Optional[str] = Field(None, description="Student ID for context")


class MCPToolCallResponse(BaseModel):
    success: bool = Field(..., description="Whether the tool call was successful")
    result: Dict[str, Any] = Field(..., description="Tool call result")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Call metadata")


class MCPContextResponse(BaseModel):
    student_id: str = Field(..., description="Student ID")
    session_start: datetime = Field(..., description="Session start time")
    tool_calls: List[Dict[str, Any]] = Field(..., description="Recent tool calls")
    user_profile: Optional[Dict[str, Any]] = Field(None, description="User profile")
    learning_progress: Dict[str, Any] = Field(..., description="Learning progress")
    preferences: Dict[str, Any] = Field(..., description="User preferences")


class MCPMetricsResponse(BaseModel):
    tool_name: str = Field(..., description="Tool name")
    call_count: int = Field(..., description="Total call count")
    success_count: int = Field(..., description="Successful calls")
    error_count: int = Field(..., description="Failed calls")
    average_latency: float = Field(..., description="Average response time")
    last_call_time: Optional[datetime] = Field(None, description="Last call time")


class MCPHealthResponse(BaseModel):
    status: str = Field(..., description="Overall MCP status")
    active_contexts: int = Field(..., description="Number of active contexts")
    total_tool_calls: int = Field(..., description="Total tool calls")
    success_rate: float = Field(..., description="Overall success rate")
    average_response_time: float = Field(..., description="Average response time")


@router.post("/tools/call", response_model=MCPToolCallResponse)
async def call_mcp_tool(request: MCPToolCallRequest):
    """
    Call an MCP tool with enhanced context management
    """
    try:
        # Validate tool exists
        if request.tool_name not in ENHANCED_TOOLS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool '{request.tool_name}' not found"
            )
        
        # Get tool definition
        tool_def = ENHANCED_TOOLS[request.tool_name]
        
        # Validate required arguments
        required_args = tool_def["args_schema"].get("required", [])
        for arg in required_args:
            if arg not in request.arguments:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required argument: {arg}"
                )
        
        # Call the appropriate tool function
        start_time = datetime.utcnow()
        
        if request.tool_name == "english_cloze.generate":
            from app.mcp.enhanced_server import english_cloze_generate_enhanced
            result = await english_cloze_generate_enhanced(
                student_id=request.arguments["student_id"],
                num_recent_errors=request.arguments.get("num_recent_errors", 5),
                difficulty_level=request.arguments.get("difficulty_level"),
                question_type=request.arguments.get("question_type", "cloze"),
                context_length=request.arguments.get("context_length", 3)
            )
        
        elif request.tool_name == "math_question.generate":
            from app.mcp.enhanced_server import math_question_generate
            result = await math_question_generate(
                student_id=request.arguments["student_id"],
                topic=request.arguments["topic"],
                difficulty_level=request.arguments.get("difficulty_level"),
                question_type=request.arguments.get("question_type", "multiple_choice"),
                num_questions=request.arguments.get("num_questions", 1)
            )
        
        elif request.tool_name == "cefr_assess":
            from app.mcp.enhanced_server import cefr_assess_enhanced
            result = await cefr_assess_enhanced(
                student_id=request.arguments["student_id"],
                assessment_text=request.arguments["assessment_text"],
                assessment_type=request.arguments.get("assessment_type", "writing"),
                detailed_feedback=request.arguments.get("detailed_feedback", True),
                strict_validation=request.arguments.get("strict_validation", True)
            )
        
        elif request.tool_name == "learning_analytics":
            from app.mcp.enhanced_server import learning_analytics_comprehensive
            result = await learning_analytics_comprehensive(
                student_id=request.arguments["student_id"],
                time_range=request.arguments.get("time_range", "30d"),
                include_progress=request.arguments.get("include_progress", True),
                include_errors=request.arguments.get("include_errors", True)
            )
        
        elif request.tool_name == "system_health":
            from app.mcp.enhanced_server import system_health_check
            result = await system_health_check(
                include_detailed=request.arguments.get("include_detailed", False),
                check_services=request.arguments.get("check_services", True)
            )
        
        else:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=f"Tool '{request.tool_name}' not yet implemented"
            )
        
        # Parse result
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Extract content from result
        content = result.content[0].text if result.content else "{}"
        result_data = content if isinstance(content, dict) else {}
        
        return MCPToolCallResponse(
            success=True,
            result=result_data,
            metadata={
                "tool_name": request.tool_name,
                "response_time": response_time,
                "timestamp": datetime.utcnow().isoformat(),
                "student_id": request.student_id
            }
        )
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.MCP_TOOL_ERROR,
            message=f"Failed to call MCP tool: {request.tool_name}",
            severity=ErrorSeverity.MEDIUM,
            context={
                "tool_name": request.tool_name,
                "arguments": request.arguments,
                "student_id": request.student_id
            }
        )


@router.get("/tools/list")
async def list_mcp_tools():
    """
    List all available MCP tools with their schemas
    """
    try:
        tools_info = {}
        
        for tool_name, tool_def in ENHANCED_TOOLS.items():
            tools_info[tool_name] = {
                "category": tool_def["category"].value,
                "description": tool_def["description"],
                "args_schema": tool_def["args_schema"],
                "result_schema": tool_def["result_schema"]
            }
        
        return {
            "tools": tools_info,
            "total_count": len(tools_info),
            "categories": [cat.value for cat in tool_def["category"].__class__]
        }
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.MCP_TOOL_ERROR,
            message="Failed to list MCP tools",
            severity=ErrorSeverity.LOW,
            context={}
        )


@router.get("/context/{student_id}", response_model=MCPContextResponse)
async def get_mcp_context(student_id: str):
    """
    Get MCP context for a specific student
    """
    try:
        context = await context_manager.get_context(student_id)
        
        return MCPContextResponse(
            student_id=context.student_id,
            session_start=context.session_start,
            tool_calls=context.tool_calls[-10:],  # Last 10 calls
            user_profile=context.user_profile,
            learning_progress=context.learning_progress,
            preferences=context.preferences
        )
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.MCP_CONTEXT_ERROR,
            message=f"Failed to get MCP context for student: {student_id}",
            severity=ErrorSeverity.MEDIUM,
            context={"student_id": student_id}
        )


@router.put("/context/{student_id}")
async def update_mcp_context(student_id: str, updates: Dict[str, Any]):
    """
    Update MCP context for a specific student
    """
    try:
        context_manager.update_context(student_id, updates)
        
        return {
            "success": True,
            "message": f"Context updated for student: {student_id}",
            "updated_fields": list(updates.keys())
        }
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.MCP_CONTEXT_ERROR,
            message=f"Failed to update MCP context for student: {student_id}",
            severity=ErrorSeverity.MEDIUM,
            context={"student_id": student_id, "updates": updates}
        )


@router.delete("/context/{student_id}")
async def clear_mcp_context(student_id: str):
    """
    Clear MCP context for a specific student
    """
    try:
        if student_id in context_manager.contexts:
            del context_manager.contexts[student_id]
            
        return {
            "success": True,
            "message": f"Context cleared for student: {student_id}"
        }
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.MCP_CONTEXT_ERROR,
            message=f"Failed to clear MCP context for student: {student_id}",
            severity=ErrorSeverity.MEDIUM,
            context={"student_id": student_id}
        )


@router.get("/metrics", response_model=List[MCPMetricsResponse])
async def get_mcp_metrics():
    """
    Get MCP performance metrics for all tools
    """
    try:
        metrics_list = []
        
        for tool_name, metrics in context_manager.metrics.items():
            metrics_list.append(MCPMetricsResponse(
                tool_name=tool_name,
                call_count=metrics.call_count,
                success_count=metrics.success_count,
                error_count=metrics.error_count,
                average_latency=metrics.average_latency,
                last_call_time=metrics.last_call_time
            ))
        
        return metrics_list
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.MCP_METRICS_ERROR,
            message="Failed to get MCP metrics",
            severity=ErrorSeverity.LOW,
            context={}
        )


@router.get("/metrics/{tool_name}", response_model=MCPMetricsResponse)
async def get_mcp_tool_metrics(tool_name: str):
    """
    Get MCP performance metrics for a specific tool
    """
    try:
        if tool_name not in context_manager.metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No metrics found for tool: {tool_name}"
            )
        
        metrics = context_manager.metrics[tool_name]
        
        return MCPMetricsResponse(
            tool_name=tool_name,
            call_count=metrics.call_count,
            success_count=metrics.success_count,
            error_count=metrics.error_count,
            average_latency=metrics.average_latency,
            last_call_time=metrics.last_call_time
        )
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.MCP_METRICS_ERROR,
            message=f"Failed to get metrics for tool: {tool_name}",
            severity=ErrorSeverity.LOW,
            context={"tool_name": tool_name}
        )


@router.get("/health", response_model=MCPHealthResponse)
async def get_mcp_health():
    """
    Get MCP system health status
    """
    try:
        # Calculate overall metrics
        total_calls = sum(m.call_count for m in context_manager.metrics.values())
        total_success = sum(m.success_count for m in context_manager.metrics.values())
        total_errors = sum(m.error_count for m in context_manager.metrics.values())
        
        success_rate = total_success / max(total_calls, 1)
        avg_response_time = sum(m.average_latency for m in context_manager.metrics.values()) / max(len(context_manager.metrics), 1)
        
        # Determine overall status
        status = "healthy"
        if success_rate < 0.8:
            status = "degraded"
        if success_rate < 0.5:
            status = "unhealthy"
        
        return MCPHealthResponse(
            status=status,
            active_contexts=len(context_manager.contexts),
            total_tool_calls=total_calls,
            success_rate=success_rate,
            average_response_time=avg_response_time
        )
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.MCP_HEALTH_ERROR,
            message="Failed to get MCP health status",
            severity=ErrorSeverity.LOW,
            context={}
        )


@router.post("/context/cleanup")
async def trigger_context_cleanup():
    """
    Manually trigger context cleanup
    """
    try:
        before_count = len(context_manager.contexts)
        context_manager.cleanup_expired_contexts()
        after_count = len(context_manager.contexts)
        
        return {
            "success": True,
            "message": "Context cleanup completed",
            "contexts_removed": before_count - after_count,
            "remaining_contexts": after_count
        }
        
    except Exception as e:
        return error_handler.handle_error(
            error=e,
            error_code=ErrorCode.MCP_CONTEXT_ERROR,
            message="Failed to trigger context cleanup",
            severity=ErrorSeverity.MEDIUM,
            context={}
        )
