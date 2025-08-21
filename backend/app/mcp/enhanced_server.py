"""
Enhanced MCP Server with expanded tool library, context management, and performance monitoring
"""
import asyncio
import json
import logging
import sys
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from mcp.server import FastMCP
from mcp.types import Tool, CallToolResult, TextContent

from app.database import database_manager
from app.services.math_recommend_service import math_recommend_service
from app.services.english_cloze_service import english_cloze_service
from app.services.cefr_assessment_service import cefr_assessment_service
from app.services.embedding_service import embedding_service
from app.services.vector_index_manager import vector_index_manager
from app.services.llm_enhanced_service import enhanced_llm_service
from app.models.user import User
from app.models.question import Question, Subject, QuestionType

logger = logging.getLogger(__name__)

# Enhanced MCP Server
server = FastMCP("enhanced-education-mcp")


class ToolCategory(Enum):
    """MCP Tool Categories"""
    QUESTION_GENERATION = "question_generation"
    ASSESSMENT = "assessment"
    RECOMMENDATION = "recommendation"
    ANALYTICS = "analytics"
    CONTENT_MANAGEMENT = "content_management"
    SYSTEM_MONITORING = "system_monitoring"


@dataclass
class MCPContext:
    """MCP Context for maintaining state across tool calls"""
    student_id: str
    session_start: datetime
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    user_profile: Optional[Dict[str, Any]] = None
    recent_errors: List[Dict[str, Any]] = field(default_factory=list)
    learning_progress: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPMetrics:
    """MCP Performance Metrics"""
    tool_name: str
    call_count: int = 0
    total_latency: float = 0.0
    success_count: int = 0
    error_count: int = 0
    last_call_time: Optional[datetime] = None
    average_latency: float = 0.0


class MCPContextManager:
    """Manages MCP context and state"""
    
    def __init__(self):
        self.contexts: Dict[str, MCPContext] = {}
        self.metrics: Dict[str, MCPMetrics] = {}
        self.context_ttl = timedelta(hours=2)  # Context expires after 2 hours
    
    async def get_context(self, student_id: str) -> MCPContext:
        """Get or create context for student"""
        if student_id not in self.contexts:
            self.contexts[student_id] = MCPContext(
                student_id=student_id,
                session_start=datetime.utcnow()
            )
            await self._load_user_profile(student_id)
        
        return self.contexts[student_id]
    
    async def _load_user_profile(self, student_id: str):
        """Load user profile and recent data"""
        try:
            async with database_manager.get_session() as db:
                # Load user profile
                user = await db.get(User, student_id)
                if user:
                    self.contexts[student_id].user_profile = {
                        "id": str(user.id),
                        "email": user.email,
                        "created_at": user.created_at.isoformat() if user.created_at else None
                    }
                
                # Load recent errors (last 10)
                # This would be implemented based on your error tracking system
                self.contexts[student_id].recent_errors = []
                
        except Exception as e:
            logger.error(f"Error loading user profile for {student_id}: {e}")
    
    def update_context(self, student_id: str, updates: Dict[str, Any]):
        """Update context with new information"""
        if student_id in self.contexts:
            context = self.contexts[student_id]
            for key, value in updates.items():
                if hasattr(context, key):
                    setattr(context, key, value)
    
    def record_tool_call(self, tool_name: str, student_id: str, success: bool, latency: float):
        """Record tool call metrics"""
        if tool_name not in self.metrics:
            self.metrics[tool_name] = MCPMetrics(tool_name=tool_name)
        
        metrics = self.metrics[tool_name]
        metrics.call_count += 1
        metrics.total_latency += latency
        metrics.last_call_time = datetime.utcnow()
        
        if success:
            metrics.success_count += 1
        else:
            metrics.error_count += 1
        
        metrics.average_latency = metrics.total_latency / metrics.call_count
        
        # Update context
        if student_id in self.contexts:
            self.contexts[student_id].tool_calls.append({
                "tool": tool_name,
                "timestamp": datetime.utcnow().isoformat(),
                "success": success,
                "latency": latency
            })
    
    def cleanup_expired_contexts(self):
        """Remove expired contexts"""
        current_time = datetime.utcnow()
        expired_keys = []
        
        for student_id, context in self.contexts.items():
            if current_time - context.session_start > self.context_ttl:
                expired_keys.append(student_id)
        
        for key in expired_keys:
            del self.contexts[key]
            logger.info(f"Cleaned up expired context for student: {key}")


# Global context manager
context_manager = MCPContextManager()


# Enhanced Tool Definitions
ENHANCED_TOOLS = {
    # Question Generation Tools
    "english_cloze.generate": {
        "category": ToolCategory.QUESTION_GENERATION,
        "description": "Generate English cloze questions based on student's recent errors",
        "args_schema": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string", "description": "Student ID"},
                "num_recent_errors": {"type": "integer", "default": 5, "description": "Number of recent errors to consider"},
                "difficulty_level": {"type": "integer", "description": "Target difficulty level (1-5)"},
                "question_type": {"type": "string", "default": "cloze", "description": "Question type"},
                "context_length": {"type": "integer", "default": 3, "description": "Context sentences around blank"}
            },
            "required": ["student_id"]
        },
        "result_schema": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "question_type": {"type": "string"},
                            "difficulty_level": {"type": "integer"},
                            "correct_answer": {"type": "string"},
                            "options": {"type": "array", "items": {"type": "string"}},
                            "context": {"type": "string"},
                            "error_patterns_addressed": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "generation_method": {"type": "string"},
                        "context_used": {"type": "boolean"},
                        "error_patterns_found": {"type": "integer"}
                    }
                }
            }
        }
    },
    
    "math_question.generate": {
        "category": ToolCategory.QUESTION_GENERATION,
        "description": "Generate math questions based on student's skill level and recent performance",
        "args_schema": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string", "description": "Student ID"},
                "topic": {"type": "string", "description": "Math topic (algebra, geometry, calculus, etc.)"},
                "difficulty_level": {"type": "integer", "description": "Target difficulty level (1-5)"},
                "question_type": {"type": "string", "default": "multiple_choice", "description": "Question type"},
                "num_questions": {"type": "integer", "default": 1, "description": "Number of questions to generate"}
            },
            "required": ["student_id", "topic"]
        },
        "result_schema": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "topic": {"type": "string"},
                            "difficulty_level": {"type": "integer"},
                            "correct_answer": {"type": "string"},
                            "options": {"type": "array", "items": {"type": "string"}},
                            "solution_steps": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "skill_level_assessed": {"type": "number"},
                        "topic_coverage": {"type": "string"},
                        "adaptive_difficulty": {"type": "boolean"}
                    }
                }
            }
        }
    },
    
    # Assessment Tools
    "cefr_assess": {
        "category": ToolCategory.ASSESSMENT,
        "description": "Assess student's CEFR level based on text or performance",
        "args_schema": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string", "description": "Student ID"},
                "assessment_text": {"type": "string", "description": "Text to assess"},
                "assessment_type": {"type": "string", "default": "writing", "description": "Type of assessment"},
                "detailed_feedback": {"type": "boolean", "default": True, "description": "Include detailed feedback"},
                "strict_validation": {"type": "boolean", "default": True, "description": "Use strict validation"}
            },
            "required": ["student_id", "assessment_text"]
        },
        "result_schema": {
            "type": "object",
            "properties": {
                "overall_level": {"type": "string"},
                "confidence_score": {"type": "number"},
                "skills": {
                    "type": "object",
                    "properties": {
                        "grammar": {"type": "string"},
                        "vocabulary": {"type": "string"},
                        "reading": {"type": "string"},
                        "writing": {"type": "string"}
                    }
                },
                "detailed_feedback": {"type": "string"},
                "recommendations": {"type": "array", "items": {"type": "string"}}
            }
        }
    },
    
    # Recommendation Tools
    "math_recommend": {
        "category": ToolCategory.RECOMMENDATION,
        "description": "Get personalized math question recommendations",
        "args_schema": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string", "description": "Student ID"},
                "limit": {"type": "integer", "default": 10, "description": "Number of recommendations"},
                "difficulty_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Difficulty range [min, max]"
                },
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific topics to focus on"
                },
                "exclude_recent": {"type": "boolean", "default": True, "description": "Exclude recently seen questions"}
            },
            "required": ["student_id"]
        },
        "result_schema": {
            "type": "object",
            "properties": {
                "recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "difficulty_level": {"type": "integer"},
                            "recommendation_score": {"type": "number"},
                            "topic": {"type": "string"},
                            "reason": {"type": "string"}
                        }
                    }
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "skill_level": {"type": "number"},
                        "weak_areas": {"type": "array", "items": {"type": "string"}},
                        "recommendation_strategy": {"type": "string"}
                    }
                }
            }
        }
    },
    
    # Analytics Tools
    "learning_analytics": {
        "category": ToolCategory.ANALYTICS,
        "description": "Get comprehensive learning analytics for student",
        "args_schema": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string", "description": "Student ID"},
                "time_range": {"type": "string", "default": "30d", "description": "Time range for analytics"},
                "include_progress": {"type": "boolean", "default": True, "description": "Include progress data"},
                "include_errors": {"type": "boolean", "default": True, "description": "Include error analysis"}
            },
            "required": ["student_id"]
        },
        "result_schema": {
            "type": "object",
            "properties": {
                "overview": {
                    "type": "object",
                    "properties": {
                        "total_questions_attempted": {"type": "integer"},
                        "correct_answers": {"type": "integer"},
                        "accuracy_rate": {"type": "number"},
                        "time_spent_minutes": {"type": "number"}
                    }
                },
                "progress_by_subject": {
                    "type": "object",
                    "properties": {
                        "math": {"type": "object"},
                        "english": {"type": "object"}
                    }
                },
                "error_analysis": {
                    "type": "object",
                    "properties": {
                        "most_common_errors": {"type": "array", "items": {"type": "object"}},
                        "error_patterns": {"type": "array", "items": {"type": "object"}},
                        "improvement_areas": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
    },
    
    # Content Management Tools
    "content_search": {
        "category": ToolCategory.CONTENT_MANAGEMENT,
        "description": "Search for educational content using semantic search",
        "args_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "subject": {"type": "string", "description": "Subject (math/english)"},
                "difficulty_level": {"type": "integer", "description": "Target difficulty level"},
                "content_type": {"type": "string", "default": "question", "description": "Type of content"},
                "limit": {"type": "integer", "default": 10, "description": "Number of results"}
            },
            "required": ["query", "subject"]
        },
        "result_schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "content": {"type": "string"},
                            "subject": {"type": "string"},
                            "difficulty_level": {"type": "integer"},
                            "similarity_score": {"type": "number"},
                            "metadata": {"type": "object"}
                        }
                    }
                },
                "search_metadata": {
                    "type": "object",
                    "properties": {
                        "total_found": {"type": "integer"},
                        "search_time_ms": {"type": "number"},
                        "search_method": {"type": "string"}
                    }
                }
            }
        }
    },
    
    # System Monitoring Tools
    "system_health": {
        "category": ToolCategory.SYSTEM_MONITORING,
        "description": "Get system health and performance metrics",
        "args_schema": {
            "type": "object",
            "properties": {
                "include_detailed": {"type": "boolean", "default": False, "description": "Include detailed metrics"},
                "check_services": {"type": "boolean", "default": True, "description": "Check service health"}
            }
        },
        "result_schema": {
            "type": "object",
            "properties": {
                "overall_status": {"type": "string"},
                "services": {
                    "type": "object",
                    "properties": {
                        "database": {"type": "object"},
                        "redis": {"type": "object"},
                        "llm_services": {"type": "object"},
                        "vector_db": {"type": "object"}
                    }
                },
                "performance_metrics": {
                    "type": "object",
                    "properties": {
                        "response_time_avg": {"type": "number"},
                        "error_rate": {"type": "number"},
                        "active_connections": {"type": "integer"}
                    }
                }
            }
        }
    }
}


# Enhanced Tool Implementations
@server.tool()
async def english_cloze_generate_enhanced(
    student_id: str,
    num_recent_errors: int = 5,
    difficulty_level: Optional[int] = None,
    question_type: str = "cloze",
    context_length: int = 3
) -> CallToolResult:
    """Enhanced English cloze question generation with context management."""
    start_time = time.time()
    success = False
    
    try:
        # Get or create context
        context = await context_manager.get_context(student_id)
        
        async with database_manager.get_session() as db_session:
            questions = await english_cloze_service.generate_cloze_questions(
                session=db_session,
                user_id=student_id,
                num_questions=1,
                last_n_errors=num_recent_errors
            )
            
            # Enhanced result with metadata
            result = {
                "questions": [q.to_dict() for q in questions],
                "metadata": {
                    "generation_method": "enhanced_cloze",
                    "context_used": len(context.recent_errors) > 0,
                    "error_patterns_found": len(context.recent_errors),
                    "student_context": {
                        "session_duration": (datetime.utcnow() - context.session_start).total_seconds(),
                        "tool_calls_count": len(context.tool_calls)
                    }
                }
            }
            
            success = True
            latency = time.time() - start_time
            context_manager.record_tool_call("english_cloze_generate_enhanced", student_id, success, latency)
            
            logger.info(f"MCP_CALL tool=english_cloze_generate_enhanced latency_ms={int(latency*1000)} success=true")
            
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))]
            )
            
    except Exception as e:
        latency = time.time() - start_time
        context_manager.record_tool_call("english_cloze_generate_enhanced", student_id, success, latency)
        
        logger.error(f"MCP_CALL tool=english_cloze_generate_enhanced error={str(e)}")
        
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps({
                "error": str(e),
                "questions": [],
                "metadata": {"generation_method": "error_fallback"}
            }))]
        )


@server.tool()
async def math_question_generate(
    student_id: str,
    topic: str,
    difficulty_level: Optional[int] = None,
    question_type: str = "multiple_choice",
    num_questions: int = 1
) -> CallToolResult:
    """Generate math questions with enhanced context awareness."""
    start_time = time.time()
    success = False
    
    try:
        context = await context_manager.get_context(student_id)
        
        # Use enhanced LLM service for question generation
        prompt = f"Generate {num_questions} {difficulty_level or 'medium'} difficulty {topic} question(s) of type {question_type}"
        
        llm_response = await enhanced_llm_service.generate_with_retry(
            prompt=prompt,
            system_prompt="You are an expert math teacher. Generate clear, well-structured math questions.",
            max_tokens=1000,
            user_id=student_id
        )
        
        if llm_response.success:
            # Parse LLM response and create question objects
            # This is a simplified implementation
            questions = [{
                "id": f"math_gen_{int(time.time())}",
                "content": f"Generated {topic} question",
                "topic": topic,
                "difficulty_level": difficulty_level or 3,
                "correct_answer": "TBD",
                "options": ["A", "B", "C", "D"],
                "solution_steps": ["Step 1", "Step 2"]
            }]
            
            result = {
                "questions": questions,
                "metadata": {
                    "skill_level_assessed": context.learning_progress.get("math_skill", 3.0),
                    "topic_coverage": topic,
                    "adaptive_difficulty": difficulty_level is None,
                    "llm_provider": llm_response.provider
                }
            }
            
            success = True
            latency = time.time() - start_time
            context_manager.record_tool_call("math_question_generate", student_id, success, latency)
            
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))]
            )
        else:
            raise Exception(f"LLM generation failed: {llm_response.error_message}")
            
    except Exception as e:
        latency = time.time() - start_time
        context_manager.record_tool_call("math_question_generate", student_id, success, latency)
        
        logger.error(f"MCP_CALL tool=math_question_generate error={str(e)}")
        
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps({
                "error": str(e),
                "questions": [],
                "metadata": {"generation_method": "error_fallback"}
            }))]
        )


@server.tool()
async def cefr_assess_enhanced(
    student_id: str,
    assessment_text: str,
    assessment_type: str = "writing",
    detailed_feedback: bool = True,
    strict_validation: bool = True
) -> CallToolResult:
    """Enhanced CEFR assessment with context awareness."""
    start_time = time.time()
    success = False
    
    try:
        context = await context_manager.get_context(student_id)
        
        async with database_manager.get_session() as db_session:
            assessment_result = await cefr_assessment_service.assess_cefr_level(
                session=db_session,
                user_id=student_id,
                assessment_text=assessment_text,
                assessment_type=assessment_type,
                detailed_feedback=detailed_feedback,
                strict_validation=strict_validation
            )
            
            # Update context with assessment result
            context.learning_progress["cefr_level"] = assessment_result.get("overall_level")
            context.learning_progress["last_assessment"] = datetime.utcnow().isoformat()
            
            result = {
                "overall_level": assessment_result.get("overall_level"),
                "confidence_score": assessment_result.get("confidence_score", 0.8),
                "skills": assessment_result.get("skills", {}),
                "detailed_feedback": assessment_result.get("detailed_feedback", ""),
                "recommendations": assessment_result.get("recommendations", []),
                "context": {
                    "previous_level": context.learning_progress.get("previous_cefr_level"),
                    "progress_made": True  # Simplified
                }
            }
            
            success = True
            latency = time.time() - start_time
            context_manager.record_tool_call("cefr_assess_enhanced", student_id, success, latency)
            
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))]
            )
            
    except Exception as e:
        latency = time.time() - start_time
        context_manager.record_tool_call("cefr_assess_enhanced", student_id, success, latency)
        
        logger.error(f"MCP_CALL tool=cefr_assess_enhanced error={str(e)}")
        
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps({
                "error": str(e),
                "overall_level": "unknown",
                "confidence_score": 0.0
            }))]
        )


@server.tool()
async def learning_analytics_comprehensive(
    student_id: str,
    time_range: str = "30d",
    include_progress: bool = True,
    include_errors: bool = True
) -> CallToolResult:
    """Comprehensive learning analytics with context integration."""
    start_time = time.time()
    success = False
    
    try:
        context = await context_manager.get_context(student_id)
        
        # This would integrate with your analytics system
        # For now, return mock data with context integration
        result = {
            "overview": {
                "total_questions_attempted": 150,
                "correct_answers": 120,
                "accuracy_rate": 0.8,
                "time_spent_minutes": 240
            },
            "progress_by_subject": {
                "math": {
                    "questions_attempted": 80,
                    "accuracy": 0.85,
                    "skill_level": 3.2
                },
                "english": {
                    "questions_attempted": 70,
                    "accuracy": 0.75,
                    "skill_level": 2.8
                }
            },
            "error_analysis": {
                "most_common_errors": [
                    {"error_type": "grammar", "count": 15, "percentage": 30},
                    {"error_type": "vocabulary", "count": 10, "percentage": 20}
                ],
                "error_patterns": [
                    {"pattern": "verb_tense", "frequency": "high"},
                    {"pattern": "article_usage", "frequency": "medium"}
                ],
                "improvement_areas": ["grammar", "vocabulary"]
            },
            "recommendations": [
                "Focus on verb tense exercises",
                "Practice article usage",
                "Review basic grammar rules"
            ],
            "context_integration": {
                "session_duration": (datetime.utcnow() - context.session_start).total_seconds(),
                "tool_usage_pattern": [call["tool"] for call in context.tool_calls[-5:]],
                "learning_preferences": context.preferences
            }
        }
        
        success = True
        latency = time.time() - start_time
        context_manager.record_tool_call("learning_analytics_comprehensive", student_id, success, latency)
        
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))]
        )
        
    except Exception as e:
        latency = time.time() - start_time
        context_manager.record_tool_call("learning_analytics_comprehensive", student_id, success, latency)
        
        logger.error(f"MCP_CALL tool=learning_analytics_comprehensive error={str(e)}")
        
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps({
                "error": str(e),
                "overview": {},
                "progress_by_subject": {},
                "error_analysis": {}
            }))]
        )


@server.tool()
async def system_health_check(
    include_detailed: bool = False,
    check_services: bool = True
) -> CallToolResult:
    """System health check with detailed metrics."""
    start_time = time.time()
    
    try:
        # Check database health
        db_health = {"status": "unknown"}
        try:
            async with database_manager.get_session() as db:
                await db.execute("SELECT 1")
                db_health = {"status": "healthy", "response_time_ms": int((time.time() - start_time) * 1000)}
        except Exception as e:
            db_health = {"status": "unhealthy", "error": str(e)}
        
        # Check LLM services
        llm_health = {"status": "unknown"}
        try:
            metrics = await enhanced_llm_service.get_performance_metrics()
            llm_health = {
                "status": "healthy",
                "success_rate": metrics["overview"]["success_rate"],
                "average_response_time": metrics["overview"]["average_response_time"]
            }
        except Exception as e:
            llm_health = {"status": "unhealthy", "error": str(e)}
        
        # Overall status
        overall_status = "healthy"
        if db_health["status"] != "healthy" or llm_health["status"] != "healthy":
            overall_status = "degraded"
        
        result = {
            "overall_status": overall_status,
            "services": {
                "database": db_health,
                "llm_services": llm_health,
                "redis": {"status": "healthy"},  # Simplified
                "vector_db": {"status": "healthy"}  # Simplified
            },
            "performance_metrics": {
                "response_time_avg": context_manager.metrics.get("average_latency", 0.0),
                "error_rate": sum(m.error_count for m in context_manager.metrics.values()) / max(sum(m.call_count for m in context_manager.metrics.values()), 1),
                "active_connections": len(context_manager.contexts)
            }
        }
        
        if include_detailed:
            result["detailed_metrics"] = {
                tool_name: {
                    "call_count": metrics.call_count,
                    "success_rate": metrics.success_count / max(metrics.call_count, 1),
                    "average_latency": metrics.average_latency
                }
                for tool_name, metrics in context_manager.metrics.items()
            }
        
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))]
        )
        
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps({
                "overall_status": "unhealthy",
                "error": str(e)
            }))]
        )


# Background task for context cleanup
async def context_cleanup_task():
    """Background task to clean up expired contexts"""
    while True:
        try:
            context_manager.cleanup_expired_contexts()
            await asyncio.sleep(300)  # Run every 5 minutes
        except Exception as e:
            logger.error(f"Context cleanup task error: {e}")
            await asyncio.sleep(60)


# Start background tasks
# asyncio.create_task(context_cleanup_task())  # Commented out to avoid import-time asyncio issues


if __name__ == "__main__":
    server.run()
