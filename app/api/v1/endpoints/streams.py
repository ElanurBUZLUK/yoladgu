"""
Redis Streams API endpoints for ML model updates
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import structlog
from app.services.enhanced_stream_consumer import stream_consumer_manager, MessageType

logger = structlog.get_logger()

router = APIRouter()

class StudentResponseMessage(BaseModel):
    student_id: int
    question_id: int
    is_correct: bool
    response_time: Optional[float] = 0
    difficulty_level: Optional[int] = 1

class RecommendationFeedbackMessage(BaseModel):
    student_id: int
    question_id: int
    feedback_score: float
    context: Optional[Dict[str, Any]] = {}

class SystemEventMessage(BaseModel):
    event_type: str
    data: Dict[str, Any]

class StreamMessage(BaseModel):
    type: str
    data: Dict[str, Any]

@router.post("/student-response")
async def publish_student_response(message: StudentResponseMessage):
    """Publish student response for ML model updates"""
    try:
        message_id = await stream_consumer_manager.publish_message(
            MessageType.STUDENT_RESPONSE,
            message.model_dump()
        )
        
        return {
            "status": "success",
            "message_id": message_id,
            "message_type": "student_response"
        }
        
    except Exception as e:
        logger.error("student_response_publish_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to publish message: {str(e)}")

@router.post("/recommendation-feedback")
async def publish_recommendation_feedback(message: RecommendationFeedbackMessage):
    """Publish recommendation feedback for bandit model updates"""
    try:
        message_id = await stream_consumer_manager.publish_message(
            MessageType.RECOMMENDATION_FEEDBACK,
            message.model_dump()
        )
        
        return {
            "status": "success",
            "message_id": message_id,
            "message_type": "recommendation_feedback"
        }
        
    except Exception as e:
        logger.error("recommendation_feedback_publish_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to publish message: {str(e)}")

@router.post("/bandit-update")
async def trigger_bandit_update(action: str = "retrain"):
    """Trigger bandit model update"""
    try:
        message_id = await stream_consumer_manager.publish_message(
            MessageType.BANDIT_UPDATE,
            {"action": action}
        )
        
        return {
            "status": "success",
            "message_id": message_id,
            "action": action
        }
        
    except Exception as e:
        logger.error("bandit_update_publish_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to publish message: {str(e)}")

@router.post("/collaborative-update")
async def trigger_collaborative_update(action: str = "train"):
    """Trigger collaborative filtering update"""
    try:
        message_id = await stream_consumer_manager.publish_message(
            MessageType.COLLABORATIVE_UPDATE,
            {"action": action}
        )
        
        return {
            "status": "success",
            "message_id": message_id,
            "action": action
        }
        
    except Exception as e:
        logger.error("collaborative_update_publish_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to publish message: {str(e)}")

@router.post("/embedding-update")
async def trigger_embedding_update(action: str = "cache_refresh", model_type: Optional[str] = None):
    """Trigger embedding service update"""
    try:
        data = {"action": action}
        if model_type:
            data["model_type"] = model_type
            
        message_id = await stream_consumer_manager.publish_message(
            MessageType.EMBEDDING_UPDATE,
            data
        )
        
        return {
            "status": "success",
            "message_id": message_id,
            "action": action
        }
        
    except Exception as e:
        logger.error("embedding_update_publish_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to publish message: {str(e)}")

@router.post("/system-event")
async def publish_system_event(message: SystemEventMessage):
    """Publish system event"""
    try:
        message_id = await stream_consumer_manager.publish_message(
            MessageType.SYSTEM_EVENT,
            message.model_dump()
        )
        
        return {
            "status": "success",
            "message_id": message_id,
            "event_type": message.event_type
        }
        
    except Exception as e:
        logger.error("system_event_publish_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to publish message: {str(e)}")

@router.post("/custom-message")
async def publish_custom_message(message: StreamMessage):
    """Publish custom stream message"""
    try:
        message_type = MessageType(message.type)
        message_id = await stream_consumer_manager.publish_message(
            message_type,
            message.data
        )
        
        return {
            "status": "success",
            "message_id": message_id,
            "message_type": message.type
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid message type: {message.type}")
    except Exception as e:
        logger.error("custom_message_publish_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to publish message: {str(e)}")

@router.get("/metrics")
async def get_stream_metrics():
    """Get stream consumer metrics"""
    try:
        metrics = stream_consumer_manager.get_metrics()
        
        return {
            "status": "success",
            "metrics": metrics,
            "timestamp": metrics.get('last_activity', 0)
        }
        
    except Exception as e:
        logger.error("stream_metrics_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@router.get("/health")
async def get_stream_health():
    """Get stream consumer health status"""
    try:
        metrics = stream_consumer_manager.get_metrics()
        
        # Determine health based on last activity
        import time
        current_time = time.time()
        last_activity = metrics.get('last_activity', 0)
        time_since_activity = current_time - last_activity
        
        is_healthy = (
            stream_consumer_manager.running and 
            time_since_activity < 300  # Less than 5 minutes since last activity
        )
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "running": stream_consumer_manager.running,
            "time_since_last_activity": time_since_activity,
            "consumer_group": stream_consumer_manager.consumer_group,
            "consumer_name": stream_consumer_manager.consumer_name
        }
        
    except Exception as e:
        logger.error("stream_health_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get health status: {str(e)}")

@router.get("/dlq/messages")
async def get_dlq_messages(limit: int = 10):
    """Get messages from Dead Letter Queue"""
    try:
        dlq_messages = stream_consumer_manager.redis_client.xrange(
            stream_consumer_manager.dlq_stream,
            min='-',
            max='+',
            count=limit
        )
        
        formatted_messages = []
        for message_id, fields in dlq_messages:
            decoded_fields = {k.decode(): v.decode() for k, v in fields.items()}
            formatted_messages.append({
                "dlq_message_id": message_id.decode(),
                "original_message_id": decoded_fields.get('original_message_id'),
                "error": decoded_fields.get('error'),
                "dlq_timestamp": decoded_fields.get('dlq_timestamp'),
                "fields": decoded_fields
            })
        
        return {
            "status": "success",
            "dlq_messages": formatted_messages,
            "count": len(formatted_messages)
        }
        
    except Exception as e:
        logger.error("dlq_messages_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get DLQ messages: {str(e)}")

@router.delete("/dlq/messages/{message_id}")
async def delete_dlq_message(message_id: str):
    """Delete a message from DLQ"""
    try:
        deleted_count = stream_consumer_manager.redis_client.xdel(
            stream_consumer_manager.dlq_stream,
            message_id
        )
        
        if deleted_count > 0:
            return {
                "status": "success",
                "message": f"Deleted DLQ message {message_id}"
            }
        else:
            raise HTTPException(status_code=404, detail=f"DLQ message {message_id} not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("dlq_delete_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete DLQ message: {str(e)}")

@router.post("/dlq/reprocess/{message_id}")
async def reprocess_dlq_message(message_id: str):
    """Reprocess a message from DLQ"""
    try:
        # Get message from DLQ
        dlq_messages = stream_consumer_manager.redis_client.xrange(
            stream_consumer_manager.dlq_stream,
            min=message_id,
            max=message_id,
            count=1
        )
        
        if not dlq_messages:
            raise HTTPException(status_code=404, detail=f"DLQ message {message_id} not found")
        
        # Extract original message data
        _, fields = dlq_messages[0]
        decoded_fields = {k.decode(): v.decode() for k, v in fields.items()}
        
        # Republish to main stream with reset retry count
        republish_fields = {
            'type': decoded_fields.get('type'),
            'data': decoded_fields.get('data'),
            'timestamp': decoded_fields.get('timestamp'),
            'retry_count': '0'  # Reset retry count
        }
        
        new_message_id = stream_consumer_manager.redis_client.xadd(
            stream_consumer_manager.main_stream,
            republish_fields
        )
        
        # Delete from DLQ
        stream_consumer_manager.redis_client.xdel(
            stream_consumer_manager.dlq_stream,
            message_id
        )
        
        return {
            "status": "success",
            "message": f"Reprocessed DLQ message {message_id}",
            "new_message_id": new_message_id.decode()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("dlq_reprocess_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to reprocess DLQ message: {str(e)}")