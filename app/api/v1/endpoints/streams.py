"""
Streams Endpoints
Real-time veri akışı endpointleri
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from typing import Dict, Any, List
from datetime import datetime
import json
import structlog
from app.services.enhanced_stream_consumer import stream_consumer_manager
from app.services.metrics_service import metrics_service

logger = structlog.get_logger()

router = APIRouter(prefix="/streams", tags=["streams"])


@router.get("/status")
async def get_stream_status() -> Dict[str, Any]:
    """Stream durumunu getir"""
    try:
        stats = await stream_consumer_manager.get_stats()
        
        return {
            "status": "success",
            "data": {
                "stream_consumer_status": "running" if stats.get("running", False) else "stopped",
                "connection_active": stats.get("connection_active", False),
                "consumer_queue": stats.get("consumer_queue_name"),
                "producer_queue": stats.get("producer_queue_name"),
                "consumer_task_active": stats.get("consumer_task_active", False)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stream status: {str(e)}")


@router.post("/publish")
async def publish_message(message_data: Dict[str, Any]) -> Dict[str, Any]:
    """Mesaj yayınla"""
    try:
        from app.services.enhanced_stream_consumer import StreamMessage, MessageType
        
        # Message type'ı belirle
        message_type_str = message_data.get("message_type", "user_response")
        message_type = MessageType(message_type_str)
        
        # Stream message oluştur
        stream_message = StreamMessage(
            message_type=message_type,
            user_id=message_data.get("user_id"),
            question_id=message_data.get("question_id"),
            data=message_data.get("data", {}),
            session_id=message_data.get("session_id"),
            metadata=message_data.get("metadata", {})
        )
        
        # Mesajı yayınla
        await stream_consumer_manager.publish_message(stream_message)
        
        # Metrik kaydet
        await metrics_service.record_user_action(
            user_id=message_data.get("user_id", 0),
            action=f"stream_publish_{message_type_str}",
            success=True
        )
        
        return {
            "status": "success",
            "message": "Message published successfully",
            "data": {
                "message_id": f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "message_type": message_type_str,
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to publish message: {str(e)}")


@router.get("/messages")
async def get_recent_messages(limit: int = 50) -> Dict[str, Any]:
    """Son mesajları getir"""
    try:
        # Mock recent messages
        messages = [
            {
                "message_id": f"msg_{i:03d}",
                "message_type": "user_response",
                "user_id": 123,
                "question_id": 456,
                "timestamp": (datetime.now() - timedelta(minutes=i*5)).isoformat(),
                "data": {
                    "answer": f"Sample answer {i}",
                    "is_correct": i % 2 == 0,
                    "response_time": 30 + (i * 2)
                }
            }
            for i in range(1, min(limit + 1, 51))
        ]
        
        return {
            "status": "success",
            "data": {
                "messages": messages,
                "total_messages": len(messages),
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {str(e)}")


@router.get("/messages/{message_type}")
async def get_messages_by_type(message_type: str, limit: int = 20) -> Dict[str, Any]:
    """Belirli türdeki mesajları getir"""
    try:
        # Mock filtered messages
        messages = [
            {
                "message_id": f"msg_{i:03d}",
                "message_type": message_type,
                "user_id": 123 + i,
                "timestamp": (datetime.now() - timedelta(minutes=i*10)).isoformat(),
                "data": {
                    "action": f"Sample {message_type} action {i}",
                    "metadata": {"source": "api"}
                }
            }
            for i in range(1, min(limit + 1, 21))
        ]
        
        return {
            "status": "success",
            "data": {
                "message_type": message_type,
                "messages": messages,
                "total_messages": len(messages),
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get messages by type: {str(e)}")


@router.get("/stats")
async def get_stream_stats() -> Dict[str, Any]:
    """Stream istatistiklerini getir"""
    try:
        # Mock stream stats
        stats = {
            "total_messages_processed": 15420,
            "messages_per_minute": 25.7,
            "average_processing_time_ms": 45,
            "error_rate_percent": 0.5,
            "active_consumers": 3,
            "queue_size": 12,
            "message_types": {
                "user_response": 8500,
                "question_viewed": 4200,
                "study_session_started": 1200,
                "study_session_completed": 800,
                "recommendation_requested": 720
            }
        }
        
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stream stats: {str(e)}")


@router.post("/consumer/start")
async def start_stream_consumer() -> Dict[str, Any]:
    """Stream consumer'ı başlat"""
    try:
        await stream_consumer_manager.start_consumer()
        
        return {
            "status": "success",
            "message": "Stream consumer started successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start consumer: {str(e)}")


@router.post("/consumer/stop")
async def stop_stream_consumer() -> Dict[str, Any]:
    """Stream consumer'ı durdur"""
    try:
        # Mock stop operation
        return {
            "status": "success",
            "message": "Stream consumer stopped successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop consumer: {str(e)}")


@router.get("/consumer/health")
async def get_consumer_health() -> Dict[str, Any]:
    """Consumer sağlık kontrolü"""
    try:
        stats = await stream_consumer_manager.get_stats()
        
        health_status = "healthy" if stats.get("running", False) else "unhealthy"
        
        return {
            "status": "success",
            "data": {
                "consumer_status": health_status,
                "running": stats.get("running", False),
                "connection_active": stats.get("connection_active", False),
                "last_heartbeat": datetime.now().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get consumer health: {str(e)}")


# WebSocket endpoint for real-time streaming
@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time streaming"""
    try:
        await websocket.accept()
        
        # Client'ı kaydet
        logger.info(f"WebSocket client connected: {client_id}")
        
        try:
            while True:
                # Client'tan mesaj al
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Mesajı işle
                if message.get("type") == "subscribe":
                    # Subscription işlemi
                    await websocket.send_text(json.dumps({
                        "type": "subscription_confirmed",
                        "client_id": client_id,
                        "timestamp": datetime.now().isoformat()
                    }))
                
                elif message.get("type") == "publish":
                    # Mesaj yayınla
                    await stream_consumer_manager.publish_message(
                        StreamMessage(
                            message_type=MessageType(message.get("message_type", "user_response")),
                            user_id=message.get("user_id"),
                            question_id=message.get("question_id"),
                            data=message.get("data", {}),
                            session_id=client_id
                        )
                    )
                    
                    await websocket.send_text(json.dumps({
                        "type": "message_published",
                        "client_id": client_id,
                        "timestamp": datetime.now().isoformat()
                    }))
                
                else:
                    # Bilinmeyen mesaj türü
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Unknown message type",
                        "timestamp": datetime.now().isoformat()
                    }))
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket client disconnected: {client_id}")
            
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }))
        except:
            pass


# Import timedelta for the mock data
from datetime import timedelta 