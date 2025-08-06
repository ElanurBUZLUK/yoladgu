"""
Enhanced Stream Consumer Service
Real-time data processing için gelişmiş stream consumer
"""

import asyncio
import json
import structlog
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import aio_pika
from aio_pika import connect_robust, Message, DeliveryMode
from app.core.config import settings
from app.services.recommendation_service import recommendation_service
from app.services.vector_store_service import vector_store_service
from app.services.enhanced_embedding_service import enhanced_embedding_service

logger = structlog.get_logger()


class MessageType(Enum):
    """Message türleri"""
    USER_RESPONSE = "user_response"
    QUESTION_VIEWED = "question_viewed"
    STUDY_SESSION_STARTED = "study_session_started"
    STUDY_SESSION_COMPLETED = "study_session_completed"
    RECOMMENDATION_REQUESTED = "recommendation_requested"
    EMBEDDING_COMPUTED = "embedding_computed"
    MODEL_UPDATED = "model_updated"
    SYSTEM_HEALTH = "system_health"
    USER_FEEDBACK = "user_feedback"
    ERROR = "error"


@dataclass
class StreamMessage:
    """Stream message data class"""
    message_type: MessageType
    user_id: Optional[int] = None
    question_id: Optional[int] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MessageProcessor:
    """Message processing için base class"""
    
    def __init__(self):
        self.processors: Dict[MessageType, Callable] = {}
        self._register_processors()
    
    def _register_processors(self):
        """Message processor'ları kaydet"""
        self.processors[MessageType.USER_RESPONSE] = self._process_user_response
        self.processors[MessageType.QUESTION_VIEWED] = self._process_question_viewed
        self.processors[MessageType.STUDY_SESSION_STARTED] = self._process_study_session_started
        self.processors[MessageType.STUDY_SESSION_COMPLETED] = self._process_study_session_completed
        self.processors[MessageType.RECOMMENDATION_REQUESTED] = self._process_recommendation_requested
        self.processors[MessageType.EMBEDDING_COMPUTED] = self._process_embedding_computed
        self.processors[MessageType.MODEL_UPDATED] = self._process_model_updated
        self.processors[MessageType.SYSTEM_HEALTH] = self._process_system_health
        self.processors[MessageType.USER_FEEDBACK] = self._process_user_feedback
        self.processors[MessageType.ERROR] = self._process_error
    
    async def process_message(self, message: StreamMessage) -> bool:
        """Message'ı işle"""
        try:
            processor = self.processors.get(message.message_type)
            if processor:
                await processor(message)
                logger.info("message_processed", 
                           message_type=message.message_type.value,
                           user_id=message.user_id)
                return True
            else:
                logger.warning("unknown_message_type", 
                              message_type=message.message_type.value)
                return False
                
        except Exception as e:
            logger.error("message_processing_error", 
                        message_type=message.message_type.value,
                        error=str(e))
            return False
    
    async def _process_user_response(self, message: StreamMessage):
        """Kullanıcı cevabını işle"""
        try:
            if not message.user_id or not message.question_id:
                return
            
            # Recommendation service'e gönder
            await recommendation_service.process_student_response(
                db=None,  # TODO: Get DB session
                student_id=message.user_id,
                question_id=message.question_id,
                answer=message.data.get("answer", ""),
                is_correct=message.data.get("is_correct", False),
                response_time=message.data.get("response_time", 0.0),
                feedback=message.data.get("feedback"),
                request_id=message.session_id
            )
            
        except Exception as e:
            logger.error("process_user_response_error", error=str(e))
            raise
    
    async def _process_question_viewed(self, message: StreamMessage):
        """Soru görüntüleme işle"""
        try:
            if not message.user_id or not message.question_id:
                return
            
            # Question view tracking
            logger.info("question_viewed", 
                       user_id=message.user_id,
                       question_id=message.question_id)
            
        except Exception as e:
            logger.error("process_question_viewed_error", error=str(e))
            raise
    
    async def _process_study_session_started(self, message: StreamMessage):
        """Çalışma oturumu başlatma işle"""
        try:
            if not message.user_id:
                return
            
            # Study session tracking
            logger.info("study_session_started", 
                       user_id=message.user_id,
                       session_id=message.session_id)
            
        except Exception as e:
            logger.error("process_study_session_started_error", error=str(e))
            raise
    
    async def _process_study_session_completed(self, message: StreamMessage):
        """Çalışma oturumu tamamlama işle"""
        try:
            if not message.user_id:
                return
            
            # Study session completion tracking
            logger.info("study_session_completed", 
                       user_id=message.user_id,
                       session_id=message.session_id,
                       duration=message.data.get("duration", 0))
            
        except Exception as e:
            logger.error("process_study_session_completed_error", error=str(e))
            raise
    
    async def _process_recommendation_requested(self, message: StreamMessage):
        """Öneri isteği işle"""
        try:
            if not message.user_id:
                return
            
            # Recommendation request tracking
            logger.info("recommendation_requested", 
                       user_id=message.user_id,
                       subject_id=message.data.get("subject_id"),
                       count=message.data.get("count", 10))
            
        except Exception as e:
            logger.error("process_recommendation_requested_error", error=str(e))
            raise
    
    async def _process_embedding_computed(self, message: StreamMessage):
        """Embedding hesaplama işle"""
        try:
            if not message.data or "text" not in message.data:
                return
            
            # Embedding computation tracking
            text = message.data["text"]
            embedding = await enhanced_embedding_service.compute_embedding_cached(text)
            
            logger.info("embedding_computed", 
                       text_length=len(text),
                       embedding_dim=len(embedding))
            
        except Exception as e:
            logger.error("process_embedding_computed_error", error=str(e))
            raise
    
    async def _process_model_updated(self, message: StreamMessage):
        """Model güncelleme işle"""
        try:
            # Model update tracking
            logger.info("model_updated", 
                       model_name=message.data.get("model_name"),
                       update_type=message.data.get("update_type"))
            
        except Exception as e:
            logger.error("process_model_updated_error", error=str(e))
            raise
    
    async def _process_system_health(self, message: StreamMessage):
        """Sistem sağlık işle"""
        try:
            # System health tracking
            logger.info("system_health", 
                       status=message.data.get("status"),
                       metrics=message.data.get("metrics", {}))
            
        except Exception as e:
            logger.error("process_system_health_error", error=str(e))
            raise
    
    async def _process_user_feedback(self, message: StreamMessage):
        """Kullanıcı geri bildirimi işle"""
        try:
            if not message.user_id or not message.question_id:
                return
            
            # User feedback tracking
            logger.info("user_feedback", 
                       user_id=message.user_id,
                       question_id=message.question_id,
                       feedback_type=message.data.get("feedback_type"),
                       rating=message.data.get("rating"))
            
        except Exception as e:
            logger.error("process_user_feedback_error", error=str(e))
            raise
    
    async def _process_error(self, message: StreamMessage):
        """Hata işle"""
        try:
            # Error tracking
            logger.error("stream_error", 
                        error_type=message.data.get("error_type"),
                        error_message=message.data.get("error_message"),
                        stack_trace=message.data.get("stack_trace"))
            
        except Exception as e:
            logger.error("process_error_error", error=str(e))
            raise


class StreamConsumerManager:
    """Stream consumer manager"""
    
    def __init__(self):
        self.connection = None
        self.channel = None
        self.processor = MessageProcessor()
        self.consuming = False
        self.stats = {
            "processed_messages": 0,
            "failed_messages": 0,
            "dlq_messages": 0,
            "start_time": None
        }
    
    async def initialize(self):
        """Kafka ve/veya RabbitMQ bağlantı kur"""
        try:
            # RabbitMQ connection
            self.connection = await connect_robust(
                settings.RABBITMQ_URL or "amqp://guest:guest@localhost/"
            )
            self.channel = await self.connection.channel()
            
            # Setup queues
            await self._setup_queues()
            
            logger.info("stream_consumer_initialized")
            
        except Exception as e:
            logger.error("stream_consumer_initialization_error", error=str(e))
            raise
    
    async def _setup_queues(self):
        """Queue'ları kur"""
        try:
            # Main queues
            queues = [
                "user_answers",
                "feedback", 
                "new_questions",
                "recommendation_requests",
                "embedding_requests",
                "model_updates"
            ]
            
            # Declare queues
            for queue_name in queues:
                await self.channel.declare_queue(queue_name, durable=True)
            
            # DLQ (Dead Letter Queue)
            await self.channel.declare_queue("dlq", durable=True)
            
            # Exchange for routing
            await self.channel.declare_exchange("yoladgu_events", "topic", durable=True)
            
            logger.info("queues_setup_completed", queues=queues)
            
        except Exception as e:
            logger.error("setup_queues_error", error=str(e))
            raise
    
    async def start_consumer(self):
        """Tüketici loop'u: user_answers, feedback, new_questions topic'leri"""
        try:
            self.consuming = True
            self.stats["start_time"] = datetime.now()
            
            # Start consuming from all queues
            queues = ["user_answers", "feedback", "new_questions"]
            
            for queue_name in queues:
                queue = await self.channel.declare_queue(queue_name, durable=True)
                await queue.consume(self._consume_messages)
            
            logger.info("consumer_started", queues=queues)
            
            # Keep consumer running
            while self.consuming:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error("start_consumer_error", error=str(e))
            raise
    
    async def _consume_messages(self, message):
        """Her event için handler fonksiyonları"""
        try:
            async with message.process():
                # Parse message
                stream_message = await self._parse_message(message)
                
                # Process message
                success = await self.processor.process_message(stream_message)
                
                if success:
                    self.stats["processed_messages"] += 1
                else:
                    self.stats["failed_messages"] += 1
                    # Send to DLQ
                    await self._send_to_dlq(message.body, "processing_failed")
                
        except Exception as e:
            logger.error("consume_messages_error", error=str(e))
            self.stats["failed_messages"] += 1
            # Send to DLQ
            await self._send_to_dlq(message.body, str(e))
    
    async def _parse_message(self, message) -> StreamMessage:
        """Message'ı parse et"""
        try:
            body = message.body.decode('utf-8')
            data = json.loads(body)
            
            return StreamMessage(
                message_type=MessageType(data.get("type", "ERROR")),
                user_id=data.get("user_id"),
                question_id=data.get("question_id"),
                data=data.get("data", {}),
                timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
                session_id=data.get("session_id"),
                metadata=data.get("metadata", {})
            )
            
        except Exception as e:
            logger.error("parse_message_error", error=str(e))
            return StreamMessage(
                message_type=MessageType.ERROR,
                data={"error": str(e)}
            )
    
    async def _send_to_dlq(self, message_body: bytes, reason: str):
        """DLQ: işlenemeyen mesajları dead-letter queue'ya gönder"""
        try:
            dlq_queue = await self.channel.declare_queue("dlq", durable=True)
            
            dlq_message = {
                "original_message": message_body.decode('utf-8'),
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            }
            
            await self.channel.default_exchange.publish(
                Message(
                    body=json.dumps(dlq_message).encode('utf-8'),
                    delivery_mode=DeliveryMode.PERSISTENT
                ),
                routing_key="dlq"
            )
            
            self.stats["dlq_messages"] += 1
            logger.warning("message_sent_to_dlq", reason=reason)
            
        except Exception as e:
            logger.error("send_to_dlq_error", error=str(e))
    
    async def publish_message(self, message: StreamMessage):
        """Message yayınla"""
        try:
            message_data = {
                "type": message.message_type.value,
                "user_id": message.user_id,
                "question_id": message.question_id,
                "data": message.data or {},
                "timestamp": datetime.now().isoformat(),
                "session_id": message.session_id,
                "metadata": message.metadata or {}
            }
            
            # Publish to appropriate queue based on message type
            queue_name = self._get_queue_for_message_type(message.message_type)
            
            await self.channel.default_exchange.publish(
                Message(
                    body=json.dumps(message_data).encode('utf-8'),
                    delivery_mode=DeliveryMode.PERSISTENT
                ),
                routing_key=queue_name
            )
            
            logger.info("message_published", 
                       queue=queue_name,
                       message_type=message.message_type.value)
            
        except Exception as e:
            logger.error("publish_message_error", error=str(e))
            raise
    
    def _get_queue_for_message_type(self, message_type: MessageType) -> str:
        """Message type'a göre queue belirle"""
        queue_mapping = {
            MessageType.USER_RESPONSE: "user_answers",
            MessageType.USER_FEEDBACK: "feedback",
            MessageType.QUESTION_VIEWED: "new_questions",
            MessageType.RECOMMENDATION_REQUESTED: "recommendation_requests",
            MessageType.EMBEDDING_COMPUTED: "embedding_requests",
            MessageType.MODEL_UPDATED: "model_updates"
        }
        return queue_mapping.get(message_type, "user_answers")
    
    async def stop_consumer(self):
        """Consumer'ı durdur"""
        try:
            self.consuming = False
            await self._cleanup_consumer()
            logger.info("consumer_stopped")
            
        except Exception as e:
            logger.error("stop_consumer_error", error=str(e))
    
    async def _cleanup_consumer(self):
        """Consumer cleanup"""
        try:
            if self.channel:
                await self.channel.close()
            if self.connection:
                await self.connection.close()
                
        except Exception as e:
            logger.error("cleanup_consumer_error", error=str(e))
    
    async def get_stats(self) -> Dict[str, Any]:
        """Metrics emit (işlenen, hatalı mesaj sayıları)"""
        try:
            uptime = None
            if self.stats["start_time"]:
                uptime = (datetime.now() - self.stats["start_time"]).total_seconds()
            
            return {
                "processed_messages": self.stats["processed_messages"],
                "failed_messages": self.stats["failed_messages"],
                "dlq_messages": self.stats["dlq_messages"],
                "uptime_seconds": uptime,
                "is_consuming": self.consuming
            }
            
        except Exception as e:
            logger.error("get_stats_error", error=str(e))
            return {}


# Global instance
stream_consumer_manager = StreamConsumerManager() 