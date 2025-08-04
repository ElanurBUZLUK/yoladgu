"""
Enhanced Redis Streams Consumer for Yoladgu ML Model Updates
Comprehensive async processing with DLQ handling and monitoring
"""

import asyncio
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

import redis
import structlog
from app.core.config import settings
from app.ml.bandits import EnhancedLinUCBBandit
from app.ml.collaborative_filter import CollaborativeFilterEngine
from app.ml.online_learner import EnhancedOnlineLearner

logger = structlog.get_logger()


class MessageType(Enum):
    """Message types for different ML model updates"""

    STUDENT_RESPONSE = "student_response"
    RECOMMENDATION_FEEDBACK = "recommendation_feedback"
    BANDIT_UPDATE = "bandit_update"
    COLLABORATIVE_UPDATE = "collaborative_update"
    EMBEDDING_UPDATE = "embedding_update"
    SYSTEM_EVENT = "system_event"


class MessageStatus(Enum):
    """Message processing status"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DLQ = "dlq"


@dataclass
class StreamMessage:
    """Structured stream message"""

    id: str
    type: MessageType
    data: Dict[str, Any]
    timestamp: float
    retry_count: int = 0
    status: MessageStatus = MessageStatus.PENDING
    consumer_group: Optional[str] = None
    consumer_name: Optional[str] = None


class StreamConsumerManager:
    """Enhanced Redis Streams consumer with DLQ handling"""

    def __init__(self):
        self.redis_client = redis.Redis.from_url(settings.redis_url)
        self.consumer_group = "yoladgu_ml_processors"
        self.consumer_name = f"consumer_{int(time.time())}"
        self.running = False
        self.message_handlers: Dict[MessageType, Callable] = {}

        # Stream names
        self.main_stream = "ml_updates"
        self.dlq_stream = "ml_updates_dlq"

        # Processing configuration
        self.max_retries = 3
        self.batch_size = 10
        self.block_time = 1000  # milliseconds

        # Initialize ML models
        self._initialize_ml_models()

        # Setup message handlers
        self._setup_message_handlers()

        # Monitoring metrics
        self.metrics = {
            "messages_processed": 0,
            "messages_failed": 0,
            "messages_dlq": 0,
            "processing_times": [],
            "last_activity": time.time(),
        }

    def _initialize_ml_models(self):
        """Initialize ML model instances for updates"""
        try:
            self.online_learner = EnhancedOnlineLearner(self.redis_client)
            self.bandit_model = EnhancedLinUCBBandit(self.redis_client)
            self.collaborative_filter = CollaborativeFilterEngine(self.redis_client)
            logger.info("ml_models_initialized_for_stream_consumer")
        except Exception as e:
            logger.error("ml_models_initialization_error", error=str(e))
            # Don't raise, allow consumer to start anyway

    def _setup_message_handlers(self):
        """Setup message type handlers"""
        self.message_handlers = {
            MessageType.STUDENT_RESPONSE: self._handle_student_response,
            MessageType.RECOMMENDATION_FEEDBACK: self._handle_recommendation_feedback,
            MessageType.BANDIT_UPDATE: self._handle_bandit_update,
            MessageType.COLLABORATIVE_UPDATE: self._handle_collaborative_update,
            MessageType.EMBEDDING_UPDATE: self._handle_embedding_update,
            MessageType.SYSTEM_EVENT: self._handle_system_event,
        }

    async def start_consumer(self):
        """Start the stream consumer with proper setup"""
        try:
            # Create consumer group if not exists
            await self._create_consumer_group()

            self.running = True
            logger.info(
                "stream_consumer_started",
                group=self.consumer_group,
                consumer=self.consumer_name,
            )

            # Main processing loop
            while self.running:
                try:
                    await self._process_batch()
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.error("consumer_loop_error", error=str(e))
                    await asyncio.sleep(1)

        except Exception as e:
            logger.error("consumer_startup_error", error=str(e))
            raise
        finally:
            await self._cleanup_consumer()

    async def _create_consumer_group(self):
        """Create Redis consumer group"""
        try:
            if self.redis_client is None:
                logger.error("redis_client_not_initialized")
                return
            self.redis_client.xgroup_create(
                self.main_stream, self.consumer_group, id="0", mkstream=True
            )
            logger.info("consumer_group_created", stream=self.main_stream)
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info("consumer_group_already_exists", stream=self.main_stream)
            else:
                logger.error("consumer_group_creation_error", error=str(e))

    async def _process_batch(self):
        """Process a batch of messages from streams"""
        try:
            if self.redis_client is None:
                logger.error("redis_client_not_initialized")
                return
            messages: Any = self.redis_client.xreadgroup(
                self.consumer_group,
                self.consumer_name,
                {self.main_stream: ">"},
                count=self.batch_size,
                block=self.block_time,
            )

            if messages and isinstance(messages, list):
                for stream_name, stream_messages in messages:
                    try:
                        if (
                            isinstance(stream_messages, list)
                            and len(stream_messages) > 0
                        ):
                            for message_tuple in stream_messages:
                                if (
                                    isinstance(message_tuple, tuple)
                                    and len(message_tuple) >= 2
                                ):
                                    message_id, fields = message_tuple
                                    if isinstance(message_id, bytes) and isinstance(
                                        fields, dict
                                    ):
                                        await self._process_single_message(
                                            stream_name.decode(),
                                            message_id.decode(),
                                            fields,
                                        )
                    except Exception as e:
                        logger.warning("invalid_stream_messages_format", error=str(e))
                        break  # Break out of the loop on error

        except Exception as e:
            logger.error("batch_processing_error", error=str(e))

    async def _process_single_message(
        self, stream_name: str, message_id: str, fields: Dict
    ):
        """Process a single stream message"""
        start_time = time.time()

        try:
            message = self._parse_message(message_id, fields)
            logger.info(
                "processing_message", message_id=message_id, type=message.type.value
            )

            handler = self.message_handlers.get(message.type)
            if not handler:
                raise ValueError(f"No handler for message type: {message.type}")

            message.status = MessageStatus.PROCESSING
            _result = await handler(message)  # Store result but don't use it

            # Acknowledge successful processing
            if self.redis_client is not None:
                self.redis_client.xack(
                    self.main_stream, self.consumer_group, message_id
                )
            message.status = MessageStatus.COMPLETED

            # Update metrics
            processing_time = time.time() - start_time
            self.metrics["messages_processed"] += 1
            self.metrics["processing_times"].append(processing_time)
            self.metrics["last_activity"] = time.time()

            # Keep only last 100 processing times
            if len(self.metrics["processing_times"]) > 100:
                self.metrics["processing_times"] = self.metrics["processing_times"][
                    -100:
                ]

            logger.info(
                "message_processed_successfully",
                message_id=message_id,
                processing_time=processing_time,
            )

        except Exception as e:
            # Handle processing error
            await self._handle_processing_error(message_id, fields, e)

    def _parse_message(self, message_id: str, fields: Dict) -> StreamMessage:
        """Parse Redis stream message into structured format"""
        try:
            decoded_fields = {k.decode(): v.decode() for k, v in fields.items()}

            message_type = MessageType(decoded_fields.get("type", "system_event"))
            data = json.loads(decoded_fields.get("data", "{}"))
            timestamp = float(decoded_fields.get("timestamp", time.time()))
            retry_count = int(decoded_fields.get("retry_count", 0))

            return StreamMessage(
                id=message_id,
                type=message_type,
                data=data,
                timestamp=timestamp,
                retry_count=retry_count,
                consumer_group=self.consumer_group,
                consumer_name=self.consumer_name,
            )

        except Exception as e:
            logger.error("message_parsing_error", message_id=message_id, error=str(e))
            raise

    async def _handle_student_response(self, message: StreamMessage) -> Dict[str, Any]:
        """Handle student response updates"""
        try:
            data = message.data
            student_id = data["student_id"]
            question_id = data["question_id"]
            is_correct = data["is_correct"]
            response_time = data.get("response_time", 0)
            difficulty = data.get("difficulty_level", 1)

            # Update online learner
            if hasattr(self, "online_learner"):
                user_features = {
                    "accuracy_rate_overall": 0.5,  # Default value
                    "avg_response_time": response_time,
                }
                question_features = {
                    "difficulty_level": difficulty,
                    "avg_success_rate": 0.5,  # Default value
                }

                self.online_learner.learn_from_answer(
                    user_features, question_features, is_correct, response_time
                )

            # Update collaborative filter
            if hasattr(self, "collaborative_filter"):
                rating = 1.0 if is_correct else 0.0
                self.collaborative_filter.add_interaction(
                    student_id, question_id, rating
                )

            logger.info(
                "student_response_processed",
                student_id=student_id,
                question_id=question_id,
                correct=is_correct,
            )

            return {
                "status": "success",
                "models_updated": ["online_learner", "collaborative_filter"],
            }

        except Exception as e:
            logger.error("student_response_handling_error", error=str(e))
            raise

    async def _handle_recommendation_feedback(
        self, message: StreamMessage
    ) -> Dict[str, Any]:
        """Handle recommendation feedback"""
        try:
            data = message.data
            student_id = data["student_id"]
            question_id = data["question_id"]
            feedback_score = data.get("feedback_score", 0.5)
            context = data.get("context", {})

            if hasattr(self, "bandit_model"):
                # Prepare user and question features for bandit update
                user_features = {
                    "accuracy_rate_overall": 0.5,
                    "topic_mastery_math": 0.5,
                }
                question_features = {
                    "difficulty_level": context.get("difficulty", 3),
                    "avg_success_rate": 0.5,
                }

                self.bandit_model.update(
                    question_id,
                    user_features,
                    question_features,
                    feedback_score,
                    context,
                )

            logger.info(
                "recommendation_feedback_processed",
                student_id=student_id,
                question_id=question_id,
                feedback_score=feedback_score,
            )

            return {"status": "success", "models_updated": ["bandit_model"]}

        except Exception as e:
            logger.error("recommendation_feedback_handling_error", error=str(e))
            raise

    async def _handle_bandit_update(self, message: StreamMessage) -> Dict[str, Any]:
        """Handle bandit model updates"""
        try:
            data = message.data
            action = data.get("action", "update")

            logger.info("bandit_update_processed", action=action)
            return {"status": "success", "action": action}

        except Exception as e:
            logger.error("bandit_update_handling_error", error=str(e))
            raise

    async def _handle_collaborative_update(
        self, message: StreamMessage
    ) -> Dict[str, Any]:
        """Handle collaborative filtering updates"""
        try:
            data = message.data
            action = data.get("action", "train")

            if action == "train" and hasattr(self, "collaborative_filter"):
                self.collaborative_filter.train_model()
                logger.info("collaborative_filter_training_completed")

            return {"status": "success", "action": action}

        except Exception as e:
            logger.error("collaborative_update_handling_error", error=str(e))
            raise

    async def _handle_embedding_update(self, message: StreamMessage) -> Dict[str, Any]:
        """Handle embedding service updates"""
        try:
            data = message.data
            action = data.get("action", "cache_refresh")

            logger.info("embedding_update_processed", action=action)
            return {"status": "success", "action": action}

        except Exception as e:
            logger.error("embedding_update_handling_error", error=str(e))
            raise

    async def _handle_system_event(self, message: StreamMessage) -> Dict[str, Any]:
        """Handle system events"""
        try:
            data = message.data
            event_type = data.get("event_type", "unknown")

            logger.info("system_event_processed", event_type=event_type)
            return {"status": "success", "event_type": event_type}

        except Exception as e:
            logger.error("system_event_handling_error", error=str(e))
            raise

    async def _handle_processing_error(
        self, message_id: str, fields: Dict, error: Exception
    ):
        """Handle message processing errors with retry and DLQ logic"""
        try:
            decoded_fields = {k.decode(): v.decode() for k, v in fields.items()}
            retry_count = int(decoded_fields.get("retry_count", 0))

            self.metrics["messages_failed"] += 1

            if retry_count < self.max_retries:
                # Retry message
                updated_fields = decoded_fields.copy()
                updated_fields["retry_count"] = str(retry_count + 1)
                updated_fields["retry_timestamp"] = str(time.time())

                if self.redis_client is not None:
                    self.redis_client.xadd(self.main_stream, updated_fields)
                logger.warning(
                    "message_retried",
                    message_id=message_id,
                    retry_count=retry_count + 1,
                    error=str(error),
                )
            else:
                # Send to DLQ
                dlq_fields = decoded_fields.copy()
                dlq_fields["original_message_id"] = message_id
                dlq_fields["error"] = str(error)
                dlq_fields["dlq_timestamp"] = str(time.time())

                if self.redis_client is not None:
                    self.redis_client.xadd(self.dlq_stream, dlq_fields)
                logger.error(
                    "message_sent_to_dlq", message_id=message_id, error=str(error)
                )
                self.metrics["messages_dlq"] += 1

            # Acknowledge to remove from pending
            if self.redis_client is not None:
                self.redis_client.xack(
                    self.main_stream, self.consumer_group, message_id
                )

        except Exception as dlq_error:
            logger.error(
                "dlq_handling_error", message_id=message_id, error=str(dlq_error)
            )

    async def _cleanup_consumer(self):
        """Cleanup consumer on shutdown"""
        try:
            self.running = False
            logger.info("stream_consumer_stopped")
        except Exception as e:
            logger.error("consumer_cleanup_error", error=str(e))

    async def publish_message(
        self, message_type: MessageType, data: Dict[str, Any]
    ) -> str:
        """Publish a message to the stream"""
        try:
            message_fields = {
                "type": message_type.value,
                "data": json.dumps(data),
                "timestamp": str(time.time()),
                "retry_count": "0",
            }

            if self.redis_client is None:
                raise RuntimeError("Redis client not initialized")
            message_id = self.redis_client.xadd(self.main_stream, message_fields)

            logger.info(
                "message_published", message_id=message_id, type=message_type.value
            )

            return message_id.decode()

        except Exception as e:
            logger.error("message_publish_error", error=str(e))
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get consumer metrics"""
        avg_processing_time = 0
        if self.metrics["processing_times"]:
            avg_processing_time = sum(self.metrics["processing_times"]) / len(
                self.metrics["processing_times"]
            )

        return {
            "messages_processed": self.metrics["messages_processed"],
            "messages_failed": self.metrics["messages_failed"],
            "messages_dlq": self.metrics["messages_dlq"],
            "avg_processing_time": avg_processing_time,
            "last_activity": self.metrics["last_activity"],
            "consumer_group": self.consumer_group,
            "consumer_name": self.consumer_name,
            "running": self.running,
        }


# Global consumer manager instance
stream_consumer_manager = StreamConsumerManager()

if __name__ == "__main__":
    import asyncio

    asyncio.run(stream_consumer_manager.start_consumer())
