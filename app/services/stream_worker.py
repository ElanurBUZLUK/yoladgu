"""
Background worker for Redis Streams consumer
Manages the lifecycle of the stream consumer process
"""

import asyncio
import signal
import sys
import time
from contextlib import asynccontextmanager

import structlog
from app.core.config import settings
from app.services.enhanced_stream_consumer import stream_consumer_manager

logger = structlog.get_logger()


class StreamWorker:
    """Background worker for stream processing"""

    def __init__(self):
        self.running = False
        self.consumer_task = None
        self.monitor_task = None

    async def start(self):
        """Start the stream worker"""
        try:
            self.running = True
            logger.info("stream_worker_starting")

            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()

            # Start consumer task
            self.consumer_task = asyncio.create_task(
                stream_consumer_manager.start_consumer()
            )

            # Start monitoring task
            self.monitor_task = asyncio.create_task(self._monitor_consumer())

            # Wait for tasks to complete
            await asyncio.gather(
                self.consumer_task, self.monitor_task, return_exceptions=True
            )

        except Exception as e:
            logger.error("stream_worker_error", error=str(e))
            raise
        finally:
            await self.stop()

    async def stop(self):
        """Stop the stream worker"""
        try:
            self.running = False
            logger.info("stream_worker_stopping")

            # Cancel tasks
            if self.consumer_task and not self.consumer_task.done():
                self.consumer_task.cancel()
                try:
                    await self.consumer_task
                except asyncio.CancelledError:
                    pass

            if self.monitor_task and not self.monitor_task.done():
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass

            # Stop consumer manager
            await stream_consumer_manager._cleanup_consumer()

            logger.info("stream_worker_stopped")

        except Exception as e:
            logger.error("stream_worker_stop_error", error=str(e))

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""

        def signal_handler(signum, frame):
            logger.info("shutdown_signal_received", signal=signum)
            asyncio.create_task(self.stop())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def _monitor_consumer(self):
        """Monitor consumer health and metrics"""
        while self.running:
            try:
                metrics = stream_consumer_manager.get_metrics()

                # Log metrics periodically
                logger.info(
                    "stream_consumer_metrics",
                    messages_processed=metrics["messages_processed"],
                    messages_failed=metrics["messages_failed"],
                    messages_dlq=metrics["messages_dlq"],
                    avg_processing_time=metrics["avg_processing_time"],
                )

                # Check for consumer health
                current_time = time.time()
                last_activity = metrics.get("last_activity", 0)
                time_since_activity = current_time - last_activity

                if time_since_activity > 600:  # 10 minutes without activity
                    logger.warning(
                        "stream_consumer_inactive",
                        time_since_activity=time_since_activity,
                    )

                # Sleep for monitoring interval
                await asyncio.sleep(60)  # Monitor every minute

            except Exception as e:
                logger.error("stream_monitor_error", error=str(e))
                await asyncio.sleep(30)  # Shorter interval on error


# Global worker instance
stream_worker = StreamWorker()


@asynccontextmanager
async def lifespan_context():
    """Context manager for FastAPI lifespan events"""
    # Startup
    try:
        if settings.USE_DLQ:
            logger.info("starting_stream_worker")
            # Start worker in background
            asyncio.create_task(stream_worker.start())
            yield
    except Exception as e:
        logger.error("stream_worker_startup_error", error=str(e))
        yield
    finally:
        # Shutdown
        try:
            if settings.USE_DLQ:
                logger.info("stopping_stream_worker")
                await stream_worker.stop()
        except Exception as e:
            logger.error("stream_worker_shutdown_error", error=str(e))


async def main():
    """Main entry point for standalone worker"""
    try:
        logger.info("starting_standalone_stream_worker")
        await stream_worker.start()
    except KeyboardInterrupt:
        logger.info("keyboard_interrupt_received")
    except Exception as e:
        logger.error("standalone_worker_error", error=str(e))
        sys.exit(1)
    finally:
        logger.info("standalone_worker_finished")


if __name__ == "__main__":
    asyncio.run(main())
