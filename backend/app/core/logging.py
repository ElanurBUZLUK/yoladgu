import logging
import sys
from typing import Any

import structlog


def init_logging(env: str = "dev") -> None:
    """Initialize structured logging with structlog and stdlib logging.

    In development, prefer human-readable console output.
    In production, emit JSON for log aggregation systems.
    """
    timestamper = structlog.processors.TimeStamper(fmt="iso")
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
    ]

    # Configure stdlib logging first
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    if env == "prod":
        renderer: structlog.typing.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )


def get_logger() -> structlog.stdlib.BoundLogger:  # convenience alias
    logger = structlog.get_logger()
    return logger


