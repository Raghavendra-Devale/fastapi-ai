import logging
import sys
from typing import Any

import structlog

from app.core.config import get_settings


def setup_logging() -> None:
    """Configure structured logging using structlog.

    Reads configuration settings to format output as readable console text
    for local development, or structured JSON for other environments.
    """
    settings = get_settings()
    log_level = settings.log_level.upper()

    # Map the string log level to standard logging levels
    level = getattr(logging, log_level, logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    is_dev = settings.environment in ("local", "development") or settings.debug

    if is_dev:
        # Dev formatting: colorized console output
        renderer = structlog.dev.ConsoleRenderer()
    else:
        # Production formatting: structured JSON
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging root logger level to handle general filtering
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger instance for the given module name.

    The logger is thread-safe and reusable across modules.
    """
    return structlog.get_logger(name)
