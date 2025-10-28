"""Structured logging configuration using structlog.

TODO:
- Configure processors for JSON output in production
- Add request ID tracking
- Add GCP Cloud Logging integration
"""

import logging
import sys

import structlog

from app.config import settings


def configure_logging():
    """
    Configure structlog for the application.

    TODO:
    - Use JSON renderer for production
    - Add context processors (timestamp, logger name, etc.)
    - Configure log levels from settings
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()  # TODO: Use JSONRenderer in production
            if settings.environment == "development"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Initialize logging on import
configure_logging()
