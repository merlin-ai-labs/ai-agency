"""Base module for Alembic migrations and database session management.

Import all models here so Alembic can detect them for autogeneration.

Uses SQLAlchemy's native connection pooling with psycopg3 driver for PostgreSQL.
"""

import logging
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from app.config import settings

# Import all models so Alembic can detect them
from app.db.models import DocumentChunk, Run, RunStatus, Tenant, WeatherApiCall

logger = logging.getLogger(__name__)

# Lazy initialization
_engine: Engine | None = None


def get_engine() -> Engine:
    """Get or create database engine.

    For PostgreSQL: Uses SQLAlchemy's native connection pooling.
    For SQLite: Uses standard SQLAlchemy engine for testing.

    Returns:
        Engine: SQLModel database engine
    """
    global _engine

    if _engine is None:
        if "sqlite" not in settings.database_url:
            # PostgreSQL: Use SQLAlchemy's native pool management
            # This works correctly with both Unix sockets (production) and TCP (local proxy)
            logger.info("Creating SQLAlchemy engine with native pool for PostgreSQL")

            _engine = create_engine(
                settings.database_url,  # Already has +psycopg
                pool_size=10,  # Reduced from 20 for Cloud Run limits
                max_overflow=5,  # Additional connections when pool exhausted
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=3600,  # Recycle connections after 1 hour
                pool_timeout=30,  # Wait 30s for available connection
                echo=settings.environment == "development",
            )

            logger.info("Successfully created PostgreSQL engine with pool_size=10, max_overflow=5")
        else:
            # SQLite: Standard engine for local testing
            logger.info("Creating SQLAlchemy engine for SQLite")

            _engine = create_engine(
                settings.database_url,
                echo=settings.environment == "development",
                connect_args={"check_same_thread": False},
            )

    return _engine


def get_session() -> Engine:
    """Get database session (returns engine for use with Session context manager).

    Returns:
        Engine: SQLModel database engine

    Example:
        >>> from app.db.base import get_session
        >>> with Session(get_session()) as session:
        ...     result = session.exec(select(Run)).all()
    """
    return get_engine()


def create_db_and_tables():
    """Create database tables.

    Should be called on application startup to ensure all tables exist.
    """
    SQLModel.metadata.create_all(get_engine())


# Export Base for Alembic
Base = SQLModel.metadata

# Ensure all models are imported for autogenerate
__all__ = [
    "Base",
    "DocumentChunk",
    "Run",
    "RunStatus",
    "Tenant",
    "WeatherApiCall",
    "get_engine",
    "get_session",
    "create_db_and_tables",
]
