"""Base module for Alembic migrations and database session management.

Import all models here so Alembic can detect them for autogeneration.

Uses psycopg ConnectionPool for PostgreSQL connections.
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
_pool: "ConnectionPool | None" = None  # Type hint as string to avoid import at module level


def get_connection_pool():
    """Get or create psycopg connection pool for PostgreSQL.

    Returns:
        ConnectionPool for PostgreSQL, None for SQLite.
    """
    global _pool

    # Only create pool for PostgreSQL, not SQLite
    if "sqlite" in settings.database_url:
        return None

    if _pool is None:
        from psycopg_pool import ConnectionPool

        # Strip SQLAlchemy driver suffix
        db_url = settings.database_url.replace("+psycopg", "").replace("+asyncpg", "")

        logger.info("Creating psycopg ConnectionPool for main application")

        _pool = ConnectionPool(
            conninfo=db_url,
            min_size=2,
            max_size=20,
            timeout=30,
        )

        logger.info("Successfully created main application ConnectionPool")

    return _pool


def get_engine() -> Engine:
    """Get or create database engine.

    For PostgreSQL: Uses psycopg ConnectionPool with SQLAlchemy's creator pattern.
    For SQLite: Uses standard SQLAlchemy engine for testing.

    Returns:
        Engine: SQLModel database engine
    """
    global _engine

    if _engine is None:
        pool = get_connection_pool()

        if pool is not None:
            # PostgreSQL: Use psycopg ConnectionPool with SQLAlchemy
            logger.info("Creating SQLAlchemy engine with psycopg ConnectionPool")

            def getconn():
                """Get connection from psycopg pool."""
                return pool.getconn()

            _engine = create_engine(
                "postgresql+psycopg://",  # Specify psycopg3 dialect, connection via creator
                creator=getconn,
                echo=settings.environment == "development",
            )
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
    "get_connection_pool",
    "get_engine",
    "get_session",
    "create_db_and_tables",
]
