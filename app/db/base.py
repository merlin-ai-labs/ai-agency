"""Base module for Alembic migrations and database session management.

Import all models here so Alembic can detect them for autogeneration.
"""

from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from app.config import settings

# Import all models so Alembic can detect them
from app.db.models import DocumentChunk, Run, RunStatus, Tenant, WeatherApiCall

# Lazy engine initialization
_engine: Engine | None = None


def get_engine() -> Engine:
    """Get or create database engine.

    Returns:
        Engine: SQLModel database engine
    """
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            echo=settings.environment == "development",
            connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
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
