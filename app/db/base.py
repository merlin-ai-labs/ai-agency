"""Base module for Alembic migrations.

Import all models here so Alembic can detect them for autogeneration.
"""

from sqlmodel import SQLModel

# Import all models so Alembic can detect them
from app.db.models import DocumentChunk, Run, RunStatus, Tenant

# Export Base for Alembic
Base = SQLModel.metadata

# Ensure all models are imported for autogenerate
__all__ = ["Base", "DocumentChunk", "Run", "RunStatus", "Tenant"]
