"""Database models using SQLModel (SQLAlchemy + Pydantic).

TODO:
- Add indexes for performance
- Add constraints and validations
- Add audit fields (created_by, updated_by)
- Add soft delete support
"""

from sqlmodel import SQLModel, Field, Column, JSON
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class RunStatus(str, Enum):
    """Status of a flow run."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Run(SQLModel, table=True):
    """
    Flow execution run.

    Represents a single execution of a flow (maturity_assessment or usecase_grooming).
    The execution loop polls this table for queued runs.

    TODO:
    - Add indexes on (status, created_at) for efficient polling
    - Add indexes on (tenant_id, flow_name) for filtering
    - Add foreign key to tenant table (if multi-tenant)
    """

    __tablename__ = "runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(unique=True, index=True)
    tenant_id: str = Field(index=True)
    flow_name: str = Field(index=True)  # e.g., "maturity_assessment"
    status: RunStatus = Field(default=RunStatus.QUEUED, index=True)

    # Input/Output
    input_data: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    output_data: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    error_message: Optional[str] = None

    # Artifacts
    artifact_urls: list[str] = Field(default=[], sa_column=Column(JSON))

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class DocumentChunk(SQLModel, table=True):
    """
    Document chunk for RAG.

    Stores text chunks with embeddings for semantic search using pgvector.

    TODO:
    - Add pgvector extension: CREATE EXTENSION vector;
    - Add vector column: embedding vector(1536) for OpenAI or vector(768) for others
    - Add index: CREATE INDEX ON document_chunks USING ivfflat (embedding vector_cosine_ops);
    - Add metadata JSONB for flexible filtering
    """

    __tablename__ = "document_chunks"

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(index=True)
    document_id: str = Field(index=True)

    # Content
    content: str
    # embedding: List[float]  # TODO: Use pgvector type

    # Metadata (renamed to avoid SQLAlchemy conflict with reserved 'metadata' attribute)
    chunk_metadata: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Tenant(SQLModel, table=True):
    """
    Tenant for multi-tenancy.

    TODO:
    - Add subscription/billing fields
    - Add settings and configurations
    - Add API key management
    """

    __tablename__ = "tenants"

    id: Optional[int] = Field(default=None, primary_key=True)
    tenant_id: str = Field(unique=True, index=True)
    name: str
    settings: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# TODO: Add more models as needed:
# - User (for authentication)
# - ApiKey (for API authentication)
# - AuditLog (for tracking changes)
