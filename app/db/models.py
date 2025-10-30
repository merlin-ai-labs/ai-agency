"""Database models using SQLModel (SQLAlchemy + Pydantic).

TODO:
- Add indexes for performance
- Add constraints and validations
- Add audit fields (created_by, updated_by)
- Add soft delete support
"""

from datetime import datetime
from enum import Enum
from typing import Any

from sqlmodel import JSON, Column, Field, SQLModel


class RunStatus(str, Enum):
    """Status of a flow run."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class FlowType(str, Enum):
    """Supported agent flow types."""

    WEATHER = "weather"
    GITHUB = "github"
    SLACK = "slack"
    MATURITY_ASSESSMENT = "maturity_assessment"
    USECASE_GROOMING = "usecase_grooming"
    INVOICE_MANAGER = "invoice_manager"


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

    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(unique=True, index=True)
    tenant_id: str = Field(index=True)
    flow_name: str = Field(index=True)  # e.g., "maturity_assessment"
    status: RunStatus = Field(default=RunStatus.QUEUED, index=True)

    # Input/Output
    input_data: dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    output_data: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    error_message: str | None = None

    # Artifacts
    artifact_urls: list[str] = Field(default=[], sa_column=Column(JSON))

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None


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

    id: int | None = Field(default=None, primary_key=True)
    tenant_id: str = Field(index=True)
    document_id: str = Field(index=True)

    # Content
    content: str
    # embedding: List[float]  # TODO: Use pgvector type

    # Metadata (renamed to avoid SQLAlchemy conflict with reserved 'metadata' attribute)
    chunk_metadata: dict[str, Any] = Field(default={}, sa_column=Column(JSON))

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

    id: int | None = Field(default=None, primary_key=True)
    tenant_id: str = Field(unique=True, index=True)
    name: str
    settings: dict[str, Any] = Field(default={}, sa_column=Column(JSON))

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class WeatherApiCall(SQLModel, table=True):
    """Store weather API call history for auditing and caching.

    Records every call to the OpenWeatherMap API including request parameters,
    response data, and performance metrics.

    TODO:
    - Add indexes on (tenant_id, location, created_at) for efficient caching lookups
    - Add indexes on (success, created_at) for error tracking
    """

    __tablename__ = "weather_api_calls"

    id: int | None = Field(default=None, primary_key=True)
    tenant_id: str = Field(index=True)

    # Request parameters
    location: str = Field(index=True)  # City name or coordinates
    units: str = "metric"  # metric, imperial, standard

    # Response data
    temperature: float | None = None
    feels_like: float | None = None
    weather_condition: str | None = None  # e.g., "Clear", "Clouds"
    weather_description: str | None = None  # e.g., "clear sky"
    humidity: int | None = None
    wind_speed: float | None = None

    # Full API response stored as JSON
    response_data: dict[str, Any] = Field(default={}, sa_column=Column(JSON))

    # Call metadata
    success: bool = Field(default=True, index=True)
    error_message: str | None = None
    api_call_ms: int | None = None  # API latency in milliseconds

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


class Conversation(SQLModel, table=True):
    """Conversation for agent chat history.

    Stores conversation metadata for multi-turn agent interactions.
    Each conversation can have multiple messages.
    """

    __tablename__ = "conversations"

    id: int | None = Field(default=None, primary_key=True)
    conversation_id: str = Field(unique=True, index=True)  # UUID
    tenant_id: str = Field(index=True)
    flow_type: str = Field(index=True)  # e.g., "weather", "github", "slack"
    flow_metadata: dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Message(SQLModel, table=True):
    """Message in a conversation.

    Stores individual messages in a conversation, including user messages,
    assistant responses, and tool calls/results.
    """

    __tablename__ = "messages"

    id: int | None = Field(default=None, primary_key=True)
    conversation_id: str = Field(index=True)
    tenant_id: str = Field(index=True)
    flow_type: str = Field(index=True)  # Denormalized for fast queries
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    tool_calls: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    message_metadata: dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)


# TODO: Add more models as needed:
# - User (for authentication)
# - ApiKey (for API authentication)
# - AuditLog (for tracking changes)
