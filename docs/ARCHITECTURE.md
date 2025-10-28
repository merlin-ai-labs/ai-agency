# AI Consulting Agency - Technical Architecture

**Version:** 0.1.0
**Last Updated:** 2025-10-28
**Status:** Wave 1 Complete - Foundation Ready

---

## Table of Contents

1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Architecture](#component-architecture)
4. [Data Flow Diagrams](#data-flow-diagrams)
5. [Database Schema](#database-schema)
6. [Deployment Architecture](#deployment-architecture)
7. [Technology Stack](#technology-stack)

---

## System Overview

### Purpose
AI-powered consulting platform that provides:
- **Maturity Assessments**: AI-driven organizational capability assessments with rubric-based scoring
- **Use Case Grooming**: Intelligent prioritization and refinement of AI/ML use cases

### Key Characteristics
- **Lean Monorepo**: Single Python service, no microservices complexity
- **Database-Backed Queue**: Simple polling instead of Pub/Sub
- **Multi-Tenant**: Tenant isolation via `tenant_id`
- **Async-First**: FastAPI with async/await throughout
- **RAG-Powered**: pgvector for semantic search

---

## High-Level Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        API_CLIENT[API Client<br/>REST/HTTP]
    end

    subgraph "API Layer - FastAPI Service"
        FASTAPI[FastAPI Application<br/>Port 8080]
        AUTH[Authentication<br/>API Key/JWT]
        ROUTER[API Routers<br/>/assessments, /use-cases]
    end

    subgraph "Business Logic Layer"
        FLOWS[Flow Orchestrator<br/>maturity_assessment<br/>usecase_grooming]
        TOOLS[Tool Registry<br/>5 Core Tools]
        ADAPTERS[LLM Adapters<br/>OpenAI | Vertex AI]
        RAG[RAG Engine<br/>Document Search]
    end

    subgraph "Core Infrastructure"
        BASE[Base Classes<br/>BaseTool, BaseFlow, BaseAdapter]
        DECORATORS[Decorators<br/>@retry, @timeout, @log_execution]
        EXCEPTIONS[Exception Hierarchy<br/>AIAgencyError]
    end

    subgraph "Data Layer"
        REPO[Repositories<br/>CRUD Operations]
        SESSION[SQLAlchemy Sessions<br/>Async Engine]
        MODELS[SQLModel Models<br/>Run, Tenant, DocumentChunk]
    end

    subgraph "External Services"
        POSTGRES[(PostgreSQL + pgvector<br/>Vector DB)]
        OPENAI[OpenAI API<br/>GPT-4, Embeddings]
        VERTEX[Vertex AI<br/>Gemini Pro]
        GCS[Google Cloud Storage<br/>Artifacts]
    end

    subgraph "Background Processing"
        WORKER[Execution Worker<br/>Polls runs table]
        QUEUE[Database Queue<br/>runs.status = 'queued']
    end

    API_CLIENT -->|HTTPS| FASTAPI
    FASTAPI --> AUTH
    AUTH --> ROUTER
    ROUTER --> FLOWS
    FLOWS --> TOOLS
    FLOWS --> ADAPTERS
    FLOWS --> RAG
    TOOLS --> BASE
    ADAPTERS --> BASE
    FLOWS --> BASE
    BASE --> DECORATORS
    BASE --> EXCEPTIONS
    FLOWS --> REPO
    TOOLS --> REPO
    RAG --> REPO
    REPO --> SESSION
    SESSION --> MODELS
    MODELS -->|SQL| POSTGRES
    ADAPTERS -->|API Calls| OPENAI
    ADAPTERS -->|API Calls| VERTEX
    FLOWS -->|Store Artifacts| GCS
    RAG -->|Vector Search| POSTGRES
    WORKER -->|Poll| QUEUE
    QUEUE -->|Update| POSTGRES
    WORKER --> FLOWS

    style FASTAPI fill:#4CAF50
    style FLOWS fill:#2196F3
    style POSTGRES fill:#FF9800
    style WORKER fill:#9C27B0
```

---

## Component Architecture

### 1. API Layer

```mermaid
graph LR
    subgraph "FastAPI Application"
        MAIN[main.py<br/>App Instance]
        MIDDLEWARE[Middleware<br/>CORS, Logging, Error Handling]

        subgraph "API Routes"
            ASSESS_ROUTER[/api/v1/assessments<br/>POST, GET]
            USECASE_ROUTER[/api/v1/use-cases<br/>POST, GET]
            HEALTH_ROUTER[/health<br/>GET]
        end
    end

    MAIN --> MIDDLEWARE
    MIDDLEWARE --> ASSESS_ROUTER
    MIDDLEWARE --> USECASE_ROUTER
    MIDDLEWARE --> HEALTH_ROUTER
```

**Purpose:**
- REST API endpoints for client interaction
- Request validation using Pydantic models
- Authentication and authorization
- OpenAPI/Swagger documentation at `/docs`

---

### 2. Flow Orchestration Layer

```mermaid
graph TB
    subgraph "Flow Orchestrator"
        BASE_FLOW[BaseFlow<br/>Abstract Base Class]

        subgraph "Concrete Flows"
            MATURITY[MaturityAssessmentFlow<br/>7-step assessment]
            GROOMING[UseCaseGroomingFlow<br/>5-step grooming]
        end

        BASE_FLOW -.implements.-> MATURITY
        BASE_FLOW -.implements.-> GROOMING
    end

    subgraph "Flow Execution"
        VALIDATE[validate<br/>Input validation]
        RUN[run<br/>Execute flow steps]
        HANDLE_ERROR[handle_error<br/>Error recovery]
    end

    MATURITY --> VALIDATE
    GROOMING --> VALIDATE
    VALIDATE --> RUN
    RUN --> HANDLE_ERROR

    subgraph "Flow Steps"
        STEP1[Step 1: Extract Context]
        STEP2[Step 2: Generate Prompt]
        STEP3[Step 3: Call LLM]
        STEP4[Step 4: Parse Response]
        STEP5[Step 5: Store Results]
    end

    RUN --> STEP1
    STEP1 --> STEP2
    STEP2 --> STEP3
    STEP3 --> STEP4
    STEP4 --> STEP5
```

**Purpose:**
- Orchestrate multi-step AI workflows
- Manage state transitions (queued → running → completed/failed)
- Coordinate between tools, adapters, and database
- Handle retries and error recovery

---

### 3. Tool System Architecture

```mermaid
graph TB
    subgraph "Tool Registry"
        REGISTRY[ToolRegistry<br/>Singleton Pattern]
        TOOL_MAP[Tool Name → Tool Instance]
    end

    subgraph "Base Tool Infrastructure"
        BASE_TOOL[BaseTool<br/>Abstract Base Class]

        VALIDATE[validate_input<br/>Schema validation]
        EXECUTE[execute<br/>Tool logic]
        RUN[run<br/>Wrapper with logging]
    end

    subgraph "Concrete Tools"
        DOC_SEARCH[DocumentSearchTool<br/>RAG Vector Search]
        RUBRIC_SCORE[RubricScoringTool<br/>Scoring Logic]
        PRIORITIZE[PrioritizationTool<br/>Ranking Algorithm]
        REPORT_GEN[ReportGeneratorTool<br/>Markdown Generation]
        STORAGE[StorageTool<br/>GCS Upload]
    end

    REGISTRY --> TOOL_MAP
    TOOL_MAP --> DOC_SEARCH
    TOOL_MAP --> RUBRIC_SCORE
    TOOL_MAP --> PRIORITIZE
    TOOL_MAP --> REPORT_GEN
    TOOL_MAP --> STORAGE

    BASE_TOOL -.implements.-> DOC_SEARCH
    BASE_TOOL -.implements.-> RUBRIC_SCORE
    BASE_TOOL -.implements.-> PRIORITIZE
    BASE_TOOL -.implements.-> REPORT_GEN
    BASE_TOOL -.implements.-> STORAGE

    BASE_TOOL --> VALIDATE
    BASE_TOOL --> EXECUTE
    BASE_TOOL --> RUN

    style REGISTRY fill:#FFC107
    style BASE_TOOL fill:#2196F3
```

**Purpose:**
- Reusable business logic components
- Version-controlled tool implementations
- Input validation and error handling
- Execution tracking and logging

---

### 4. LLM Adapter Architecture

```mermaid
graph TB
    subgraph "Adapter Layer"
        BASE_ADAPTER[BaseAdapter<br/>Abstract Base Class]

        subgraph "Provider Adapters"
            OPENAI_ADAPTER[OpenAIAdapter<br/>GPT-4, text-embedding-ada-002]
            VERTEX_ADAPTER[VertexAIAdapter<br/>Gemini Pro, text-embedding-004]
        end

        BASE_ADAPTER -.implements.-> OPENAI_ADAPTER
        BASE_ADAPTER -.implements.-> VERTEX_ADAPTER
    end

    subgraph "Core Methods"
        COMPLETE[complete<br/>Chat completion]
        COMPLETE_META[complete_with_metadata<br/>+ token usage, latency]
        CREATE_MSG[create_message<br/>Message formatting]
    end

    OPENAI_ADAPTER --> COMPLETE
    VERTEX_ADAPTER --> COMPLETE
    COMPLETE --> COMPLETE_META
    COMPLETE_META --> CREATE_MSG

    subgraph "Features"
        RETRY[Retry Logic<br/>@retry decorator]
        TIMEOUT[Timeout Handling<br/>@timeout decorator]
        LOGGING[Execution Logging<br/>@log_execution]
        RATE_LIMIT[Rate Limiting<br/>Token bucket]
    end

    COMPLETE --> RETRY
    COMPLETE --> TIMEOUT
    COMPLETE --> LOGGING
    COMPLETE --> RATE_LIMIT

    subgraph "External APIs"
        OPENAI_API[OpenAI API]
        VERTEX_API[Vertex AI API]
    end

    OPENAI_ADAPTER -->|HTTPS| OPENAI_API
    VERTEX_ADAPTER -->|HTTPS| VERTEX_API

    style BASE_ADAPTER fill:#4CAF50
    style OPENAI_ADAPTER fill:#00ACC1
    style VERTEX_ADAPTER fill:#00ACC1
```

**Purpose:**
- Unified interface for multiple LLM providers
- Provider-specific implementation details hidden
- Automatic retries, timeouts, rate limiting
- Token usage and cost tracking

---

### 5. RAG Engine Architecture

```mermaid
graph TB
    subgraph "RAG Pipeline"
        INGESTION[Document Ingestion]
        CHUNKING[Text Chunking<br/>Overlapping windows]
        EMBEDDING[Embedding Generation<br/>OpenAI text-embedding-ada-002]
        STORAGE_VEC[Vector Storage<br/>pgvector]
    end

    subgraph "Retrieval"
        QUERY[User Query]
        QUERY_EMBED[Query Embedding]
        VECTOR_SEARCH[Cosine Similarity Search<br/>IVFFlat Index]
        RERANK[Reranking<br/>Optional]
        RESULTS[Top-K Results]
    end

    subgraph "Generation"
        CONTEXT[Context Assembly]
        PROMPT[Prompt Construction]
        LLM_CALL[LLM Completion]
        RESPONSE[Final Response]
    end

    INGESTION --> CHUNKING
    CHUNKING --> EMBEDDING
    EMBEDDING --> STORAGE_VEC

    QUERY --> QUERY_EMBED
    QUERY_EMBED --> VECTOR_SEARCH
    VECTOR_SEARCH -->|pgvector| POSTGRES[(PostgreSQL<br/>document_chunks)]
    POSTGRES --> RERANK
    RERANK --> RESULTS

    RESULTS --> CONTEXT
    CONTEXT --> PROMPT
    PROMPT --> LLM_CALL
    LLM_CALL --> RESPONSE

    style VECTOR_SEARCH fill:#FF9800
    style POSTGRES fill:#FF9800
```

**Purpose:**
- Semantic document search using vector embeddings
- Efficient similarity search with pgvector's IVFFlat index
- Context-aware LLM responses
- Support for large document collections

---

### 6. Database Architecture

```mermaid
erDiagram
    TENANTS ||--o{ RUNS : has
    TENANTS ||--o{ DOCUMENT_CHUNKS : has

    TENANTS {
        int id PK
        string tenant_id UK "Unique tenant identifier"
        string name
        json settings "Tenant configuration"
        datetime created_at
        datetime updated_at
    }

    RUNS {
        int id PK
        string run_id UK "Unique run identifier"
        string tenant_id FK "Tenant isolation"
        string flow_name "maturity_assessment | usecase_grooming"
        string status "queued | running | completed | failed"
        json input_data "Flow input parameters"
        json output_data "Flow results"
        string error_message "Failure reason"
        json artifact_urls "GCS URLs"
        datetime created_at
        datetime started_at
        datetime completed_at
    }

    DOCUMENT_CHUNKS {
        int id PK
        string tenant_id FK "Tenant isolation"
        string document_id "Source document"
        string content "Text chunk"
        vector-1536 embedding "OpenAI embedding"
        json chunk_metadata "Flexible metadata"
        datetime created_at
    }
```

**Key Indexes:**

```sql
-- Runs table indexes (efficient polling and filtering)
CREATE INDEX ix_runs_status_created_at ON runs (status, created_at);
CREATE INDEX ix_runs_tenant_flow ON runs (tenant_id, flow_name);

-- Document chunks indexes (vector similarity search)
CREATE INDEX ix_document_chunks_embedding ON document_chunks
  USING ivfflat (embedding vector_cosine_ops);
```

**Purpose:**
- **tenants**: Multi-tenant isolation and configuration
- **runs**: Database-backed queue for flow execution
- **document_chunks**: RAG vector storage with pgvector

---

## Data Flow Diagrams

### Maturity Assessment Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Flow
    participant Tools
    participant LLM
    participant DB
    participant GCS

    Client->>API: POST /api/v1/assessments
    API->>DB: Create run (status=queued)
    API-->>Client: 202 Accepted {run_id}

    Note over Flow,DB: Background Worker Polls Queue

    Flow->>DB: Poll for queued runs
    DB-->>Flow: Run with status=queued
    Flow->>DB: Update status=running

    Flow->>Tools: DocumentSearchTool.run()
    Tools->>DB: Vector search (pgvector)
    DB-->>Tools: Relevant context
    Tools-->>Flow: Search results

    Flow->>LLM: Complete with context
    LLM-->>Flow: Assessment response

    Flow->>Tools: RubricScoringTool.run()
    Tools-->>Flow: Scores by dimension

    Flow->>Tools: ReportGeneratorTool.run()
    Tools-->>Flow: Markdown report

    Flow->>Tools: StorageTool.run()
    Tools->>GCS: Upload report
    GCS-->>Tools: GCS URL
    Tools-->>Flow: Artifact URL

    Flow->>DB: Update status=completed, output_data, artifact_urls

    Client->>API: GET /api/v1/assessments/{run_id}
    API->>DB: Query run by run_id
    DB-->>API: Run with results
    API-->>Client: 200 OK {assessment_data}
```

---

### Use Case Grooming Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Flow
    participant Tools
    participant LLM
    participant DB
    participant GCS

    Client->>API: POST /api/v1/use-cases/groom
    API->>DB: Create run (status=queued)
    API-->>Client: 202 Accepted {run_id}

    Flow->>DB: Poll for queued runs
    DB-->>Flow: Run with status=queued
    Flow->>DB: Update status=running

    Flow->>LLM: Extract features from descriptions
    LLM-->>Flow: Structured use cases

    Flow->>Tools: PrioritizationTool.run()
    Tools->>LLM: Score by impact/feasibility
    LLM-->>Tools: Scores
    Tools-->>Flow: Ranked use cases

    Flow->>Tools: ReportGeneratorTool.run()
    Tools-->>Flow: Grooming report

    Flow->>Tools: StorageTool.run()
    Tools->>GCS: Upload report
    GCS-->>Tools: GCS URL
    Tools-->>Flow: Artifact URL

    Flow->>DB: Update status=completed

    Client->>API: GET /api/v1/use-cases/{run_id}
    API->>DB: Query run
    DB-->>API: Run with results
    API-->>Client: 200 OK {groomed_use_cases}
```

---

## Deployment Architecture

### Local Development

```mermaid
graph TB
    subgraph "Developer Machine"
        DOCKER[Docker Compose]

        subgraph "Containers"
            DB_CONTAINER[PostgreSQL + pgvector<br/>Port 5432]
            APP_CONTAINER[FastAPI App<br/>Port 8080<br/>Hot Reload]
        end

        VENV[Python venv<br/>Local execution]
    end

    DOCKER --> DB_CONTAINER
    DOCKER --> APP_CONTAINER
    APP_CONTAINER -->|Database URL| DB_CONTAINER
    VENV -->|Development| APP_CONTAINER
```

---

### GCP Production Deployment

```mermaid
graph TB
    subgraph "GCP Project"
        subgraph "Cloud Run"
            APP_INSTANCE[FastAPI Container<br/>Auto-scaling<br/>0-100 instances]
        end

        subgraph "Cloud SQL"
            POSTGRES[PostgreSQL 15<br/>+ pgvector<br/>Private IP]
            CLOUD_SQL_PROXY[Cloud SQL Proxy<br/>Secure Connection]
        end

        subgraph "Storage"
            GCS_BUCKET[GCS Bucket<br/>Artifacts Storage]
        end

        subgraph "Secrets"
            SECRET_MGR[Secret Manager<br/>API Keys, DB Credentials]
        end

        subgraph "Monitoring"
            CLOUD_LOGGING[Cloud Logging<br/>Structured JSON Logs]
            CLOUD_MONITORING[Cloud Monitoring<br/>Metrics & Alerts]
        end

        subgraph "External"
            OPENAI_EXT[OpenAI API]
            VERTEX_EXT[Vertex AI API]
        end
    end

    subgraph "Client"
        HTTPS_CLIENT[HTTPS Client]
    end

    HTTPS_CLIENT -->|HTTPS| APP_INSTANCE
    APP_INSTANCE --> CLOUD_SQL_PROXY
    CLOUD_SQL_PROXY -->|Private IP| POSTGRES
    APP_INSTANCE -->|API Calls| GCS_BUCKET
    APP_INSTANCE -->|Fetch Secrets| SECRET_MGR
    APP_INSTANCE -->|Logs| CLOUD_LOGGING
    APP_INSTANCE -->|Metrics| CLOUD_MONITORING
    APP_INSTANCE -->|API Calls| OPENAI_EXT
    APP_INSTANCE -->|API Calls| VERTEX_EXT

    style APP_INSTANCE fill:#4CAF50
    style POSTGRES fill:#FF9800
    style SECRET_MGR fill:#F44336
```

**Key Features:**
- **Auto-scaling**: Cloud Run scales 0→100 based on traffic
- **Secure DB Connection**: Cloud SQL Proxy with private IP
- **Secrets Management**: All credentials in Secret Manager
- **Monitoring**: Structured logging + metrics + alerts

---

### Multi-Cloud Database Portability

```mermaid
graph TB
    subgraph "Application"
        APP[FastAPI App]
        CONFIG[Environment Config<br/>DATABASE_URL]
    end

    APP --> CONFIG

    subgraph "Database Providers"
        GCP[GCP Cloud SQL<br/>PostgreSQL]
        AWS[AWS RDS<br/>PostgreSQL]
        AZURE[Azure Database<br/>PostgreSQL]
        LOCAL[Local PostgreSQL<br/>Docker]
    end

    CONFIG -.->|Switch via env var| GCP
    CONFIG -.->|Switch via env var| AWS
    CONFIG -.->|Switch via env var| AZURE
    CONFIG -.->|Switch via env var| LOCAL

    style CONFIG fill:#FFC107
```

**Connection Strings:**
```bash
# GCP Cloud SQL
postgresql+psycopg://user:pass@/dbname?host=/cloudsql/project:region:instance

# AWS RDS
postgresql+psycopg://user:pass@rds-instance.region.rds.amazonaws.com:5432/dbname?sslmode=require

# Azure Database
postgresql+psycopg://user@server:pass@server.postgres.database.azure.com:5432/dbname?sslmode=require

# Local Docker
postgresql+psycopg://postgres:postgres@localhost:5432/ai_agency
```

---

## Technology Stack

### Backend Framework
- **FastAPI** 0.109.0+ - Modern async web framework
- **Uvicorn** 0.27.0+ - ASGI server with hot reload
- **Pydantic** 2.5.0+ - Data validation and settings

### Database & ORM
- **PostgreSQL** 15+ - Primary database
- **pgvector** 0.2.4+ - Vector similarity search
- **SQLModel** 0.0.14+ - SQLAlchemy + Pydantic integration
- **Alembic** 1.13.0+ - Database migrations
- **psycopg** 3.1.0+ - PostgreSQL adapter (async support)

### LLM Providers
- **OpenAI** - GPT-4, text-embedding-ada-002
- **Vertex AI** - Gemini Pro, text-embedding-004

### Storage
- **Google Cloud Storage** - Artifact storage

### Development Tools
- **Ruff** - Fast linter & formatter (replaces Black + isort)
- **mypy** - Static type checking
- **pytest** - Testing framework with async support
- **pre-commit** - Git hooks for quality checks

### DevOps
- **Docker** + **Docker Compose** - Containerization
- **GitHub Actions** - CI/CD pipeline
- **Alembic** - Database migration management

### Utilities
- **structlog** 24.1.0+ - Structured logging
- **httpx** 0.26.0+ - Async HTTP client
- **tenacity** 8.2.0+ - Retry logic
- **python-dotenv** 1.0.0+ - Environment management

---

## Core Design Patterns

### 1. Repository Pattern
```python
class BaseRepository(ABC):
    """Abstract base class for data access."""

    @abstractmethod
    async def get_by_id(self, id: int) -> Optional[T]: ...

    @abstractmethod
    async def create(self, obj: T) -> T: ...

    @abstractmethod
    async def update(self, obj: T) -> T: ...

    @abstractmethod
    async def delete(self, id: int) -> bool: ...
```

### 2. Strategy Pattern (LLM Adapters)
```python
class BaseAdapter(ABC):
    """Abstract adapter for LLM providers."""

    @abstractmethod
    async def complete(self, messages: List[Message]) -> str: ...
```

### 3. Template Method Pattern (Flows)
```python
class BaseFlow(ABC):
    """Abstract base class for flows."""

    @abstractmethod
    async def run(self, input_data: Dict) -> Dict: ...

    async def execute(self, run_id: str) -> None:
        """Template method with common flow logic."""
        # 1. Validate
        # 2. Run
        # 3. Handle errors
```

### 4. Registry Pattern (Tools)
```python
class ToolRegistry:
    """Singleton registry for tool instances."""
    _tools: Dict[str, BaseTool] = {}

    @classmethod
    def register(cls, name: str, tool: BaseTool): ...

    @classmethod
    def get(cls, name: str) -> BaseTool: ...
```

### 5. Decorator Pattern (Cross-Cutting Concerns)
```python
@retry(max_attempts=3, backoff_type="exponential")
@timeout(seconds=30)
@log_execution(level="INFO")
@measure_time
async def call_llm(prompt: str) -> str:
    """All decorators compose cleanly."""
    ...
```

---

## Security Architecture

### Authentication & Authorization
```mermaid
graph LR
    CLIENT[Client Request] --> API_KEY[API Key Validation]
    API_KEY --> TENANT[Tenant Resolution]
    TENANT --> AUTHZ[Authorization Check]
    AUTHZ --> ENDPOINT[Protected Endpoint]
```

### Tenant Isolation
- All queries filtered by `tenant_id`
- Row-level security in database
- No cross-tenant data access

### Secrets Management
- Environment variables for development
- Secret Manager for production
- No hardcoded credentials
- Sensitive data redacted in logs

---

## Performance Considerations

### Database Optimization
- **Composite Indexes**: `(status, created_at)` for efficient polling
- **Vector Index**: IVFFlat for fast similarity search
- **Connection Pooling**: SQLAlchemy async pool
- **Query Optimization**: Eager loading, select specific columns

### Caching Strategy
- **In-Memory Cache**: `@cache_result` decorator (MVP)
- **Future**: Redis for distributed caching
- **Embedding Cache**: Avoid re-embedding same content

### Async Operations
- All I/O operations are async (DB, HTTP, file operations)
- Proper use of `asyncio.gather()` for parallel operations
- Non-blocking LLM calls

---

## Monitoring & Observability

### Structured Logging
```python
import structlog

logger = structlog.get_logger()
logger.info(
    "flow_execution_complete",
    run_id=run_id,
    flow_name=flow_name,
    duration_ms=duration,
    status="completed"
)
```

### Metrics to Track
- Request latency (p50, p95, p99)
- LLM token usage and costs
- Database query performance
- Error rates by endpoint
- Queue depth (queued runs)
- Flow execution duration

### Health Checks
```
GET /health
- Database connectivity
- LLM provider availability
- Storage service status
```

---

## Development Workflow

```mermaid
graph LR
    DEV[Local Development] --> COMMIT[Git Commit]
    COMMIT --> HOOKS[Pre-commit Hooks<br/>Ruff, mypy]
    HOOKS --> PUSH[Git Push]
    PUSH --> CI[GitHub Actions CI]
    CI --> TESTS[Run Tests<br/>Coverage >= 70%]
    TESTS --> LINT[Linting & Formatting]
    LINT --> MIGRATIONS[Test Migrations]
    MIGRATIONS --> DEPLOY[Deploy to Cloud Run]
```

---

## Next Steps (Wave 2-6)

### Wave 2: Core Services
- [ ] Database repositories (CRUD operations)
- [ ] Session management (async SQLAlchemy)
- [ ] OpenAI adapter implementation
- [ ] Vertex AI adapter implementation
- [ ] GCS storage integration

### Wave 3: RAG Implementation
- [ ] Document ingestion pipeline
- [ ] Chunking strategy
- [ ] Embedding generation
- [ ] Vector similarity search
- [ ] Context retrieval

### Wave 4: Business Logic
- [ ] 5 core tools implementation
- [ ] Maturity assessment flow
- [ ] Use case grooming flow
- [ ] Background worker for polling

### Wave 5: Quality & Security
- [ ] Comprehensive test suite (80%+ coverage)
- [ ] Security hardening
- [ ] API documentation
- [ ] Performance optimization

### Wave 6: Final Review
- [ ] Code review by code-reviewer agent
- [ ] Production deployment
- [ ] Monitoring setup
- [ ] Documentation finalization

---

## References

- [Deployment Guide](DEPLOYMENT.md) - Multi-cloud deployment instructions
- [Coding Standards](CODING_STANDARDS.md) - Python best practices
- [Wave 1 Review](WAVE1_REVIEW.md) - Foundation implementation review
- [API Documentation](http://localhost:8080/docs) - OpenAPI/Swagger UI

---

**Maintained by:** AI Consulting Agency Development Team
**Questions?** See [README.md](../README.md) or contact the team.
