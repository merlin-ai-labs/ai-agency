# Architecture

**Status:** Wave 1 Complete - Foundation deployed
**Last Updated:** 2025-10-28

## Overview

Simple AI consulting platform built as a monorepo with FastAPI, PostgreSQL + pgvector, and Cloud Run.

**What's Live:**
- FastAPI API with stub endpoints
- PostgreSQL 15 with pgvector on Cloud SQL
- Auto-deploy on push to main via GitHub Actions
- Local development with Docker Compose

**What's Not Built Yet:**
- Actual LLM integration (stubs only)
- RAG implementation (stubs only)
- Business logic tools (stubs only)
- Flow orchestration (stubs only)

---

## System Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTPS
       ▼
┌─────────────────────────────┐
│  FastAPI Application        │
│  ┌──────────────────────┐   │
│  │  API Endpoints       │   │
│  │  /runs, /health      │   │
│  └──────────────────────┘   │
│  ┌──────────────────────┐   │
│  │  Flow Stubs          │   │
│  │  (Wave 2)            │   │
│  └──────────────────────┘   │
│  ┌──────────────────────┐   │
│  │  Tool Registry       │   │
│  │  (Stubs)             │   │
│  └──────────────────────┘   │
│  ┌──────────────────────┐   │
│  │  LLM Adapters        │   │
│  │  (Stubs)             │   │
│  └──────────────────────┘   │
│  ┌──────────────────────┐   │
│  │  Database Layer      │   │
│  │  SQLModel + Alembic  │   │
│  └──────────────────────┘   │
└──────────┬──────────────────┘
           │
           ▼
  ┌──────────────────┐
  │   PostgreSQL 15  │
  │   + pgvector     │
  └──────────────────┘
```

---

## Database Schema

Current tables (minimal Wave 1 setup):

### runs
Main execution tracking table (stub implementation for now)

```sql
CREATE TABLE runs (
    id UUID PRIMARY KEY,
    run_id VARCHAR UNIQUE NOT NULL,
    tenant_id VARCHAR NOT NULL,
    flow_name VARCHAR NOT NULL,  -- 'maturity_assessment' | 'usecase_grooming'
    status VARCHAR NOT NULL,       -- 'queued' | 'running' | 'completed' | 'failed'
    input_data JSONB,
    output_data JSONB,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

CREATE INDEX idx_runs_tenant_id ON runs(tenant_id);
CREATE INDEX idx_runs_status ON runs(status);
```

### tenants (placeholder)
```sql
CREATE TABLE tenants (
    id UUID PRIMARY KEY,
    tenant_id VARCHAR UNIQUE NOT NULL,
    name VARCHAR,
    settings JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### document_chunks (pgvector, for Wave 2+)
```sql
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY,
    tenant_id VARCHAR NOT NULL,
    document_id VARCHAR NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(1536),  -- pgvector
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_chunks_tenant ON document_chunks(tenant_id);
-- Vector search index added in Wave 2
```

---

## Infrastructure Mapping

### Local Development

```
Developer Laptop
├── Python venv (FastAPI app)
├── Docker (PostgreSQL + pgvector)
└── .env (secrets)

External:
└── OpenAI API (via OPENAI_API_KEY)
```

**What you need:**
- Docker or Colima
- Python 3.11+
- OpenAI API key in .env

**URLs:**
- API: http://localhost:8080
- DB: postgresql://localhost:5432/ai_agency
- Docs: http://localhost:8080/docs

---

### GitHub Actions CI/CD

```
GitHub Runner
├── PostgreSQL service container (tests)
├── Ruff, mypy, pytest (quality checks)
├── Docker build (create image)
├── Push to Artifact Registry
├── Alembic migrations (Cloud SQL Proxy)
└── Deploy to Cloud Run
```

**Workflows:**
- `.github/workflows/ci.yml` - Tests on every PR/push
- `.github/workflows/deploy.yml` - Deploy on push to main

**Secrets:**
- `GCP_SA_KEY` - Service account for deployment
- `GCP_PROJECT_ID` - merlin-notebook-lm

---

### GCP Production

```
Cloud Run (europe-west1)
├── FastAPI container (auto-scaling 0-100)
├── Cloud SQL Proxy → PostgreSQL 15
├── Secret Manager → OPENAI_API_KEY
└── GCS bucket → artifacts (future)

External:
├── OpenAI API
└── Vertex AI (future)
```

**Services:**
- **Cloud Run**: ai-agency (europe-west1)
- **Cloud SQL**: ai-agency-db (PostgreSQL 15 + pgvector)
- **GCS**: merlin-ai-agency-artifacts-eu
- **Artifact Registry**: europe-west1-docker.pkg.dev/merlin-notebook-lm/ai-agency

**URLs:**
- Service: https://ai-agency-4ebxrg4hdq-ew.a.run.app
- API Docs: https://ai-agency-4ebxrg4hdq-ew.a.run.app/docs
- Health: https://ai-agency-4ebxrg4hdq-ew.a.run.app/health

---

## Technology Stack

### Backend
- **FastAPI** 0.109+ - Web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation

### Database
- **PostgreSQL** 15 - Primary database
- **pgvector** 0.2.4+ - Vector similarity (for Wave 2+)
- **SQLModel** - ORM (SQLAlchemy + Pydantic)
- **Alembic** - Schema migrations
- **psycopg** 3.1+ - Async PostgreSQL adapter

### Cloud (GCP)
- **Cloud Run** - Serverless containers
- **Cloud SQL** - Managed PostgreSQL
- **Secret Manager** - Credentials
- **GCS** - Object storage (future)
- **Artifact Registry** - Container images

### Development
- **Ruff** - Linting + formatting
- **mypy** - Type checking
- **pytest** - Testing
- **Docker Compose** - Local database

### Future (Wave 2+)
- **OpenAI** - GPT-4, embeddings
- **Vertex AI** - Gemini (alternative)
- **LangGraph** - Flow orchestration (maybe)

---

## Code Structure

```
app/
├── api/              # FastAPI routes
│   └── main.py       # /runs endpoints (stubs)
├── core/             # Base classes, exceptions, decorators
│   ├── base.py       # BaseTool, BaseFlow, BaseAdapter
│   ├── exceptions.py # Exception hierarchy
│   ├── decorators.py # @retry, @timeout, @log_execution
│   └── types.py      # Type definitions
├── db/               # Database layer
│   ├── models.py     # SQLModel tables
│   └── base.py       # Session management
├── adapters/         # LLM providers (stubs)
│   ├── llm_factory.py
│   ├── llm_openai.py  # TODO: Wave 2
│   └── llm_vertex.py  # TODO: Wave 2
├── tools/            # Business logic (stubs)
│   └── registry.py   # Tool registration
├── flows/            # Workflows (stubs)
│   ├── maturity_assessment/
│   └── usecase_grooming/
├── rag/              # Vector search (stubs for Wave 2+)
│   ├── ingestion.py
│   └── retriever.py
└── main.py           # FastAPI app entry point
```

---

## Design Patterns

### Base Classes
All tools, flows, and adapters inherit from base classes in `app/core/base.py`:

```python
class BaseTool(ABC):
    @abstractmethod
    async def execute(self, input_data: dict) -> dict:
        pass

class BaseFlow(ABC):
    @abstractmethod
    async def run(self, input_data: dict) -> dict:
        pass

class BaseAdapter(ABC):
    @abstractmethod
    async def generate_completion(self, prompt: str) -> str:
        pass
```

### Exception Hierarchy
All exceptions inherit from `AIAgencyError`:

```python
AIAgencyError
├── DatabaseError
├── LLMError
├── FlowError
├── ToolError
├── ValidationError
├── AuthError
└── StorageError
```

### Async First
Everything uses async/await:
- FastAPI endpoints
- Database operations (SQLModel async)
- LLM API calls (future)
- All business logic

---

## API Endpoints

### Current (Wave 1)

**POST /runs**
Create a flow execution (stub implementation)
```json
{
  "tenant_id": "my-company",
  "flow_name": "maturity_assessment",
  "input_data": {}
}
```
Returns: `{"run_id": "...", "status": "queued"}`

**GET /runs/{run_id}**
Get execution status (stub)
Returns: `{"run_id": "...", "status": "completed", "output_data": {...}}`

**GET /health**
Health check
Returns: `{"status": "ok", "service": "ai-agency"}`

**GET /docs**
Interactive API documentation (Swagger UI)

### Future (Wave 2+)

These will be implemented with actual logic:
- POST /api/v1/assessments - Maturity assessment flow
- GET /api/v1/assessments/{run_id} - Get assessment results
- POST /api/v1/use-cases/groom - Use case grooming flow
- GET /api/v1/use-cases/{run_id} - Get grooming results
- POST /api/v1/documents - Upload documents for RAG
- GET /api/v1/documents/{doc_id} - Retrieve document

---

## Environment Configuration

### Local (.env)
```bash
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/ai_agency
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
LOG_LEVEL=INFO
ENVIRONMENT=development
```

### GitHub Actions (Secrets)
- `GCP_SA_KEY` - Service account JSON
- `GCP_PROJECT_ID` - merlin-notebook-lm

### GCP Production (Secret Manager)
- `OPENAI_API_KEY` - OpenAI API key
- `DATABASE_PASSWORD` - If needed
- `VERTEX_AI_CREDENTIALS` - For Vertex AI (future)

---

## Deployment Flow

```
1. Developer pushes to main
2. GitHub Actions runs:
   ├── Tests (pytest, coverage)
   ├── Linting (ruff)
   ├── Type checking (mypy)
   └── Build Docker image
3. Push image to Artifact Registry
4. Run database migrations (Alembic via Cloud SQL Proxy)
5. Deploy to Cloud Run
6. Run smoke tests
7. Service is live
```

**Auto-scaling:**
- Cloud Run scales from 0 to 100 instances
- Cold start: ~2-3 seconds
- Requests per instance: 80 concurrent

---

## Security

### Current Implementation

**Secrets:**
- ✅ No secrets in code or .env.example
- ✅ GCP Secret Manager in production
- ✅ GitHub Secrets for CI/CD

**Database:**
- ✅ Cloud SQL with private IP
- ✅ Connection via Cloud SQL Proxy
- ✅ Automated backups

**API:**
- ❌ No authentication yet (Wave 5)
- ❌ No rate limiting yet (Wave 5)
- ⚠️  Public endpoint (for now)

### Future (Wave 5)
- API key authentication
- Rate limiting per tenant
- CORS configuration
- Input validation (already have Pydantic)

---

## Monitoring

### Current
- Cloud Run metrics (requests, latency, errors)
- Cloud SQL metrics (connections, CPU, memory)
- Cloud Logging (structured JSON logs)

### Future (Wave 5+)
- Custom metrics for LLM usage
- Cost tracking per tenant
- Error alerting
- Uptime monitoring

---

## Known Limitations

### Wave 1 Gaps
- LLM adapters are stubs (no actual API calls)
- RAG is not implemented (tables exist)
- Flow orchestration is stub code
- Test coverage is low (8%)
- No authentication

### Production Issues
- `/healthz` endpoint returns 404 (use `/health` instead)
- Cold starts can be slow with large dependencies

### Technical Debt
- Need to increase test coverage to 80%
- Need to implement actual retry logic
- Need to add request logging
- Need to optimize Docker image size

---

## Next Steps (Wave 2)

**Priority 1:**
1. Implement OpenAI adapter with real API calls
2. Add retry logic and error handling
3. Write tests for adapters (80% coverage)

**Priority 2:**
4. Implement RAG ingestion with pgvector
5. Implement semantic search retrieval
6. Test pgvector performance

**Priority 3:**
7. Build the 5 core business logic tools
8. Implement flow orchestration
9. End-to-end integration tests

See [wave2/](../wave2/) for detailed planning.

---

## References

- [DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md) - How to build on this
- [CODING_STANDARDS.md](CODING_STANDARDS.md) - Code style guide
- [TESTING_GUIDE.md](../TESTING_GUIDE.md) - Testing reference
- [wave1/README.md](../wave1%20-%20Foundation/README.md) - Wave 1 summary
