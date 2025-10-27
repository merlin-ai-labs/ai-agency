# AI Agency — Lean Repository Architecture

## Overview

This repository implements a **minimalist AI agents platform** for orchestrating multi-step workflows on GCP. The design prioritizes:

- **Fast iteration**: No heavy frameworks, minimal abstractions
- **Easy provider swap**: Thin adapters for LLM/Embeddings
- **Operational simplicity**: Single FastAPI service, database-backed queue
- **Multi-tenancy**: Tenant isolation via `tenant_id` field

## Target Flows

### Flow A: Maturity Assessment
1. Parse uploaded documents (PDF, Word)
2. Score against maturity rubrics using LLM
3. Generate prioritized recommendations
4. Emit `assessment.json` + artifacts to GCS

### Flow B: Use-Case Grooming
1. Load `assessment.json` from previous flow
2. Rank use-cases using RICE/WSJF scoring
3. Generate prioritized backlog
4. Emit `backlog.json` to GCS

## Architecture Principles

### 1. Single Service (No Microservices)
- **FastAPI** handles both API and execution in the same process
- No Pub/Sub, no separate workers (initially)
- Execution loop polls database for queued runs

### 2. Database-Backed Queue
- `runs` table with `status` field (`queued`, `running`, `completed`, `failed`)
- Simple polling: `SELECT * FROM runs WHERE status='queued' ORDER BY created_at LIMIT 1`
- No external queue infrastructure needed

### 3. Thin Adapters
- LLM: `OpenAIChatModel`, `VertexChatModel` with identical interface
- Swap providers via `LLM_PROVIDER` environment variable
- No vendor lock-in

### 4. Versioned Tools
- Tools live under `app/tools/{tool_name}/{version}/`
- Registry resolves by name and version: `registry.resolve("parse_docs", "1.x")`
- Enables incremental improvements without breaking flows

### 5. Lean Storage
- Artifacts go to **GCS** (PDFs, JSON outputs)
- RAG uses **Postgres + pgvector** for embeddings
- SQLite for local dev

## Directory Structure

```
ai-agency/
├─ app/
│  ├─ main.py                 # FastAPI app (healthz, POST/GET /runs)
│  ├─ exec_loop.py            # Execution loop (polls runs table)
│  ├─ config.py               # Settings from environment variables
│  ├─ logging.py              # Structured logging (structlog)
│  │
│  ├─ flows/                  # Flow orchestration
│  │  ├─ maturity_assessment/
│  │  │  ├─ graph.py          # Flow definition
│  │  │  └─ schemas/assessment.schema.json
│  │  └─ usecase_grooming/
│  │     ├─ graph.py
│  │     └─ schemas/usecase.schema.json
│  │
│  ├─ tools/                  # Versioned, reusable tools
│  │  ├─ parse_docs/v1/
│  │  ├─ score_rubrics/v1/
│  │  ├─ gen_recs/v1/
│  │  ├─ rank_usecases/v1/
│  │  ├─ write_backlog/v1/
│  │  └─ registry.py          # Tool resolution
│  │
│  ├─ adapters/               # External service wrappers
│  │  ├─ llm_openai.py
│  │  ├─ llm_vertex.py
│  │  └─ llm_factory.py
│  │
│  ├─ rag/                    # RAG (Retrieval-Augmented Generation)
│  │  ├─ retriever.py         # Semantic search (pgvector)
│  │  └─ ingestion.py         # Document chunking + embedding
│  │
│  └─ db/
│     ├─ models.py            # SQLModel (Run, DocumentChunk, Tenant)
│     └─ migrations/          # Alembic (TODO)
│
├─ scripts/
│  ├─ seed.py                 # Create tables + seed test data
│  └─ smoke_run.py            # Trigger a local test run
│
├─ tests/                     # Pytest tests
├─ docs/                      # This file + other docs
├─ Dockerfile                 # Cloud Run deployment
├─ pyproject.toml             # Dependencies
├─ .env.example               # Environment variable template
└─ README.md                  # Quick start
```

## Data Flow

### 1. Create Run
```
POST /runs
{
  "flow_name": "maturity_assessment",
  "tenant_id": "acme",
  "input_data": {"documents": ["gs://..."]}
}
→ Insert into runs table (status=queued)
→ Return run_id
```

### 2. Execution Loop
```
while True:
  run = SELECT * FROM runs WHERE status='queued' LIMIT 1
  if run:
    UPDATE runs SET status='running'
    result = execute_flow(run.flow_name, run.input_data)
    UPDATE runs SET status='completed', output_data=result
  sleep(2)
```

### 3. Flow Execution
```python
# app/flows/maturity_assessment/graph.py
async def run(input_data):
  # 1. Parse documents
  parsed = await registry.resolve("parse_docs", "1.x").parse_documents(input_data["documents"])

  # 2. Score rubrics
  scores = await registry.resolve("score_rubrics", "1.x").score_against_rubrics(parsed)

  # 3. Generate recommendations
  recs = await registry.resolve("gen_recs", "1.x").generate_recommendations(scores)

  # 4. Upload artifacts
  assessment_url = await upload_to_gcs(assessment_json)

  return {"assessment_url": assessment_url}
```

## Technology Stack

### Core
- **Python 3.11+**
- **FastAPI** (API + execution)
- **SQLModel** (ORM, combines SQLAlchemy + Pydantic)
- **Structlog** (structured logging)

### Database
- **Postgres + pgvector** (production)
- **SQLite** (local dev)

### LLM/Embeddings
- **OpenAI API** (gpt-4o, text-embedding-3-large)
- **Vertex AI** (gemini-2.0-flash-exp, textembedding-gecko)

### Storage
- **GCS** (artifacts, documents)

### Deployment
- **Cloud Run** (serverless containers)
- **Secret Manager** (secrets)

## Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/ai_agency

# GCP
GCP_PROJECT_ID=your-project
GCS_BUCKET=your-bucket

# LLM
LLM_PROVIDER=openai  # or "vertex"
OPENAI_API_KEY=sk-...
VERTEX_AI_LOCATION=us-central1

# App
LOG_LEVEL=INFO
ENVIRONMENT=production
```

## Deployment

### Local Development
```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # Edit with your values
python scripts/seed.py
uvicorn app.main:app --reload
```

### Cloud Run (GCP)
```bash
# Build and deploy
gcloud run deploy ai-agency \
  --source . \
  --region us-central1 \
  --set-env-vars DATABASE_URL=$DB_URL,GCS_BUCKET=$BUCKET,LLM_PROVIDER=vertex

# Set up database
gcloud sql instances create ai-agency-db --database-version=POSTGRES_15
gcloud sql databases create ai_agency --instance=ai-agency-db
psql $DATABASE_URL -c "CREATE EXTENSION vector;"
```

## Testing

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=app

# Run smoke test (requires running server)
python scripts/smoke_run.py
```

## Operational Patterns

### Multi-Tenancy
- All tables have `tenant_id` field
- Flows receive `tenant_id` in input
- RAG retrieval filters by `tenant_id`
- GCS artifacts stored under `gs://{bucket}/{tenant_id}/`

### Error Handling
- Runs marked `status=failed` on error
- Error message stored in `error_message` field
- Retries can be implemented by resetting status to `queued`

### Monitoring (TODO)
- Health check: `GET /healthz`
- Metrics: TODO (Prometheus, Cloud Monitoring)
- Tracing: TODO (OpenTelemetry)
- Alerting: TODO (Cloud Alerting)

## Extension Points

### Adding a New Flow
1. Create `app/flows/my_flow/graph.py`
2. Define schema: `app/flows/my_flow/schemas/output.schema.json`
3. Implement `async def run(input_data)` method
4. Register tools in `registry.py` (if new)

### Adding a New Tool
1. Create `app/tools/my_tool/v1/__init__.py`
2. Implement tool function(s)
3. Add to `registry.list_tools()`
4. Use in flows: `registry.resolve("my_tool", "1.x")`

### Swapping LLM Providers
1. Set `LLM_PROVIDER=vertex` (or create new adapter)
2. Implement `VertexChatModel.complete()` method
3. Update `llm_factory.get_chat_model()`

## Non-Goals (Initially)

- ❌ Frontend (no UI)
- ❌ Pub/Sub (use database queue)
- ❌ Heavy orchestration (LangGraph, Prefect, etc.)
- ❌ Observability frameworks (OpenTelemetry)
- ❌ Infrastructure-as-code (Terraform)

## Roadmap

### Phase 1: MVP (Current)
- ✅ Skeleton repository structure
- ⬜ Implement core flows (maturity assessment, use-case grooming)
- ⬜ Implement core tools (parse, score, rank)
- ⬜ Basic RAG with pgvector

### Phase 2: Production-Ready
- ⬜ Add authentication/API keys
- ⬜ Add rate limiting
- ⬜ Add monitoring and alerting
- ⬜ Add database migrations (Alembic)
- ⬜ Add CI/CD pipeline

### Phase 3: Scale
- ⬜ Migrate to Pub/Sub for queue
- ⬜ Add separate worker processes
- ⬜ Add caching (Redis)
- ⬜ Add advanced RAG (reranking, hybrid search)

## References

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [SQLModel Docs](https://sqlmodel.tiangolo.com/)
- [pgvector](https://github.com/pgvector/pgvector)
- [Cloud Run](https://cloud.google.com/run)
