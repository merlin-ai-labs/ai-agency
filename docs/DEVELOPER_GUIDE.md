# Developer Guide

Quick start guide for building LLM flows and agents on the AI Consulting Agency Platform.

## Prerequisites

- Python 3.11+
- Docker and Docker Compose (or Colima on macOS)
- Git
- Google Cloud CLI (optional, for production deployment)

## First Time Setup

### 1. Clone and Install

```bash
git clone <repo-url>
cd ConsultingAgency
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set your API keys:
```bash
# Required: Choose your LLM provider
LLM_PROVIDER=openai  # or "vertex"
OPENAI_API_KEY=sk-...  # If using OpenAI

# Required: Database (Docker handles this)
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/ai_agency

# Optional: GCP (only if using Vertex AI or production)
GCP_PROJECT_ID=your-project
GCS_BUCKET=your-bucket
```

### 3. Start Database

```bash
docker-compose up -d db
```

Or with Colima on macOS:
```bash
colima start
docker-compose up -d db
```

### 4. Run Migrations

```bash
./scripts/run_migrations.sh
```

### 5. Start the API

```bash
uvicorn app.main:app --reload --port 8080
```

Visit: http://localhost:8080/docs

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest -v

# With coverage report
pytest -v --cov=app --cov-report=html
open htmlcov/index.html
```

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for more testing commands.

### Code Quality

```bash
# Format and lint
ruff check app tests --fix
ruff format app tests

# Type checking
mypy app
```

### Making Changes

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Write tests
4. Run quality checks
5. Commit and push
6. Open a PR

## Building LLM Flows

### Flow Structure

Flows are in `app/flows/`. Each flow has:
- `graph.py` - Flow orchestration
- `nodes.py` - Individual processing steps
- `state.py` - Flow state definition

### Example: Creating a New Flow

```python
# app/flows/my_flow/state.py
from typing import TypedDict

class MyFlowState(TypedDict):
    input: str
    output: str
    metadata: dict

# app/flows/my_flow/nodes.py
from app.core.base import BaseTool

class MyTool(BaseTool):
    async def execute(self, input_data: dict) -> dict:
        # Your logic here
        return {"result": "processed"}

# app/flows/my_flow/graph.py
from langgraph.graph import StateGraph

def create_graph():
    graph = StateGraph(MyFlowState)
    # Define nodes and edges
    return graph.compile()
```

### Using LLM Adapters

```python
from app.adapters.llm_factory import LLMFactory

# Get the configured LLM
llm = await LLMFactory.create()

# Generate completions
response = await llm.generate_completion(
    prompt="Your prompt here",
    temperature=0.7,
    max_tokens=1000
)
```

### Using RAG

```python
from app.rag.ingestion import RAGIngestion
from app.rag.retriever import RAGRetriever

# Ingest documents
ingestion = RAGIngestion()
await ingestion.ingest_document(
    tenant_id="your-tenant",
    document_content="...",
    metadata={"source": "file.pdf"}
)

# Retrieve context
retriever = RAGRetriever()
context = await retriever.retrieve(
    tenant_id="your-tenant",
    query="What is...",
    top_k=5
)
```

## Database Operations

### Creating Models

Models are in `app/db/models.py`:

```python
from sqlmodel import SQLModel, Field
from uuid import UUID, uuid4

class MyModel(SQLModel, table=True):
    __tablename__ = "my_table"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    tenant_id: str = Field(index=True)
    name: str
    data: dict = Field(default_factory=dict, sa_column=Column(JSON))
```

### Creating Migrations

```bash
# Auto-generate migration
alembic revision --autogenerate -m "Add my_table"

# Review the generated file in alembic/versions/

# Apply migration
alembic upgrade head
```

### Using Database Sessions

```python
from app.db.base import get_session
from sqlmodel import select

async with get_session() as session:
    result = await session.exec(
        select(MyModel).where(MyModel.tenant_id == "tenant-123")
    )
    items = result.all()
```

## Project Structure

```
ConsultingAgency/
├── app/
│   ├── api/              # API endpoints
│   ├── core/             # Base classes, exceptions, decorators
│   ├── db/               # Database models and sessions
│   ├── flows/            # Business workflows
│   │   ├── maturity_assessment/
│   │   └── usecase_grooming/
│   ├── llm/              # (deprecated, use adapters/)
│   ├── adapters/         # LLM provider adapters
│   ├── rag/              # RAG ingestion and retrieval
│   ├── tools/            # Business logic tools
│   └── main.py           # FastAPI app
├── tests/                # Test suite
├── scripts/              # Utility scripts
│   ├── setup_dev.sh      # Development environment setup
│   ├── run_migrations.sh # Database migrations
│   └── seed.py           # Seed test data
├── docs/                 # Documentation
├── alembic/              # Database migrations
└── docker-compose.yml    # Local services (PostgreSQL)
```

## Common Tasks

### Add a New API Endpoint

1. Create route in `app/api/`:
```python
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/my-endpoint", tags=["My Feature"])

@router.post("/")
async def my_endpoint(data: MyRequest):
    return {"result": "success"}
```

2. Register in `app/main.py`:
```python
from app.api import my_endpoint
app.include_router(my_endpoint.router)
```

### Add Environment Variable

1. Add to `.env.example`:
```bash
MY_NEW_VAR=default_value
```

2. Add to `app/config.py`:
```python
class Settings(BaseSettings):
    my_new_var: str = "default_value"
```

3. Use in code:
```python
from app.config import get_settings

settings = get_settings()
value = settings.my_new_var
```

### Debug a Flow

```python
# Add logging
from app.logging import get_logger

logger = get_logger(__name__)

async def my_function():
    logger.info("Starting process", extra={"tenant_id": tenant_id})
    try:
        result = await process()
        logger.info("Process complete", extra={"result": result})
    except Exception as e:
        logger.error("Process failed", exc_info=True)
        raise
```

## Deployment

The application auto-deploys to Google Cloud Run when you push to `main`.

### Local Testing with Docker

```bash
# Build image
docker build -t ai-agency .

# Run locally
docker run -p 8080:8080 --env-file .env ai-agency
```

### Production Environment

- **Service URL**: https://ai-agency-4ebxrg4hdq-ew.a.run.app
- **Region**: europe-west1
- **Database**: Cloud SQL PostgreSQL 15
- **Storage**: Google Cloud Storage

## Troubleshooting

### Database Connection Issues

```bash
# Check if database is running
docker-compose ps

# Restart database
docker-compose restart db

# Check migrations
alembic current
alembic history
```

### Import Errors

```bash
# Reinstall in development mode
pip install -e ".[dev]"
```

### Tests Failing

```bash
# Clear cache
pytest --cache-clear

# Run specific test
pytest tests/test_main.py::test_healthz -v -s
```

### Port Already in Use

```bash
# Find process on port 8080
lsof -i :8080

# Kill process
kill -9 <PID>
```

## Coding Standards

- Follow PEP 8 style guide
- Use type hints for all functions
- Write docstrings for public APIs
- Keep functions small and focused
- Prefer composition over inheritance

See [project/CODING_STANDARDS.md](project/CODING_STANDARDS.md) for detailed guidelines.

## Architecture

For technical architecture details, see [project/ARCHITECTURE.md](project/ARCHITECTURE.md).

## Getting Help

- Check the [TESTING_GUIDE.md](TESTING_GUIDE.md) for testing help
- Review existing code for examples
- Ask the team in your communication channel

## Quick Reference

```bash
# Start development
docker-compose up -d db
source venv/bin/activate
uvicorn app.main:app --reload --port 8080

# Run tests
pytest -v

# Code quality
ruff check app tests --fix
ruff format app tests

# Database migrations
alembic upgrade head

# View API docs
open http://localhost:8080/docs
```
