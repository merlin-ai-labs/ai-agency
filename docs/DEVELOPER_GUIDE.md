# Developer Guide

Quick guide for building on the AI Consulting Agency Platform.

## Quick Start

```bash
# 1. Setup
git clone <repo-url>
cd ConsultingAgency
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Start database
docker-compose up -d db

# 4. Run migrations
./scripts/run_migrations.sh

# 5. Start API
uvicorn app.main:app --reload --port 8080
```

Visit http://localhost:8080/docs

---

## Running Tests

```bash
# Run all tests
pytest -v

# With coverage
pytest --cov=app --cov-report=html
open htmlcov/index.html

# Specific test
pytest tests/test_main.py::test_healthz -v
```

## Code Quality

```bash
# Format and lint
ruff check app tests --fix
ruff format app tests

# Type check
mypy app
```

---

## Project Structure

```
app/
├── api/            # FastAPI routes (/runs, /health)
├── core/           # Base classes, exceptions, decorators
├── db/             # SQLModel models, sessions
├── adapters/       # LLM providers (stubs for Wave 2)
├── flows/          # Workflows (stubs for Wave 2)
├── tools/          # Business logic (stubs for Wave 2)
├── rag/            # Vector search (stubs for Wave 2)
└── main.py         # FastAPI app
```

**Key Files:**
- `app/main.py` - API endpoints
- `app/db/models.py` - Database tables
- `app/core/base.py` - Base classes
- `.env` - Local config
- `alembic/versions/` - Migrations

---

## Building Flows (Wave 2)

### 1. Create Flow Structure

```bash
mkdir -p app/flows/my_flow
```

Create these files:

**`app/flows/my_flow/state.py`**
```python
from typing import TypedDict

class MyFlowState(TypedDict):
    input: str
    output: str
```

**`app/flows/my_flow/nodes.py`**
```python
from app.core.base import BaseTool

class MyTool(BaseTool):
    async def execute(self, input_data: dict) -> dict:
        # Your logic here
        return {"result": "processed"}
```

**`app/flows/my_flow/graph.py`**
```python
from .state import MyFlowState
from .nodes import MyTool

async def run_flow(input_data: dict) -> dict:
    tool = MyTool()
    result = await tool.execute(input_data)
    return result
```

### 2. Register in API

Add to `app/main.py`:

```python
@app.post("/my-flow")
async def my_flow(data: dict):
    from app.flows.my_flow.graph import run_flow
    result = await run_flow(data)
    return result
```

---

## Using LLM Adapters (Wave 2)

```python
from app.adapters.llm_factory import LLMFactory

# Get configured LLM
llm = await LLMFactory.create()

# Generate completion
response = await llm.generate_completion(
    prompt="Your prompt",
    temperature=0.7,
    max_tokens=1000
)
```

**Note:** Adapters are stubs in Wave 1. Implement in Wave 2.

---

## Database Operations

### Creating Models

Edit `app/db/models.py`:

```python
from sqlmodel import SQLModel, Field
from uuid import UUID, uuid4

class MyModel(SQLModel, table=True):
    __tablename__ = "my_table"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    tenant_id: str = Field(index=True)
    name: str
    data: dict = Field(default_factory=dict)
```

### Creating Migrations

```bash
# Generate migration
alembic revision --autogenerate -m "Add my_table"

# Review the file in alembic/versions/

# Apply migration
alembic upgrade head
```

### Using Sessions

```python
from app.db.base import get_session
from sqlmodel import select

async with get_session() as session:
    result = await session.exec(
        select(MyModel).where(MyModel.tenant_id == "tenant-123")
    )
    items = result.all()
```

---

## RAG Implementation (Wave 2)

### Ingesting Documents

```python
from app.rag.ingestion import RAGIngestion

ingestion = RAGIngestion()
await ingestion.ingest_document(
    tenant_id="my-tenant",
    document_content="Your document text...",
    metadata={"source": "file.pdf"}
)
```

### Retrieving Context

```python
from app.rag.retriever import RAGRetriever

retriever = RAGRetriever()
context = await retriever.retrieve(
    tenant_id="my-tenant",
    query="What is...",
    top_k=5
)
```

**Note:** RAG is stubbed in Wave 1. Implement in Wave 2.

---

## Common Tasks

### Add Environment Variable

1. Add to `.env.example`:
```bash
MY_VAR=default_value
```

2. Add to `app/config.py`:
```python
class Settings(BaseSettings):
    my_var: str = "default_value"
```

3. Use in code:
```python
from app.config import get_settings
settings = get_settings()
value = settings.my_var
```

### Add API Endpoint

Add to `app/main.py`:

```python
@app.get("/my-endpoint")
async def my_endpoint():
    return {"status": "ok"}
```

### Debug with Logging

```python
from app.logging import get_logger

logger = get_logger(__name__)

logger.info("Starting process", extra={"tenant_id": tenant_id})
logger.error("Process failed", exc_info=True)
```

---

## Troubleshooting

### Database Connection Issues

```bash
# Check if DB is running
docker-compose ps

# Restart DB
docker-compose restart db

# Check migrations
alembic current
```

### Import Errors

```bash
# Reinstall in dev mode
pip install -e ".[dev]"
```

### Port Already in Use

```bash
# Find process
lsof -i :8080

# Kill it
kill -9 <PID>
```

### Tests Failing

```bash
# Clear cache
pytest --cache-clear

# Run with verbose output
pytest -vv -s
```

---

## Development Workflow

```bash
# 1. Create branch
git checkout -b feature/my-feature

# 2. Make changes

# 3. Run tests
pytest -v

# 4. Quality checks
ruff check app tests --fix
ruff format app tests

# 5. Commit
git add .
git commit -m "feat: my feature"

# 6. Push
git push origin feature/my-feature

# 7. Open PR
# Merging to main auto-deploys to production
```

---

## What's Implemented vs Stubs

### ✅ Working (Wave 1)
- FastAPI app with endpoints
- Database models and sessions
- Alembic migrations
- Base classes and exceptions
- Docker Compose setup
- CI/CD pipeline

### ⏳ Stubs (Wave 2+)
- LLM adapters (no real API calls)
- RAG ingestion/retrieval
- Flow orchestration
- Business logic tools
- Authentication

---

## References

- [ARCHITECTURE.md](project/ARCHITECTURE.md) - System architecture
- [CODING_STANDARDS.md](project/CODING_STANDARDS.md) - Code style
- Production: https://ai-agency-4ebxrg4hdq-ew.a.run.app/docs

---

## Quick Commands

```bash
# Development
docker-compose up -d db
uvicorn app.main:app --reload --port 8080

# Testing
pytest -v --cov=app

# Quality
ruff check app tests --fix
ruff format app tests

# Migrations
alembic revision --autogenerate -m "description"
alembic upgrade head

# Deploy
git push origin main  # Auto-deploys via GitHub Actions
```
