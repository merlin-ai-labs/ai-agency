# Developer Guide

Complete guide for building on the AI Consulting Agency Platform.

**Important:** Before writing code, read [CODING_STANDARDS.md](CODING_STANDARDS.md) for best practices on decorators, base classes, error handling, and testing patterns.

## Table of Contents

- [Quick Start](#quick-start)
- [Local Development Setup](#local-development-setup)
- [Building New Flows](#building-new-flows)
- [Code Organization](#code-organization)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

See [QUICKSTART.md](QUICKSTART.md) for 5-minute setup guide.

**TL;DR:**
```bash
./dev setup    # One-command setup
./dev server   # Start development server
./dev help     # Show all commands
```

---

## Local Development Setup

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Required for local dev
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5433/ai_agency
OPENAI_API_KEY=sk-your-key-here

# Optional - only needed if using these features
GCP_PROJECT_ID=your-project-id
GCS_BUCKET=your-bucket
VERTEX_AI_LOCATION=europe-west1
```

### Database Setup - Cloud SQL Proxy

The platform uses Cloud SQL Proxy for local development to connect to the production Cloud SQL database:

#### Prerequisites
```bash
# Install Cloud SQL Proxy (macOS)
brew install cloud-sql-proxy

# Or download binary
curl -o cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.8.0/cloud-sql-proxy.darwin.amd64
chmod +x cloud-sql-proxy
```

#### Authentication
```bash
# Authenticate with GCP
gcloud auth application-default login

# Verify access to project
gcloud config set project merlin-notebook-lm
```

#### Get Cloud SQL Connection Details
```bash
# List Cloud SQL instances
gcloud sql instances list

# Get connection name (format: PROJECT:REGION:INSTANCE)
# Example: merlin-notebook-lm:europe-west1:ai-agency-db

# Get database password (stored in Secret Manager)
gcloud secrets versions access latest --secret="cloud-sql-password"
```

#### Two-Terminal Workflow

**Terminal 1: Start Cloud SQL Proxy**
```bash
# Start proxy on port 5433
cloud-sql-proxy merlin-notebook-lm:europe-west1:ai-agency-db \
  --port 5433

# You should see: "Listening on 127.0.0.1:5433"
# Keep this terminal running

# Or use the dev CLI:
./dev db-proxy
```

**Terminal 2: Configure and Run Application**
```bash
# Update .env with Cloud SQL connection
DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5433/ai_agency

# Run migrations
alembic upgrade head

# Verify connection
PGPASSWORD='YOUR_PASSWORD' psql -h localhost -p 5433 -U postgres -d ai_agency -c "SELECT version();"

# Start API
uvicorn app.main:app --reload --port 8000
```

#### Test the Setup
```bash
# Health check
curl http://localhost:8000/health

# Test weather endpoint
curl -X POST http://localhost:8000/weather-chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in Paris?", "tenant_id": "test"}'
```

#### Troubleshooting Cloud SQL Proxy

**Issue: "Permission denied"**
```bash
# Solution: Add IAM permissions
gcloud projects add-iam-policy-binding merlin-notebook-lm \
  --member="user:YOUR_EMAIL@gmail.com" \
  --role="roles/cloudsql.client"
```

**Issue: "Connection refused"**
```bash
# Solution: Check proxy is running
./dev db-check

# Or manually check
lsof -ti:5433

# Restart proxy
pkill -f cloud-sql-proxy
./dev db-proxy
```

**Issue: "Database connection failed"**
```bash
# Solution: Verify credentials
# 1. Check password from Secret Manager
gcloud secrets versions access latest --secret="cloud-sql-password"

# 2. Test connection directly
PGPASSWORD='PASSWORD' psql -h localhost -p 5433 -U postgres -d ai_agency -c "\dt"
```

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "Add new table"

# Review generated file in alembic/versions/

# Apply migrations
alembic upgrade head

# Check current version
alembic current

# Rollback one version
alembic downgrade -1
```

---

## Building New Flows

The platform uses a **multi-flow architecture** where different agent types share the same conversation infrastructure.

### Understanding the Multi-Flow Architecture

All agent flows use:
- Single `conversations` table with `flow_type` field
- Single `messages` table with `flow_type` field (denormalized for performance)
- `ConversationRepository` for database operations
- Provider-agnostic LLM adapters

**Why this design?**
- One codebase to maintain for conversation logic
- Easy to add new flow types
- Consistent API patterns
- Efficient queries with composite indexes

### The WeatherAgentFlow Template

The `WeatherAgentFlow` is the reference implementation. Study it at:
`app/flows/agents/weather_agent.py`

It demonstrates:
- Loading conversation history
- Adding user messages
- Calling LLM with tool definitions
- Handling tool calls (function calling)
- Saving responses to database
- Multi-turn conversations

### Step-by-Step: Building a New Flow

#### 1. Create Flow Class

Create `app/flows/agents/your_flow.py`:

```python
"""Your custom agent flow."""

import logging
from typing import Any
from sqlmodel import Session

from app.adapters.llm_factory import get_llm_adapter
from app.core.base import BaseFlow
from app.core.decorators import log_execution, timeout
from app.db.base import get_session
from app.db.repositories.conversation_repository import ConversationRepository

logger = logging.getLogger(__name__)


class YourAgentFlow(BaseFlow):
    """Your agent description.

    Example:
        >>> flow = YourAgentFlow()
        >>> result = await flow.run(
        ...     user_message="Your question",
        ...     tenant_id="user_123"
        ... )
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(
            name="your_agent",
            description="Your agent description",
            version="1.0.0",
        )

        # Initialize LLM adapter (provider-agnostic)
        self.llm = get_llm_adapter(provider=provider, model=model)

        logger.info(
            f"Initialized YourAgentFlow with {self.llm.provider_name}",
            extra={"provider": self.llm.provider_name}
        )

    @log_execution
    @timeout(seconds=120.0)
    async def run(
        self,
        user_message: str,
        tenant_id: str,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Process user message through your agent.

        Args:
            user_message: User's message/question
            tenant_id: Tenant identifier
            conversation_id: Optional conversation ID

        Returns:
            {
                "response": str,
                "conversation_id": str,
                # ... your custom fields
            }
        """
        with Session(get_session()) as session:
            repo = ConversationRepository(session)

            # Create or load conversation
            if not conversation_id:
                conversation_id = repo.create_conversation(
                    tenant_id=tenant_id,
                    flow_type="your_flow_type",  # Use consistent flow_type
                )

            # Load conversation history
            history = repo.get_conversation_history(conversation_id)

            # Save user message
            repo.save_message(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                flow_type="your_flow_type",
                role="user",
                content=user_message,
            )

            # Build messages for LLM
            messages = []
            if not history:
                messages.append({
                    "role": "system",
                    "content": "Your system prompt here"
                })

            messages.extend(history)
            messages.append({"role": "user", "content": user_message})

            # Call LLM
            response = await self.llm.generate_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
            )

            assistant_message = response["content"]

            # Save assistant response
            repo.save_message(
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                flow_type="your_flow_type",
                role="assistant",
                content=assistant_message,
            )

            return {
                "response": assistant_message,
                "conversation_id": conversation_id,
            }
```

#### 2. Add Tool Support (Optional)

If your agent needs tools/function calling:

```python
# Define tool
YOUR_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "your_tool_name",
        "description": "What your tool does",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "Parameter description",
                },
            },
            "required": ["param1"],
        },
    },
}

# In your run() method, call LLM with tools:
async def _call_llm_with_tools(self, messages, tenant_id):
    """Call LLM with tool definitions."""
    response = await self.llm.generate_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=1000,
        tools=[YOUR_TOOL_DEFINITION],
        tool_choice="auto",
    )
    return response

# Then handle tool calls:
if response.get("tool_calls"):
    for tool_call in response["tool_calls"]:
        if tool_call["function"]["name"] == "your_tool_name":
            # Execute your tool
            result = await execute_your_tool(**tool_call["function"]["arguments"])
            # Add tool result to messages and call LLM again
```

See `WeatherAgentFlow` for complete tool calling example.

#### 3. Create API Endpoint

Add to `app/main.py`:

```python
@app.post("/your-flow/chat")
async def your_flow_chat(
    user_message: str,
    tenant_id: str,
    conversation_id: str | None = None,
):
    """Your flow endpoint."""
    from app.flows.agents.your_flow import YourAgentFlow

    flow = YourAgentFlow()
    result = await flow.run(
        user_message=user_message,
        tenant_id=tenant_id,
        conversation_id=conversation_id,
    )
    return result
```

#### 4. Add to FlowType Enum

Update `app/db/models.py`:

```python
class FlowType(str, Enum):
    """Supported agent flow types."""
    WEATHER = "weather"
    YOUR_FLOW = "your_flow"  # Add this
    # ... other flows
```

### Database: Using ConversationRepository

The `ConversationRepository` provides all conversation storage operations:

```python
from sqlmodel import Session
from app.db.base import get_session
from app.db.repositories.conversation_repository import ConversationRepository

with Session(get_session()) as session:
    repo = ConversationRepository(session)

    # Create conversation
    conv_id = repo.create_conversation(
        tenant_id="tenant_123",
        flow_type="weather",
    )

    # Save message
    repo.save_message(
        conversation_id=conv_id,
        tenant_id="tenant_123",
        flow_type="weather",
        role="user",
        content="Hello!",
    )

    # Get history (returns list of dicts)
    history = repo.get_conversation_history(conv_id)
    # [{"role": "user", "content": "Hello!"}, ...]

    # Check if conversation exists
    exists = repo.conversation_exists(conv_id)
```

### LLM Integration

The platform supports multiple LLM providers through adapters.

#### Get LLM Adapter

```python
from app.adapters.llm_factory import get_llm_adapter

# Use configured provider (from settings)
llm = get_llm_adapter()

# Or specify provider
llm = get_llm_adapter(provider="openai")
llm = get_llm_adapter(provider="vertex")
llm = get_llm_adapter(provider="mistral")

# Or specify provider and model
llm = get_llm_adapter(provider="openai", model="gpt-4")
```

#### Generate Completions

```python
# Simple completion
response = await llm.generate_completion(
    messages=[
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_tokens=1000,
)
print(response["content"])

# With tools/function calling
response = await llm.generate_completion(
    messages=messages,
    tools=[tool_definition],
    tool_choice="auto",
)

if response.get("tool_calls"):
    # Handle tool calls
    pass
```

#### Supported Providers

**OpenAI** (default)
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4-turbo-2024-04-09  # Optional override
```

**Vertex AI (Google Gemini)**
```bash
LLM_PROVIDER=vertex
VERTEX_AI_PROJECT_ID=your-project
VERTEX_AI_LOCATION=us-central1
VERTEX_AI_MODEL=gemini-2.0-flash-exp  # Optional override
```

**Mistral**
```bash
LLM_PROVIDER=mistral
MISTRAL_API_KEY=...
MISTRAL_MODEL=mistral-medium-latest  # Optional override
```

### Rate Limiting

Rate limiting is configured per provider in `app/config.py`:

```python
# OpenAI limits
openai_rate_limit_tpm: int = 90000  # Tokens per minute
openai_rate_limit_tph: int = 5000000  # Tokens per hour

# Mistral limits
mistral_rate_limit_tpm: int = 2000000
mistral_rate_limit_tph: int = 100000000
```

Rate limiting is applied per tenant and provider automatically by the adapters.

### Error Handling

Use structured logging and proper error handling:

```python
from app.logging import get_logger

logger = get_logger(__name__)

try:
    result = await flow.run(...)
except TimeoutError:
    logger.error("Flow timed out", extra={"tenant_id": tenant_id})
    raise
except Exception as e:
    logger.error(
        "Flow failed",
        exc_info=True,
        extra={"tenant_id": tenant_id, "error": str(e)}
    )
    raise
```

### Testing Your Flow

Create `tests/test_flows/test_your_flow.py`:

```python
import pytest
from app.flows.agents.your_flow import YourAgentFlow


@pytest.mark.asyncio
async def test_your_flow_basic():
    """Test basic flow execution."""
    flow = YourAgentFlow(provider="openai")

    result = await flow.run(
        user_message="Test message",
        tenant_id="test_tenant",
    )

    assert "response" in result
    assert "conversation_id" in result
    assert result["response"]  # Not empty


@pytest.mark.asyncio
async def test_your_flow_conversation():
    """Test multi-turn conversation."""
    flow = YourAgentFlow()

    # First message
    result1 = await flow.run(
        user_message="First message",
        tenant_id="test_tenant",
    )
    conv_id = result1["conversation_id"]

    # Second message in same conversation
    result2 = await flow.run(
        user_message="Second message",
        tenant_id="test_tenant",
        conversation_id=conv_id,
    )

    assert result2["conversation_id"] == conv_id
```

---

## Code Organization

```
app/
├── main.py                    # FastAPI app and endpoints
├── config.py                  # Settings and environment variables
├── logging.py                 # Structured logging setup
├── exec_loop.py              # Background worker (legacy)
│
├── core/                      # Base classes and utilities
│   ├── base.py               # BaseFlow, BaseTool
│   ├── decorators.py         # @log_execution, @timeout
│   └── exceptions.py         # Custom exceptions
│
├── db/                        # Database layer
│   ├── models.py             # SQLModel models (tables)
│   ├── base.py               # Database session management
│   └── repositories/         # Data access layer
│       └── conversation_repository.py
│
├── adapters/                  # LLM provider adapters
│   ├── llm_factory.py        # Provider factory
│   ├── base_llm.py           # Base LLM interface
│   ├── openai_adapter.py     # OpenAI implementation
│   ├── vertex_adapter.py     # Vertex AI implementation
│   └── mistral_adapter.py    # Mistral implementation
│
├── flows/                     # Business workflows
│   ├── agents/               # Agent flows
│   │   └── weather_agent.py # Weather agent (reference)
│   ├── maturity_assessment/  # Maturity flow (legacy)
│   └── usecase_grooming/     # Grooming flow (legacy)
│
├── tools/                     # Business logic tools
│   └── weather/              # Weather tool
│       └── v1.py            # Weather API implementation
│
└── rag/                       # RAG (future)
    ├── ingestion.py
    └── retriever.py
```

### Key Files

**`app/main.py`** - API endpoints and FastAPI app
**`app/config.py`** - All configuration via environment variables
**`app/db/models.py`** - Database table definitions
**`app/db/repositories/conversation_repository.py`** - Conversation data access
**`app/adapters/llm_factory.py`** - LLM provider selection
**`app/flows/agents/weather_agent.py`** - Reference implementation

---

## Testing

### Running Tests

```bash
# All tests
pytest -v

# With coverage
pytest --cov=app --cov-report=html
open htmlcov/index.html

# Specific test file
pytest tests/test_flows/test_weather_agent.py -v

# Specific test
pytest tests/test_flows/test_weather_agent.py::test_weather_agent_basic -v

# With output (see prints)
pytest -v -s

# Stop on first failure
pytest -x
```

### Writing Tests

#### Unit Tests

```python
import pytest
from app.your_module import your_function


def test_your_function():
    """Test your function."""
    result = your_function(input_data)
    assert result == expected_output


@pytest.mark.asyncio
async def test_async_function():
    """Test async function."""
    result = await async_function()
    assert result is not None
```

#### Integration Tests (with Database)

```python
import pytest
from sqlmodel import Session
from app.db.base import get_session
from app.db.models import Conversation


@pytest.mark.asyncio
async def test_database_operation():
    """Test database interaction."""
    with Session(get_session()) as session:
        # Create test data
        conv = Conversation(
            conversation_id="test-123",
            tenant_id="test",
            flow_type="test",
        )
        session.add(conv)
        session.commit()

        # Test query
        result = session.get(Conversation, conv.id)
        assert result.conversation_id == "test-123"

        # Cleanup
        session.delete(conv)
        session.commit()
```

#### Mocking LLM Calls

```python
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
@patch("app.adapters.openai_adapter.OpenAIAdapter.generate_completion")
async def test_flow_with_mock_llm(mock_llm):
    """Test flow with mocked LLM."""
    # Setup mock
    mock_llm.return_value = {
        "content": "Mocked response",
        "finish_reason": "stop",
    }

    # Run flow
    flow = YourAgentFlow()
    result = await flow.run(
        user_message="test",
        tenant_id="test",
    )

    # Verify
    assert result["response"] == "Mocked response"
    mock_llm.assert_called_once()
```

### Test Structure

```
tests/
├── test_main.py              # API endpoint tests
├── test_config.py            # Configuration tests
│
├── test_db/
│   ├── test_models.py        # Model tests
│   └── test_repositories.py  # Repository tests
│
├── test_adapters/
│   ├── test_openai.py        # OpenAI adapter tests
│   └── test_vertex.py        # Vertex adapter tests
│
├── test_flows/
│   └── test_weather_agent.py # Flow tests
│
└── test_tools/
    └── test_weather/
        └── test_weather_tool.py  # Tool tests
```

### Coverage Goals

- Overall: 80%+
- Critical paths: 90%+
- LLM adapters: 85%+
- Database operations: 90%+

---

## Deployment

### Local Deployment

```bash
# Start Cloud SQL Proxy (Terminal 1)
./dev db-proxy

# In Terminal 2:
# Run migrations
alembic upgrade head

# Start API
uvicorn app.main:app --reload --port 8080

# Or use the dev CLI:
./dev server
```

### GCP Deployment (Production)

**Prerequisites:**
- GCP project configured
- Cloud Run, Cloud SQL, Artifact Registry enabled
- Secrets in Secret Manager

**Deploy via CI/CD:**
```bash
git add .
git commit -m "feat: your changes"
git push origin main  # Auto-deploys to production
```

**Manual Deploy:**
```bash
# Build and push Docker image
./scripts/deploy_to_gcp.sh

# Or use gcloud directly
gcloud run deploy ai-agency \
  --image europe-west1-docker.pkg.dev/PROJECT/ai-agency/app:latest \
  --region europe-west1 \
  --platform managed
```

**Environment Variables in Cloud Run:**

Set in Cloud Run console or via gcloud:
```bash
gcloud run services update ai-agency \
  --set-env-vars DATABASE_URL=secret-value \
  --region europe-west1
```

Use Secret Manager for sensitive values:
```bash
gcloud run services update ai-agency \
  --set-secrets OPENAI_API_KEY=openai-key:latest \
  --region europe-west1
```

**Database Migrations:**

Run migrations before deploying:
```bash
# Start Cloud SQL Proxy (Terminal 1)
./dev db-proxy

# In another terminal (Terminal 2)
# Get password from Secret Manager
gcloud secrets versions access latest --secret='cloud-sql-password'

# Set DATABASE_URL with password
export DATABASE_URL="postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5433/ai_agency"
alembic upgrade head
```

### Production URLs

- **API**: https://ai-agency-4ebxrg4hdq-ew.a.run.app
- **Docs**: https://ai-agency-4ebxrg4hdq-ew.a.run.app/docs
- **Region**: europe-west1 (Belgium)

---

## Troubleshooting

### Database Issues

**Connection refused:**
```bash
# Check if Cloud SQL Proxy is running
./dev db-check

# Restart proxy
pkill -f cloud-sql-proxy
./dev db-proxy

# Check connection string
echo $DATABASE_URL
```

**Migration conflicts:**
```bash
# Check current version
alembic current

# Check migration history
alembic history

# Rollback and reapply
alembic downgrade -1
alembic upgrade head
```

**Cloud SQL Proxy not working:**
```bash
# Kill existing proxies
pkill cloud_sql_proxy

# Start fresh
./scripts/start_cloudsql_proxy.sh

# Check logs
tail -f /tmp/cloudsql_proxy.log
```

### LLM Issues

**OpenAI API errors:**
```bash
# Check API key
echo $OPENAI_API_KEY

# Test API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

**Rate limiting:**
- Check `app/config.py` rate limits
- Wait and retry
- Consider using different provider

**Vertex AI authentication:**
```bash
# Check authentication
gcloud auth list

# Login
gcloud auth application-default login

# Set project
gcloud config set project YOUR_PROJECT_ID
```

### Import Errors

```bash
# Reinstall in dev mode
pip install -e ".[dev]"

# Check installation
pip list | grep ai-agency

# Clear Python cache
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete
```

### Port Already in Use

```bash
# Find process on port 8080
lsof -i :8080

# Kill process
kill -9 <PID>

# Or use different port
uvicorn app.main:app --reload --port 8081
```

### Tests Failing

```bash
# Clear pytest cache
pytest --cache-clear

# Run with verbose output
pytest -vv -s

# Run single test
pytest tests/test_main.py::test_healthz -vv

# Check test database
echo $DATABASE_URL
```

### Cloud SQL Proxy Issues

```bash
# Check if proxy is running
./dev db-check

# Restart proxy
pkill -f cloud-sql-proxy
./dev db-proxy

# Verify credentials
gcloud secrets versions access latest --secret='cloud-sql-password'

# Test connection directly
PGPASSWORD='YOUR_PASSWORD' psql -h localhost -p 5433 -U postgres -d ai_agency -c "\dt"

# Check IAM permissions
gcloud projects get-iam-policy merlin-notebook-lm \
  --flatten="bindings[].members" \
  --filter="bindings.role:roles/cloudsql.client"
```

---

## Common Tasks

### Add Environment Variable

1. Add to `.env.example`:
```bash
YOUR_VAR=default_value
```

2. Add to `app/config.py`:
```python
class Settings(BaseSettings):
    your_var: str = "default_value"
```

3. Use in code:
```python
from app.config import settings
value = settings.your_var
```

### Add Database Model

1. Add to `app/db/models.py`:
```python
class YourModel(SQLModel, table=True):
    __tablename__ = "your_table"

    id: int | None = Field(default=None, primary_key=True)
    tenant_id: str = Field(index=True)
    name: str
```

2. Create migration:
```bash
alembic revision --autogenerate -m "Add your_table"
```

3. Review and apply:
```bash
alembic upgrade head
```

### Add New Tool

1. Create `app/tools/your_tool/v1.py`
2. Implement tool function
3. Add tool definition for LLM
4. Use in your flow

### Add API Endpoint

Add to `app/main.py`:
```python
@app.post("/your-endpoint")
async def your_endpoint(data: dict):
    # Your logic
    return {"status": "ok"}
```

### Run Code Quality Checks

```bash
# Format code
ruff format app tests

# Lint code
ruff check app tests --fix

# Type check
mypy app

# All checks
ruff format app tests && ruff check app tests --fix && mypy app
```

---

## Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/your-feature

# 2. Make changes

# 3. Run tests
pytest -v

# 4. Code quality
ruff format app tests
ruff check app tests --fix
mypy app

# 5. Test coverage
pytest --cov=app --cov-report=html

# 6. Commit
git add .
git commit -m "feat: your feature description"

# 7. Push
git push origin feature/your-feature

# 8. Create PR (merging to main auto-deploys)
```

---

## Quick Reference

### Essential Commands

```bash
# Development
./dev db-proxy              # Start Cloud SQL Proxy
./dev server                # Start development server
./dev db-check              # Check if proxy is running
uvicorn app.main:app --reload --port 8080  # Alternative server start

# Testing
pytest -v --cov=app

# Code Quality
ruff format app tests
ruff check app tests --fix
mypy app

# Database
alembic revision --autogenerate -m "description"
alembic upgrade head
alembic current

# Git
git checkout -b feature/name
git add .
git commit -m "feat: description"
git push origin feature/name
```

### API Endpoints

- `GET /health` - Health check
- `GET /docs` - API documentation
- `POST /weather/chat` - Weather agent
- More endpoints in `app/main.py`

### Environment Variables

Key variables in `.env`:
- `DATABASE_URL` - Database connection
- `OPENAI_API_KEY` - OpenAI API key
- `LLM_PROVIDER` - openai|vertex|mistral
- `GCP_PROJECT_ID` - GCP project
- `LOG_LEVEL` - INFO|DEBUG|WARNING

See `.env.example` for full list.

---

## Additional Resources

- [QUICKSTART.md](QUICKSTART.md) - 5-minute setup
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- Production: https://ai-agency-4ebxrg4hdq-ew.a.run.app/docs
- OpenAPI Spec: https://ai-agency-4ebxrg4hdq-ew.a.run.app/openapi.json
