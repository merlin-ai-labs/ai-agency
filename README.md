# AI Consulting Agency Platform

Multi-tenant conversational AI platform with multi-LLM support (OpenAI, Vertex AI, Mistral) and production-ready infrastructure on Google Cloud Run.

## Table of Contents

- [What This Platform Does](#what-this-platform-does)
- [Tech Stack](#tech-stack)
- [Quick Setup (10 minutes)](#quick-setup-10-minutes)
- [Development](#development)
- [Building New Agents](#building-new-agents)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

---

## What This Platform Does

**Current Features (Wave 2 Complete):**
- ✅ Multi-LLM conversational agents (reference: Weather Agent)
- ✅ Multi-tenant conversation management with persistence
- ✅ Rate limiting per tenant + LLM provider
- ✅ Function calling / tool use support
- ✅ Production deployment on Cloud Run with auto-scaling

**Coming Soon:**
- 🚧 RAG with pgvector (Wave 3)
- 🚧 Business logic flows (maturity assessment, use case grooming)
- 🚧 Authentication & authorization

**Live API:** https://ai-agency-4ebxrg4hdq-ew.a.run.app/docs

---

## Tech Stack

**Backend:**
- FastAPI (async) + Python 3.11+
- PostgreSQL 15 + pgvector (Cloud SQL)
- Alembic migrations
- SQLModel (ORM)

**LLM Providers:**
- OpenAI (GPT-4.1, GPT-4.1-mini)
- Google Vertex AI (Gemini 2.0 Flash)
- Mistral (mistral-medium-latest)

**Infrastructure:**
- Google Cloud Run (containerized)
- Cloud SQL (PostgreSQL with HA)
- Cloud Storage (GCS)
- GitHub Actions (CI/CD)

**Local Development:**
- Cloud SQL Proxy (connects to production database)
- `./dev` CLI tool (unified development commands)

---

## Quick Setup (10 minutes)

### Prerequisites

- **Python 3.11+** - Check with `python3 --version`
- **Git** - Check with `git --version`
- **GCP Access** - You need access to the `merlin-notebook-lm` project
- **gcloud CLI** - [Install guide](https://cloud.google.com/sdk/docs/install)

### Step 1: GCP Authentication

```bash
# Login to GCP
gcloud auth login

# Set the project
gcloud config set project merlin-notebook-lm

# Verify authentication
gcloud auth list
# You should see your email marked as ACTIVE

# Enable application default credentials (required for Cloud SQL Proxy)
gcloud auth application-default login
```

### Step 2: Install Cloud SQL Proxy

```bash
# macOS
brew install cloud-sql-proxy

# Verify installation
cloud-sql-proxy --version
```

**Other platforms:** See [official install guide](https://cloud.google.com/sql/docs/postgres/sql-proxy#install)

### Step 3: Get Database Password

```bash
# Get the password (save this, you'll need it for .env)
gcloud secrets versions access latest --secret='cloud-sql-password'
```

**Save this password** - you'll use it in Step 6.

### Step 4: Start Cloud SQL Proxy

Open a **separate terminal window** and start the proxy (keep it running):

```bash
cloud-sql-proxy merlin-notebook-lm:europe-west1:ai-agency-db --port 5433
```

You should see:
```
Ready for new connections
```

**Keep this terminal open!** The proxy needs to run continuously while you develop.

### Step 5: Clone and Setup Project

In a **new terminal**, clone and set up the project:

```bash
# Clone repository
git clone git@github.com:merlin-ai-labs/ai-agency.git
cd ai-agency

# Run automated setup (creates venv, installs dependencies, runs migrations)
./dev setup
```

This will:
- ✅ Create Python virtual environment in `./venv`
- ✅ Install all Python dependencies
- ✅ Run database migrations (applies schema changes only - **preserves all data**)
- ✅ Create `.env` file from `.env.example`

**⚠️ Database Safety**: The setup command runs `alembic upgrade head` which only applies new schema migrations. It does NOT drop tables or delete data. Safe to run multiple times.

### Step 6: Configure Environment Variables

Edit the `.env` file created in the root directory:

```bash
# Open .env in your editor
nano .env  # or vim, vscode, etc.
```

**Required configuration:**

```bash
# Database - REQUIRED
# Replace YOUR_PASSWORD with the password from Step 3
DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5433/ai_agency

# LLM Provider - REQUIRED (choose one)
LLM_PROVIDER=openai  # or "vertex" or "mistral"

# OpenAI - REQUIRED if using OpenAI
OPENAI_API_KEY=sk-proj-...your-key-here

# Weather API - REQUIRED for weather agent
OPENWEATHER_API_KEY=your-key-here  # Get free key at https://openweathermap.org/api

# Google Cloud - Optional (only needed for Vertex AI or GCS)
GCP_PROJECT_ID=merlin-notebook-lm
VERTEX_AI_LOCATION=us-central1
GCS_BUCKET=your-artifacts-bucket

# Application - Optional
LOG_LEVEL=INFO
ENVIRONMENT=development
```

**Where to get API keys:**
- **OpenAI:** https://platform.openai.com/api-keys
- **OpenWeather:** https://home.openweathermap.org/api_keys (free tier available)

### Step 7: Start Development Server

```bash
./dev server
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### Step 8: Test It Works

Open your browser or use curl to test the API:

**1. Health Check:**
```bash
curl http://localhost:8000/health
# Expected: {"status":"healthy"}
```

**2. Interactive API Docs:**

Visit: http://localhost:8000/docs

**3. Try the Weather Agent:**
```bash
curl -X POST http://localhost:8000/weather/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the weather in London?",
    "tenant_id": "test-user"
  }'
```

---

## Development

### Daily Workflow

**Terminal 1 (Cloud SQL Proxy):**
```bash
cloud-sql-proxy merlin-notebook-lm:europe-west1:ai-agency-db --port 5433
# Keep running
```

**Terminal 2 (Development):**
```bash
cd ai-agency
./dev server        # Start API
# Make code changes
./dev test          # Run tests
./dev quality       # Check code quality
```

### Common Commands

```bash
./dev help         # Show all available commands
./dev db-check     # Verify Cloud SQL Proxy is running
./dev db-migrate   # Run database migrations
./dev db-seed      # Seed database with test data
./dev test         # Run all tests
./dev lint         # Run linter (Ruff)
./dev format       # Format code (Ruff)
./dev quality      # Run lint + format + type check
./dev clean        # Clean generated files (.pyc, __pycache__, etc.)
```

### Key Patterns

**1. Multi-tenant by design:**
```python
# Every conversation belongs to a tenant
conversation = await repo.create_conversation(
    tenant_id="company-123",
    flow_type="weather"
)
```

**2. Provider-agnostic LLM access:**
```python
# Works with OpenAI, Vertex AI, or Mistral
llm = get_llm_adapter(provider="openai")
response = await llm.generate(messages, tools=tools)
```

**3. Repository pattern for data access:**
```python
# Never access database directly
from app.db.repositories import ConversationRepository
repo = ConversationRepository(session)
await repo.create_conversation(tenant_id, flow_type)
```

**4. Decorators for cross-cutting concerns:**
```python
from app.core.decorators import log_execution, timeout

@log_execution
@timeout(30)
async def my_flow():
    pass
```

See `app/flows/agents/weather_agent.py` for complete reference implementation.

---

## Building New Agents

### Understanding the Architecture

All agent flows use:
- Single `conversations` table with `flow_type` field
- Single `messages` table with `flow_type` field (denormalized for performance)
- `ConversationRepository` for database operations
- Provider-agnostic LLM adapters

### Reference Implementation

Study `app/flows/agents/weather_agent.py` - it demonstrates:
- Loading conversation history
- Adding user messages
- Calling LLM with tool definitions
- Handling tool calls (function calling)
- Saving responses to database
- Multi-turn conversations

### Step-by-Step: Building a New Agent

#### 1. Create Flow Class

Create `app/flows/agents/your_agent.py`:

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
    """Your agent description."""

    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        super().__init__(
            name="your_agent",
            description="Your agent description",
            version="1.0.0",
        )
        self.llm = get_llm_adapter(provider=provider, model=model)

    @log_execution
    @timeout(seconds=120.0)
    async def run(
        self,
        user_message: str,
        tenant_id: str,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Process user message through your agent."""
        with Session(get_session()) as session:
            repo = ConversationRepository(session)

            # Create or load conversation
            if not conversation_id:
                conversation_id = repo.create_conversation(
                    tenant_id=tenant_id,
                    flow_type="your_flow_type",
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

#### 2. Create API Endpoint

Add to `app/main.py`:

```python
@app.post("/your-agent/chat")
async def your_agent_chat(
    user_message: str,
    tenant_id: str,
    conversation_id: str | None = None,
):
    """Your agent endpoint."""
    from app.flows.agents.your_agent import YourAgentFlow

    flow = YourAgentFlow()
    result = await flow.run(
        user_message=user_message,
        tenant_id=tenant_id,
        conversation_id=conversation_id,
    )
    return result
```

#### 3. Add to FlowType Enum

Update `app/db/models.py`:

```python
class FlowType(str, Enum):
    """Supported agent flow types."""
    WEATHER = "weather"
    YOUR_AGENT = "your_agent"  # Add this
```

### LLM Integration

```python
from app.adapters.llm_factory import get_llm_adapter

# Use configured provider (from settings)
llm = get_llm_adapter()

# Or specify provider
llm = get_llm_adapter(provider="openai")
llm = get_llm_adapter(provider="vertex")
llm = get_llm_adapter(provider="mistral")

# Generate completions
response = await llm.generate_completion(
    messages=[
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_tokens=1000,
)
print(response["content"])
```

**Supported Providers:**

**OpenAI** (default)
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

**Vertex AI (Google Gemini)**
```bash
LLM_PROVIDER=vertex
VERTEX_AI_PROJECT_ID=your-project
VERTEX_AI_LOCATION=us-central1
```

**Mistral**
```bash
LLM_PROVIDER=mistral
MISTRAL_API_KEY=...
```

### Tool Calling (Function Calling)

**Overview:**

Tool calling (also called function calling) allows LLMs to request execution of external functions/tools. The LLM decides when and how to call tools based on the conversation context.

**Supported Providers:**
- ✅ OpenAI - Native support
- ✅ Mistral - OpenAI-compatible format
- ✅ Vertex AI - Full support with automatic format conversion

**How It Works:**
1. Define tool schemas (function signatures with JSON Schema)
2. Pass tools to LLM adapter
3. LLM returns tool calls when needed
4. Execute tools and return results
5. LLM generates final response with tool results

#### Defining Tools

Tools are defined using OpenAI's function calling format:

```python
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name (e.g., 'London', 'Paris')",
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units",
                },
            },
            "required": ["location"],
        },
    },
}
```

#### Using Tools with Adapters

All adapters support tool calling through the same interface:

```python
from app.adapters.llm_factory import get_llm_adapter

# Initialize adapter (works with OpenAI or Mistral)
llm = get_llm_adapter()  # Uses configured provider

# Call LLM with tools
response = await llm.complete_with_metadata(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in Paris?"}
    ],
    tools=[weather_tool],  # Pass tool definitions
    tool_choice="auto",     # Let LLM decide when to use tools
    tenant_id="user_123",
)

# Check if LLM wants to call a tool
if response.get("tool_calls"):
    for tool_call in response["tool_calls"]:
        # Extract tool call details
        tool_name = tool_call["function"]["name"]
        tool_args = json.loads(tool_call["function"]["arguments"])

        # Execute your tool
        if tool_name == "get_weather":
            result = await get_weather(**tool_args)

        # Add tool result to conversation
        messages.append({
            "role": "assistant",
            "content": response["content"] or "",
            "tool_calls": [tool_call],
        })
        messages.append({
            "role": "tool",
            "content": json.dumps(result),
            "tool_call_id": tool_call["id"],
        })

        # Call LLM again with tool results
        final_response = await llm.complete_with_metadata(
            messages=messages,
            tools=[weather_tool],
            tenant_id="user_123",
        )

        assistant_message = final_response["content"]
else:
    # No tool call, use response directly
    assistant_message = response["content"]
```

#### Tool Choice Options

Control when tools are used:

```python
# Let LLM decide (default)
response = await llm.complete_with_metadata(
    messages=messages,
    tools=[weather_tool],
    tool_choice="auto",  # LLM decides
)

# Force LLM to NOT use tools
response = await llm.complete_with_metadata(
    messages=messages,
    tools=[weather_tool],
    tool_choice="none",  # Never call tools
)

# Force specific tool (OpenAI/Mistral only)
response = await llm.complete_with_metadata(
    messages=messages,
    tools=[weather_tool],
    tool_choice={
        "type": "function",
        "function": {"name": "get_weather"}
    },  # Must call this tool
)
```

#### Complete Example: Multi-Turn Tool Calling

See `app/flows/agents/weather_agent.py` for a complete reference implementation. Key pattern:

```python
# Step 1: Call LLM with tools
llm_response = await self.llm.complete_with_metadata(
    messages=messages,
    tools=[WEATHER_TOOL_DEFINITION],
    tool_choice="auto",
    tenant_id=tenant_id,
)

# Step 2: Check for tool calls
if llm_response.get("tool_calls"):
    # Step 3: Execute tool
    tool_call = llm_response["tool_calls"][0]
    tool_name = tool_call["function"]["name"]
    tool_args = json.loads(tool_call["function"]["arguments"])

    result = await execute_tool(tool_name, tool_args)

    # Step 4: Add tool messages to conversation
    messages.append({
        "role": "assistant",
        "content": llm_response["content"] or "",
        "tool_calls": [tool_call],
    })
    messages.append({
        "role": "tool",
        "content": json.dumps(result),
        "tool_call_id": tool_call["id"],
    })

    # Step 5: Call LLM again with tool results
    final_response = await self.llm.complete_with_metadata(
        messages=messages,
        tools=[WEATHER_TOOL_DEFINITION],
        tenant_id=tenant_id,
    )

    assistant_message = final_response["content"]
```

#### Tips for Tool Calling

**1. Clear descriptions are crucial:**
```python
# Bad
"description": "Gets weather"

# Good
"description": "Get current weather conditions including temperature, humidity, and conditions for a specific city. Returns temperature in specified units."
```

**2. Use specific parameter descriptions:**
```python
"location": {
    "type": "string",
    "description": "City name, e.g., 'London', 'New York', 'Tokyo'"
}
```

**3. Handle missing tool_call_id gracefully:**
```python
tool_call_id = tool_call.get("id", f"call_{uuid.uuid4().hex[:8]}")
```

**4. Always validate tool arguments:**
```python
tool_args = json.loads(tool_call["function"]["arguments"])
if "location" not in tool_args:
    # Handle missing required parameter
    pass
```

**5. Test with multiple providers:**
```python
# Test that tools work with both OpenAI and Mistral
@pytest.mark.parametrize("provider", ["openai", "mistral"])
async def test_tool_calling(provider):
    llm = get_llm_adapter(provider=provider)
    response = await llm.complete_with_metadata(
        messages=messages,
        tools=[tool],
    )
    # Verify tool calling works
```

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

# Stop on first failure
pytest -x
```

### Writing Tests

```python
import pytest
from app.flows.agents.your_agent import YourAgentFlow


@pytest.mark.asyncio
async def test_your_agent_basic():
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
async def test_your_agent_conversation():
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
./dev server
```

### GCP Deployment (Production)

**Auto-deploys on:** Push to `main` branch

**CI/CD Pipeline:**
1. Run tests (pytest with coverage)
2. Lint + format (Ruff)
3. Type check (mypy)
4. Build Docker image
5. Run database migrations
6. Deploy to Cloud Run
7. Smoke test (`/health` endpoint)

**Manual Deploy:**
```bash
git add .
git commit -m "feat: your changes"
git push origin main  # Auto-deploys to production
```

**Production URLs:**
- **API**: https://ai-agency-4ebxrg4hdq-ew.a.run.app
- **Docs**: https://ai-agency-4ebxrg4hdq-ew.a.run.app/docs
- **Region**: europe-west1 (Belgium)

---

## Troubleshooting

### Issue: `Cloud SQL Proxy not running`

**Solution:**
```bash
# Check if proxy is running
lsof -i :5433

# If nothing is running, start it:
cloud-sql-proxy merlin-notebook-lm:europe-west1:ai-agency-db --port 5433

# If port is in use by something else, kill it:
lsof -ti :5433 | xargs kill -9
# Then start proxy
```

### Issue: `Connection refused to database`

**Causes:**
1. Cloud SQL Proxy not running → Start it (see above)
2. Wrong password in `.env` → Get password: `gcloud secrets versions access latest --secret='cloud-sql-password'`
3. Not authenticated with GCP → Run: `gcloud auth application-default login`

### Issue: `OpenAI API key not found`

**Solution:**
Edit `.env` and add your OpenAI API key:
```bash
OPENAI_API_KEY=sk-proj-...your-actual-key
```

Get a key at: https://platform.openai.com/api-keys

### Issue: `Port 8000 already in use`

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000

# Kill it
lsof -ti :8000 | xargs kill -9

# Restart server
./dev server
```

### Issue: `Module not found` or import errors

**Solution:**
```bash
# Re-run setup to reinstall dependencies
./dev setup

# Verify Python version
python3 --version  # Should be 3.11+

# Activate virtual environment manually if needed
source venv/bin/activate
```

### Issue: `Permission denied` when running `./dev`

**Solution:**
```bash
# Make dev script executable
chmod +x ./dev

# Try again
./dev help
```

### Database Migration Issues

```bash
# Check current version
alembic current

# Check migration history
alembic history

# Rollback and reapply
alembic downgrade -1
alembic upgrade head
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

**Vertex AI authentication:**
```bash
# Check authentication
gcloud auth list

# Login
gcloud auth application-default login

# Set project
gcloud config set project merlin-notebook-lm
```

---

## Project Structure

```
ConsultingAgency/
├── app/
│   ├── adapters/              # LLM provider adapters (OpenAI, Vertex AI, Mistral)
│   ├── core/                  # Base classes, rate limiter, exceptions
│   ├── db/
│   │   ├── models.py          # SQLModel database models
│   │   ├── repositories/      # Repository pattern for data access
│   │   └── migrations/        # Alembic database migrations
│   ├── flows/
│   │   ├── agents/            # Conversational agents (weather_agent = reference)
│   │   ├── maturity_assessment/  # Business flow (stub)
│   │   └── usecase_grooming/  # Business flow (stub)
│   ├── rag/                   # RAG ingestion and retrieval (stub)
│   ├── tools/
│   │   ├── weather/           # Weather tool (fully implemented)
│   │   ├── parse_docs/        # Business tools (stubs)
│   │   ├── score_rubrics/
│   │   ├── gen_recs/
│   │   ├── rank_usecases/
│   │   └── write_backlog/
│   └── main.py                # FastAPI application entry point
├── tests/                     # Test suite (60+ tests for LLM adapters)
│   ├── test_adapters/         # LLM adapter tests
│   ├── test_tools/            # Tool tests
│   ├── test_db/               # Database tests
│   └── integration/           # Integration tests
├── alembic/                   # Alembic configuration
├── docs/                      # Documentation
│   ├── CODING_STANDARDS.md    # **READ THIS FIRST** before coding
│   ├── ARCHITECTURE.md        # System design and technical decisions
│   └── wave2/                 # Wave 2 planning and implementation logs
├── scripts/archive/           # Archived Wave 1 scripts
├── dev                        # Development CLI (replaces docker-compose)
└── .claude/                   # Claude Code agent configurations
```

### Key Files

- **`app/main.py`** - API endpoints and FastAPI app
- **`app/config.py`** - All configuration via environment variables
- **`app/db/models.py`** - Database table definitions
- **`app/db/repositories/conversation_repository.py`** - Conversation data access
- **`app/adapters/llm_factory.py`** - LLM provider selection
- **`app/flows/agents/weather_agent.py`** - **Reference implementation - study this!**

---

## Development Phases

| Wave | Focus | Status |
|------|-------|--------|
| Wave 1 | Foundation (FastAPI, Cloud Run, CI/CD) | ✅ Complete |
| Wave 2 | Multi-LLM + Weather Agent | ✅ Complete |
| Wave 3 | RAG with pgvector | ⏳ Next |
| Wave 4 | Business flows (maturity, grooming) | ⏳ Planned |
| Wave 5 | Auth + 80% test coverage | ⏳ Planned |

---

## Contributing

1. Read [docs/CODING_STANDARDS.md](docs/CODING_STANDARDS.md) - **mandatory**
2. Study the weather agent reference implementation
3. Write tests (aim for 80% coverage)
4. Run quality checks: `./dev quality`
5. Submit PR

---

## Quick Reference

### Essential Commands

```bash
# Development
./dev setup                # One-time setup
./dev server               # Start development server
./dev db-check             # Check if proxy is running
./dev test                 # Run tests
./dev quality              # Code quality checks

# Cloud SQL Proxy
cloud-sql-proxy merlin-notebook-lm:europe-west1:ai-agency-db --port 5433

# Database
alembic upgrade head       # Apply migrations
alembic current            # Check current version

# Testing
pytest -v --cov=app

# Code Quality
ruff format app tests
ruff check app tests --fix
```

### API Endpoints

- `GET /health` - Health check
- `GET /docs` - API documentation
- `POST /weather/chat` - Weather agent

### Environment Variables

Key variables in `.env`:
- `DATABASE_URL` - Database connection
- `OPENAI_API_KEY` - OpenAI API key
- `LLM_PROVIDER` - openai|vertex|mistral
- `GCP_PROJECT_ID` - GCP project
- `LOG_LEVEL` - INFO|DEBUG|WARNING

See `.env.example` for full list.

---

**Current Status:** Production-ready conversational AI platform with multi-LLM support. Weather agent operational as reference implementation for building new agents.
