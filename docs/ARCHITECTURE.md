# Architecture

Technical architecture and design decisions for the AI Consulting Agency Platform.

## Table of Contents

- [System Overview](#system-overview)
- [Multi-Flow Conversation Architecture](#multi-flow-conversation-architecture)
- [Infrastructure](#infrastructure)
- [LLM Adapters](#llm-adapters)
- [Technology Choices](#technology-choices)

---

## System Overview

The AI Consulting Agency Platform is a multi-tenant, multi-LLM conversational AI system for business consulting workflows.

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                         FastAPI API                         │
│  GET /health, POST /weather/chat, POST /runs, etc.         │
└──────────────┬──────────────────────────────────────────────┘
               │
               ├─────────────┬────────────────┬───────────────┐
               │             │                │               │
         ┌─────▼──────┐ ┌───▼────────┐ ┌────▼─────┐  ┌──────▼──────┐
         │   Flows    │ │  Adapters  │ │  Tools   │  │     RAG     │
         │  (Agents)  │ │   (LLMs)   │ │(Business)│  │  (Future)   │
         └─────┬──────┘ └─────┬──────┘ └────┬─────┘  └──────┬──────┘
               │              │              │                │
               └──────────────┴──────────────┴────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   PostgreSQL DB    │
                    │  (Cloud SQL/Local) │
                    └────────────────────┘
```

### Tech Stack

- **Backend Framework**: FastAPI (async Python)
- **Database**: PostgreSQL 15 with pgvector extension
- **ORM**: SQLModel (SQLAlchemy + Pydantic)
- **Migrations**: Alembic
- **LLM Providers**: OpenAI, Google Vertex AI, Mistral
- **Cloud Platform**: Google Cloud Platform
- **Containerization**: Docker
- **CI/CD**: GitHub Actions

---

## Multi-Flow Conversation Architecture

The platform uses a **unified conversation architecture** that supports multiple agent types (weather, GitHub, Slack, etc.) through a single database schema.

### Design Principle: Single Table with flow_type

Instead of separate tables for each agent type, we use:
- ONE `conversations` table with `flow_type` field
- ONE `messages` table with `flow_type` field (denormalized for performance)

**Why this design?**

1. **Simplicity**: One codebase for conversation storage/retrieval
2. **Scalability**: Adding new agent types requires no schema changes
3. **Performance**: Composite indexes on `(tenant_id, flow_type, created_at)` make queries efficient
4. **Consistency**: All agents use the same `ConversationRepository` API
5. **Multi-tenant**: Easy tenant isolation with indexed `tenant_id` field

### Database Schema

#### conversations Table

```sql
CREATE TABLE conversations (
    id                SERIAL PRIMARY KEY,
    conversation_id   VARCHAR UNIQUE NOT NULL,  -- UUID
    tenant_id         VARCHAR NOT NULL,
    flow_type         VARCHAR NOT NULL,         -- 'weather', 'github', 'slack', etc.
    flow_metadata     JSONB DEFAULT '{}',
    created_at        TIMESTAMP NOT NULL,
    updated_at        TIMESTAMP NOT NULL
);

-- Composite indexes for efficient queries
CREATE INDEX idx_conversations_tenant_flow
    ON conversations(tenant_id, flow_type, created_at);
CREATE INDEX idx_conversations_flow_type
    ON conversations(flow_type);
```

#### messages Table

```sql
CREATE TABLE messages (
    id                SERIAL PRIMARY KEY,
    conversation_id   VARCHAR NOT NULL,
    tenant_id         VARCHAR NOT NULL,          -- Denormalized
    flow_type         VARCHAR NOT NULL,          -- Denormalized for fast queries
    role              VARCHAR NOT NULL,          -- 'user', 'assistant', 'system', 'tool'
    content           TEXT NOT NULL,
    tool_calls        JSONB,                     -- For function calling
    message_metadata  JSONB DEFAULT '{}',
    created_at        TIMESTAMP NOT NULL
);

-- Composite indexes for fast conversation retrieval
CREATE INDEX idx_messages_conversation
    ON messages(conversation_id, created_at);
CREATE INDEX idx_messages_tenant_flow
    ON messages(tenant_id, flow_type, created_at);
```

### Why Denormalize flow_type in messages?

**Denormalization for Performance:**
- Allows fast queries without JOINing conversations table
- Common query: "Get all messages for tenant X with flow type Y"
- Trade-off: Slight data redundancy for significant performance gain
- Storage cost: Minimal (VARCHAR vs JOIN overhead)

**Example Query Performance:**
```sql
-- Fast (no JOIN needed)
SELECT * FROM messages
WHERE tenant_id = 'tenant_123'
  AND flow_type = 'weather'
ORDER BY created_at DESC
LIMIT 100;

-- vs. Slow (requires JOIN)
SELECT m.* FROM messages m
JOIN conversations c ON m.conversation_id = c.conversation_id
WHERE c.tenant_id = 'tenant_123'
  AND c.flow_type = 'weather'
ORDER BY m.created_at DESC
LIMIT 100;
```

### How Flows Use ConversationRepository

All agent flows follow the same pattern:

```python
from app.db.repositories.conversation_repository import ConversationRepository

with Session(get_session()) as session:
    repo = ConversationRepository(session)

    # 1. Create or load conversation
    conv_id = repo.create_conversation(
        tenant_id="tenant_123",
        flow_type="weather",  # Or "github", "slack", etc.
    )

    # 2. Load history
    history = repo.get_conversation_history(conv_id)

    # 3. Save user message
    repo.save_message(
        conversation_id=conv_id,
        tenant_id="tenant_123",
        flow_type="weather",
        role="user",
        content="What's the weather?",
    )

    # 4. Save assistant response
    repo.save_message(
        conversation_id=conv_id,
        tenant_id="tenant_123",
        flow_type="weather",
        role="assistant",
        content="It's 15°C and sunny",
    )
```

### Composite Indexes Explained

**Index on conversations(tenant_id, flow_type, created_at):**
- Query: "Show all weather conversations for tenant_123 ordered by date"
- Fast because index covers all query columns in order
- PostgreSQL can use index scan instead of sequential scan

**Index on messages(conversation_id, created_at):**
- Query: "Get all messages in conversation ABC ordered by date"
- Critical for loading conversation history efficiently
- Covers 99% of message retrieval queries

**Index on messages(tenant_id, flow_type, created_at):**
- Query: "Get recent weather messages across all conversations for tenant_123"
- Useful for analytics and debugging
- Supports tenant-wide queries without scanning entire table

### Scaling Considerations

**Current design scales to:**
- 10M+ conversations
- 100M+ messages
- 1000+ tenants
- 10+ agent types

**When to shard:**
- If single-tenant exceeds 50M messages
- If queries become slow despite indexes
- Consider sharding by tenant_id or time-based partitioning

---

## Infrastructure

### Local Development

```
┌──────────────────────────┐
│  Your Computer           │
│                          │
│  ┌────────────┐          │
│  │  FastAPI   │          │  Port 8080
│  │  (uvicorn) │          │
│  └──────┬─────┘          │
│         │                │
│  ┌──────▼──────────┐     │
│  │ Cloud SQL Proxy │     │  Port 5433
│  │  (PostgreSQL)   │     │
│  └────────┬────────┘     │
│           │              │
│           └──────────────┼──────┐
│                          │      │
└──────────────────────────┘      │
                                  │
                    ┌─────────────▼────────────┐
                    │   Google Cloud           │
                    │   Cloud SQL              │
                    │   PostgreSQL 15          │
                    └──────────────────────────┘
```

**Components:**
- FastAPI app running via uvicorn
- PostgreSQL 15 via Cloud SQL Proxy (connects to Cloud SQL)
- Local filesystem for logs

**Start:**
```bash
# Start Cloud SQL Proxy (connects to Cloud SQL instance)
./dev db-proxy
# Or manually:
# cloud-sql-proxy --port 5433 PROJECT_ID:REGION:INSTANCE_NAME

# In another terminal, start FastAPI
uvicorn app.main:app --reload --port 8080
```

### Production (GCP)

```
┌─────────────────────────────────────────────────────────┐
│                    Google Cloud                         │
│                                                         │
│  ┌───────────────────────┐                             │
│  │     Cloud Run         │  Region: europe-west1       │
│  │  (FastAPI Container)  │                             │
│  │  Auto-scaling: 0-10   │                             │
│  └───────┬───────────────┘                             │
│          │                                              │
│          ├─────────────┬────────────────┬──────────────┤
│          │             │                │              │
│   ┌──────▼──────┐ ┌───▼────────┐ ┌────▼──────┐ ┌────▼─────┐
│   │  Cloud SQL  │ │   Secret   │ │Artifact   │ │   GCS    │
│   │ PostgreSQL  │ │  Manager   │ │ Registry  │ │(Buckets) │
│   │  (Primary)  │ │(API Keys)  │ │(Images)   │ │(Files)   │
│   └─────────────┘ └────────────┘ └───────────┘ └──────────┘
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Cloud Run:**
- Region: europe-west1 (Belgium)
- Container: europe-west1-docker.pkg.dev/PROJECT/ai-agency/app:latest
- Auto-scaling: 0-10 instances
- CPU: 2 vCPU per instance
- Memory: 4 GB per instance
- Timeout: 300s (5 minutes)

**Cloud SQL:**
- PostgreSQL 15
- Region: europe-west1
- High availability: Enabled (standby in europe-west1-b)
- Automatic backups: Daily at 3 AM UTC
- Point-in-time recovery: 7 days
- Connection: Via Unix socket (/cloudsql/PROJECT:REGION:INSTANCE)

**Secret Manager:**
- OPENAI_API_KEY: OpenAI API key
- DATABASE_PASSWORD: Cloud SQL password
- MISTRAL_API_KEY: Mistral API key (if enabled)

**Artifact Registry:**
- Repository: ai-agency
- Format: Docker
- Location: europe-west1

**Cloud Storage (GCS):**
- Bucket: merlin-ai-agency-artifacts-eu
- Location: europe-west1
- Usage: Document storage, generated reports

### CI/CD Pipeline (GitHub Actions)

**On Pull Request: `.github/workflows/ci.yml`**
```
1. Checkout code
2. Setup Python 3.11
3. Install dependencies
4. Run Ruff (lint + format check)
5. Run mypy (type check)
6. Start PostgreSQL + pgvector
7. Run pytest with coverage
8. Upload coverage report
```

**On Push to main: `.github/workflows/deploy.yml`**
```
1. Checkout code
2. Authenticate to GCP
3. Build Docker image
4. Push to Artifact Registry (europe-west1)
5. Deploy to Cloud Run (europe-west1)
6. Run smoke tests (GET /health)
7. Notify on failure
```

**Auto-deployment:**
- Every push to main triggers deployment
- Zero-downtime deployment (gradual traffic shift)
- Automatic rollback on health check failure

---

## LLM Adapters

The platform uses a **provider-agnostic adapter pattern** for LLM integration.

### Architecture

```
┌─────────────────────────────────────────────────┐
│              Flow Code                          │
│  (Weather, GitHub, Slack agents)               │
└────────────────────┬────────────────────────────┘
                     │
                     │ llm.generate_completion()
                     │
          ┌──────────▼───────────┐
          │    LLM Factory       │
          │  (Provider Router)   │
          └──────────┬───────────┘
                     │
       ┌─────────────┼─────────────┐
       │             │             │
┌──────▼─────┐ ┌────▼──────┐ ┌───▼────────┐
│  OpenAI    │ │  Vertex   │ │  Mistral   │
│  Adapter   │ │  Adapter  │ │  Adapter   │
└──────┬─────┘ └────┬──────┘ └───┬────────┘
       │             │             │
┌──────▼─────┐ ┌────▼──────┐ ┌───▼────────┐
│ OpenAI API │ │Vertex API │ │Mistral API │
└────────────┘ └───────────┘ └────────────┘
```

### Base LLM Interface

All adapters implement the same interface (`BaseLLM`):

```python
class BaseLLM:
    async def generate_completion(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ) -> dict:
        """Generate completion from LLM."""
        pass
```

**Benefits:**
- Flow code doesn't depend on specific provider
- Easy to add new providers
- Can switch providers via environment variable
- Testing with mock adapters

### Provider Implementations

#### OpenAI Adapter

```python
# Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4-turbo-2024-04-09

# Features
- Function calling (tools)
- Streaming (future)
- JSON mode
- Vision (future)

# Rate Limits (configurable)
- 90,000 tokens/minute
- 5,000,000 tokens/hour
```

#### Vertex AI Adapter (Google Gemini)

```python
# Configuration
LLM_PROVIDER=vertex
VERTEX_AI_PROJECT_ID=your-project
VERTEX_AI_LOCATION=us-central1
VERTEX_AI_MODEL=gemini-2.0-flash-exp

# Features
- Function calling (tools)
- Multimodal (text + images)
- Grounding (web search)
- Safety filters

# Rate Limits
- No hard limits (usage-based pricing)
- Concurrent request limits apply
```

#### Mistral Adapter

```python
# Configuration
LLM_PROVIDER=mistral
MISTRAL_API_KEY=...
MISTRAL_MODEL=mistral-medium-latest

# Features
- Function calling (tools)
- Fast inference
- European data residency

# Rate Limits (configurable)
- 2,000,000 tokens/minute
- 100,000,000 tokens/hour
```

### Rate Limiting

Rate limiting is implemented per-tenant and per-provider:

**Algorithm:** Token bucket with sliding window
**Granularity:** Per tenant + provider combination
**Tracking:** In-memory (Redis in future for distributed systems)

**Example:**
- Tenant "acme_corp" using OpenAI
- Limit: 90,000 tokens/minute
- Request consumes 1,500 tokens
- Bucket tracks: tokens used in last 60 seconds
- If bucket full -> return 429 Rate Limit Exceeded

**Configuration in `app/config.py`:**
```python
openai_rate_limit_tpm: int = 90000
openai_rate_limit_tph: int = 5000000
mistral_rate_limit_tpm: int = 2000000
mistral_rate_limit_tph: int = 100000000
```

### Function Calling (Tool Use)

**Implementation Status:**
- ✅ OpenAI - Full support via native API
- ✅ Mistral - Full support (OpenAI-compatible format)
- ✅ Vertex AI - Full support with automatic format conversion

**Architecture:**

Tool calling is implemented at the adapter interface level, making it provider-agnostic:

```
┌──────────────────────────────────────────────────┐
│            Flow Code (Weather Agent)             │
└────────────────────┬─────────────────────────────┘
                     │
                     │ complete_with_metadata(
                     │   messages, tools, tool_choice)
                     │
          ┌──────────▼───────────┐
          │    BaseAdapter       │
          │  (Interface Layer)   │
          └──────────┬───────────┘
                     │
       ┌─────────────┼─────────────┐
       │             │             │
┌──────▼─────┐ ┌────▼──────┐ ┌───▼────────┐
│  OpenAI    │ │  Mistral  │ │  Vertex    │
│ Native API │ │ Compatible│ │  Convert   │
└────────────┘ └───────────┘ └────────────┘
```

**Type Definitions:**

```python
# From app/core/types.py

class ToolFunction(TypedDict):
    """Function definition for tool calling."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema

class ToolDefinition(TypedDict):
    """Complete tool definition (OpenAI format)."""
    type: str  # "function"
    function: ToolFunction

class ToolCall(TypedDict):
    """LLM's request to call a tool."""
    id: str
    type: str  # "function"
    function: ToolCallFunction

class LLMResponseWithTools(TypedDict):
    """LLM response that includes tool calls."""
    content: str
    model: str
    tokens_used: int
    finish_reason: str
    tool_calls: list[ToolCall] | None
```

**Adapter Interface:**

```python
# From app/core/base.py

class BaseAdapter:
    async def complete_with_metadata(
        self,
        messages: MessageHistory,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMResponse | LLMResponseWithTools:
        """Generate completion with optional tool calling."""
        pass
```

**Usage Example:**

```python
# Define tool
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"],
        },
    },
}

# Call LLM with tool
response = await llm.complete_with_metadata(
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[weather_tool],
    tool_choice="auto",
    tenant_id="user_123",
)

# Check for tool calls
if response.get("tool_calls"):
    for tool_call in response["tool_calls"]:
        # Extract details
        tool_name = tool_call["function"]["name"]
        tool_args = json.loads(tool_call["function"]["arguments"])

        # Execute tool
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

        # Call LLM again with tool result
        final_response = await llm.complete_with_metadata(
            messages=messages,
            tools=[weather_tool],
            tenant_id="user_123",
        )
```

**Provider Implementation Details:**

**OpenAI Adapter:**
- Uses native `tools` parameter in chat completions API
- Supports `tool_choice`: "none", "auto", or specific function
- Returns tool calls in standard format
- Handles multiple tool calls in single response

**Mistral Adapter:**
- Uses OpenAI-compatible tool calling format
- Same `tools` parameter structure
- Identical tool call response format
- No format conversion needed

**Vertex AI Adapter:**
- Automatic format conversion: OpenAI → Gemini FunctionDeclaration
- Generates deterministic tool_call_id (Vertex doesn't provide IDs)
- Uses `Tool` and `FunctionDeclaration` objects from Vertex AI SDK
- Converts function calls back to OpenAI format for consistency
- Handles multiple tool calls in single response

**Testing:**

Tool calling is tested comprehensively:
- 7 tests for OpenAI adapter (91% coverage)
- 6 tests for Mistral adapter (85% coverage)
- 9 tests for Vertex AI adapter (92% coverage)
- Tests cover: tool calls, multiple calls, format conversion, backward compatibility

**Reference Implementation:**

See `app/flows/agents/weather_agent.py` for complete multi-turn tool calling example.

---

## Technology Choices

### Why FastAPI?

**Chosen for:**
- Async/await support (critical for LLM calls)
- Automatic OpenAPI documentation
- Type hints with Pydantic validation
- High performance (Starlette + uvicorn)
- Modern Python (3.11+ features)

**Alternatives considered:**
- Django: Too heavyweight, sync-focused
- Flask: No built-in async, no type validation
- Express.js: Would require Node.js + Python bridge

### Why PostgreSQL?

**Chosen for:**
- ACID compliance (conversation integrity)
- JSONB support (flexible metadata)
- pgvector extension (future RAG)
- Rich indexing (composite, partial, GIN)
- Excellent Cloud SQL support

**Alternatives considered:**
- MongoDB: No ACID, schema flexibility not needed
- MySQL: Weaker JSON support, no pgvector
- SQLite: Not production-ready for multi-tenant

### Why SQLModel?

**Chosen for:**
- Combines SQLAlchemy + Pydantic
- Single definition for DB model and API schema
- Type safety throughout stack
- FastAPI native integration
- Alembic support for migrations

**Alternatives considered:**
- Raw SQLAlchemy: No Pydantic integration
- Django ORM: Tied to Django framework
- Tortoise ORM: Less mature, smaller community

### Why Alembic?

**Chosen for:**
- Industry standard for SQLAlchemy migrations
- Auto-generate migrations from models
- Version control for schema changes
- Rollback support
- Works with SQLModel

**Alternatives considered:**
- Django migrations: Requires Django
- Flyway: Java-based, overkill
- Raw SQL: Error-prone, no versioning

### Why Multi-Provider LLM Support?

**Reasons:**
1. **Cost optimization**: Switch to cheaper provider for simple tasks
2. **Reliability**: Fallback if one provider is down
3. **Regional compliance**: Use EU-based models (Mistral) for GDPR
4. **Feature access**: Use best model for specific tasks
5. **Vendor independence**: Not locked to single provider

**Trade-offs:**
- More complex adapter layer
- Testing across providers
- Slight performance overhead (adapter abstraction)

**Worth it because:**
- LLM market is rapidly evolving
- Provider pricing changes frequently
- Different providers excel at different tasks

### Why Google Cloud Platform?

**Chosen for:**
- Cloud Run: Serverless, auto-scaling, pay-per-use
- Cloud SQL: Managed PostgreSQL with automatic backups
- Vertex AI: Native Google Gemini support
- Secret Manager: Secure credential storage
- European regions: GDPR compliance (europe-west1)

**Alternatives considered:**
- AWS: More complex, Lambda limitations
- Azure: Less competitive pricing in EU
- Heroku: Less control, limited scalability

### Why Docker?

**Chosen for production deployment:**
- Cloud Run native format (requires containers)
- Consistent production environment
- Dependency isolation
- Version control of environment
- Multi-stage builds for smaller images

**Note on local development:**
- Docker is used ONLY for production deployment to Cloud Run
- Local development uses Cloud SQL Proxy (not Docker PostgreSQL)
- This ensures dev database matches production Cloud SQL exactly

**Build strategy:**
- Multi-stage builds (smaller images)
- Layer caching (faster builds)
- Alpine base (security + size)

### Why GitHub Actions?

**Chosen for:**
- Free for public repos
- Native GitHub integration
- YAML-based configuration
- Extensive marketplace
- Easy GCP authentication

**Alternatives considered:**
- GitLab CI: Would require migration
- Jenkins: Too heavyweight for our needs
- CircleCI: Additional cost

---

## Security Architecture

### Multi-Tenancy

**Isolation strategy:**
- Tenant ID in every table (indexed)
- Row-level security via application logic
- No shared data between tenants
- Separate API keys per tenant (future)

**Query pattern:**
```sql
SELECT * FROM conversations
WHERE tenant_id = 'tenant_123'  -- Always filter by tenant
```

### Secrets Management

**Development:**
- `.env` file (git-ignored)
- Local environment variables

**Production:**
- GCP Secret Manager
- Secrets mounted as environment variables in Cloud Run
- Automatic rotation support (future)

**Never in code:**
- API keys
- Database passwords
- Service account keys

### API Security (Future)

**Planned:**
- API key authentication
- Rate limiting per tenant
- Request signing
- IP allowlisting for enterprise

**Current state:**
- Open API (development only)
- Behind GCP IAM in production

---

## Performance Considerations

### Database Queries

**Optimizations:**
- Composite indexes on (tenant_id, flow_type, created_at)
- EXPLAIN ANALYZE for slow queries
- Connection pooling (SQLAlchemy)
- Query result caching (future)

**Monitoring:**
- Cloud SQL Insights for query performance
- Slow query log (> 1 second)

### LLM Calls

**Optimizations:**
- Async/await for parallel calls
- Streaming responses (future)
- Result caching for common queries (future)
- Rate limiting to prevent abuse

**Typical latencies:**
- OpenAI: 1-3 seconds
- Vertex AI: 0.5-2 seconds
- Mistral: 0.5-1.5 seconds

### Cloud Run Auto-scaling

**Settings:**
- Min instances: 0 (cost optimization)
- Max instances: 10 (prevents runaway costs)
- Concurrency: 80 requests per instance
- Startup time: ~2 seconds (cold start)

**Scaling triggers:**
- CPU utilization > 60%
- Request queue depth > 10

---

## Future Architecture Changes

### Phase 1: RAG Implementation (Wave 3)
- Add pgvector extension to PostgreSQL
- Implement document ingestion pipeline
- Add vector search for context retrieval
- Integrate retrieval into LLM prompts

### Phase 2: Distributed Rate Limiting
- Move rate limiting from in-memory to Redis
- Support multiple Cloud Run instances
- Implement distributed token bucket

### Phase 3: Observability
- Add structured logging (JSON)
- Implement distributed tracing (OpenTelemetry)
- Set up Grafana dashboards
- Alert on error rates > 1%

### Phase 4: Advanced Security
- Add API key authentication
- Implement OAuth2 for user authentication
- Add audit logging for all operations
- Enable Cloud SQL IAM authentication (Cloud SQL proxy already implemented)

---

## Deployment Diagram

```
GitHub Repository (main branch)
        │
        ├─ Push commit
        │
        ▼
┌───────────────────────────────────────┐
│     GitHub Actions (deploy.yml)      │
│                                       │
│  1. Build Docker image                │
│  2. Push to Artifact Registry         │
│  3. Deploy to Cloud Run               │
└───────────────────┬───────────────────┘
                    │
                    ▼
┌───────────────────────────────────────┐
│   GCP europe-west1                    │
│                                       │
│   ┌─────────────────────────┐        │
│   │   Cloud Run Service     │        │
│   │   ai-agency             │        │
│   │                         │        │
│   │   Revision: 006 ◄── Active      │
│   │   Traffic: 100%         │        │
│   │   Instances: 0-10       │        │
│   └──────────┬──────────────┘        │
│              │                        │
│   ┌──────────▼──────────────┐        │
│   │     Cloud SQL           │        │
│   │   PostgreSQL 15         │        │
│   │   High Availability     │        │
│   └─────────────────────────┘        │
│                                       │
└───────────────────────────────────────┘
```

---

## Summary

The AI Consulting Agency Platform is built on:

1. **Multi-flow architecture**: Single database schema for all agent types
2. **Provider-agnostic LLM**: Support for OpenAI, Vertex AI, Mistral
3. **Cloud-native**: Serverless on GCP with auto-scaling
4. **Developer-friendly**: FastAPI, SQLModel, comprehensive type hints
5. **Production-ready**: CI/CD, monitoring, high availability

**Key architectural decisions:**
- PostgreSQL for conversation storage (ACID + performance)
- Composite indexes for efficient multi-tenant queries
- Adapter pattern for LLM flexibility
- Cloud Run for serverless deployment
- GitHub Actions for CI/CD automation

**Scales to:**
- 10M+ conversations
- 100M+ messages
- 1000+ tenants
- 10+ agent types
