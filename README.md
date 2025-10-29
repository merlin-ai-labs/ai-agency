# AI Consulting Agency Platform

Multi-tenant conversational AI platform with multi-LLM support (OpenAI, Vertex AI, Mistral) and production-ready infrastructure on Google Cloud Run.

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

## Quick Links

- **[Get Started →](docs/QUICKSTART.md)** - Complete setup guide with GCP authentication
- **[Development Guide →](docs/DEVELOPER_GUIDE.md)** - Build agents, flows, and tools
- **[Coding Standards →](docs/CODING_STANDARDS.md)** - **READ THIS FIRST** before coding
- **[Architecture →](docs/ARCHITECTURE.md)** - System design and technical decisions
- **[Live API →](https://ai-agency-4ebxrg4hdq-ew.a.run.app/docs)** - Production environment (europe-west1)

## Tech Stack

**Backend:**
- FastAPI (async) + Python 3.11+
- PostgreSQL 15 + pgvector (Cloud SQL)
- Alembic migrations

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

## Project Structure

```
ConsultingAgency/
├── app/
│   ├── adapters/         # LLM provider adapters (OpenAI, Vertex AI, Mistral)
│   ├── flows/agents/     # Conversational agents (weather_agent = reference)
│   ├── tools/            # Tools agents can use (weather, etc.)
│   ├── db/               # Database models, repositories, migrations
│   ├── core/             # Base classes, rate limiter, exceptions
│   └── main.py           # FastAPI application
├── tests/                # Test suite (60+ tests for LLM adapters)
├── docs/                 # Documentation
├── dev                   # Development CLI (replaces docker-compose)
└── .claude/              # Claude Code agent configurations
```

## Development Workflow

```bash
# Setup (one time)
./dev setup              # Install deps, run migrations

# Daily workflow
./dev db-proxy           # Start Cloud SQL Proxy (separate terminal)
./dev server             # Start API server (http://localhost:8000)
./dev test               # Run tests
./dev quality            # Lint + format + type check

# See all commands
./dev help
```

## Example: Weather Agent

The weather agent is the **reference implementation** for building agents:

```bash
curl -X POST http://localhost:8000/weather/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the weather in London?",
    "tenant_id": "test-user"
  }'
```

See `app/flows/agents/weather_agent.py` for implementation patterns.

## Key Patterns

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

See [CODING_STANDARDS.md](docs/CODING_STANDARDS.md) for complete patterns.

## Production Deployment

**Live Service:** https://ai-agency-4ebxrg4hdq-ew.a.run.app

**Auto-deploys on:** Push to `main` branch

**CI/CD Pipeline:**
1. Run tests (pytest with coverage)
2. Lint + format (Ruff)
3. Type check (mypy)
4. Build Docker image
5. Run database migrations
6. Deploy to Cloud Run
7. Smoke test (`/health` endpoint)

**Monitoring:**
- Cloud Run metrics (CPU, memory, requests)
- Structured JSON logging
- Cloud SQL connection pooling

## Documentation Structure

1. **[QUICKSTART.md](docs/QUICKSTART.md)** - Start here! Complete setup in 10 minutes
2. **[CODING_STANDARDS.md](docs/CODING_STANDARDS.md)** - Read before writing code!
3. **[DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)** - Building agents, flows, and tools
4. **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design and decisions

## Development Phases

| Wave | Focus | Status |
|------|-------|--------|
| Wave 1 | Foundation (FastAPI, Cloud Run, CI/CD) | ✅ Complete |
| Wave 2 | Multi-LLM + Weather Agent | ✅ Complete |
| Wave 3 | RAG with pgvector | ⏳ Next |
| Wave 4 | Business flows (maturity, grooming) | ⏳ Planned |
| Wave 5 | Auth + 80% test coverage | ⏳ Planned |

## Contributing

1. Read [CODING_STANDARDS.md](docs/CODING_STANDARDS.md) - **mandatory**
2. Study the weather agent reference implementation
3. Write tests (aim for 80% coverage)
4. Run quality checks: `./dev quality`
5. Submit PR

## Support

**Getting Started:**
- Setup issues? → [QUICKSTART.md](docs/QUICKSTART.md)
- Code questions? → [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)
- Architecture questions? → [ARCHITECTURE.md](docs/ARCHITECTURE.md)

**Need Help?**
- Open an issue on GitHub
- Check existing issues first

---

**Current Status:** Production-ready conversational AI platform with multi-LLM support. Weather agent operational as reference implementation for building new agents.
