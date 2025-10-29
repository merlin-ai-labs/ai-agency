# AI Consulting Agency Platform - Claude Code Configuration

This document provides guidance for Claude Code when working on this repository.

## Repository Overview

**Project**: AI Consulting Agency Platform - Multi-tenant conversational AI system for business workflows

**Current Status**: Wave 2 complete - Weather agent flow operational

### What's Implemented (Production-Ready)

#### Wave 1: Foundation Infrastructure
- ✅ **Backend Framework**: FastAPI with async/await support
- ✅ **Database**: PostgreSQL 15 with multi-flow conversation architecture
  - Tables: `conversations`, `messages` (with `flow_type` field)
  - Repository: `ConversationRepository` for all conversation operations
- ✅ **LLM Adapters**: Provider-agnostic architecture
  - OpenAI (GPT-4 Turbo, GPT-4.1)
  - Google Vertex AI (Gemini 2.0 Flash)
  - Mistral (mistral-medium-latest)
  - Unified interface via `get_llm_adapter()` factory
- ✅ **Rate Limiting**: Token bucket algorithm per tenant + provider
- ✅ **Core Infrastructure**:
  - Base classes: `BaseFlow`, `BaseTool`, custom exceptions
  - Decorators: `@log_execution`, `@timeout`
  - Structured logging with JSON output
  - Database migrations via Alembic
- ✅ **Deployment**:
  - Docker containerization
  - Google Cloud Run (europe-west1)
  - Cloud SQL (PostgreSQL with high availability)
  - GitHub Actions CI/CD pipeline
  - Automated migrations in deployment

#### Wave 2: Weather Agent (Reference Implementation)
- ✅ **Weather Agent Flow**: `app/flows/agents/weather_agent.py`
  - Multi-turn conversations with context
  - Function calling / tool use
  - Conversation history management
  - Provider-agnostic LLM integration
- ✅ **Weather Tool**: `app/tools/weather/`
  - OpenWeatherMap API integration
  - Error handling with retries
  - Type-safe implementation
- ✅ **API Endpoints**:
  - `POST /weather/chat` - Weather agent endpoint
  - `GET /health` - Health check
  - `GET /docs` - Swagger documentation

### What's Stub/Placeholder (Not Yet Implemented)

- ⚠️ **RAG**: Only placeholder files (`app/rag/ingestion.py`, `app/rag/retriever.py`)
- ⚠️ **Legacy Flows**: Maturity assessment and use-case grooming (stubs)
- ⚠️ **Legacy Tools**: parse_docs, score_rubrics, gen_recs, rank_usecases, write_backlog (stubs)
- ⚠️ **Authentication**: No API keys, no auth middleware
- ⚠️ **Advanced Security**: Input validation exists, but no RBAC or audit logging
- ⚠️ **Testing**: Minimal coverage (~8-15%), comprehensive testing not yet implemented
- ⚠️ **GCS Storage**: Configuration exists, but not actively used

### Reference Implementation Pattern

**⭐ The Weather Agent is the template for all future agents.**

When implementing new flows:
1. Study `app/flows/agents/weather_agent.py`
2. Follow the same patterns:
   - Use `ConversationRepository` for persistence
   - Use `get_llm_adapter()` for LLM access
   - Implement `run()` method with type hints
   - Add `@log_execution` and `@timeout` decorators
   - Save all messages (user + assistant) to DB
   - Handle tool calls following function calling pattern
3. See `docs/DEVELOPER_GUIDE.md` sections "Building New Flows" and "Step-by-Step"

---

## Specialized Agents Available

This repository uses specialized agent definitions in `.claude/agents/` for different development tasks. Each agent has detailed instructions and templates.

### When to Use Each Agent

#### 🏗️ Foundation & Architecture

**tech-lead** - Architectural decisions and foundational patterns
- **Status**: Wave 1 complete. Core infrastructure implemented.
- **Use when**: Making architectural decisions, establishing new patterns, refactoring core infrastructure
- **Example**: "Design authentication architecture" or "Establish error handling patterns for new module"

**devops-engineer** - Infrastructure, deployment, CI/CD
- **Status**: Wave 1 complete. Docker, Cloud Run, GitHub Actions operational.
- **Use when**: Changing deployment, adding new infrastructure, modifying CI/CD pipeline
- **Example**: "Add staging environment" or "Optimize Docker build process"

**database-engineer** - Database models, migrations, repositories
- **Status**: Wave 1 complete. Conversation schema + ConversationRepository implemented.
- **Use when**: Adding new tables, creating migrations, implementing repository patterns
- **Example**: "Create user_profiles table" or "Add index for performance optimization"

#### 🤖 AI & Integration

**llm-engineer** - LLM providers, adapters, prompting
- **Status**: Wave 1 complete. OpenAI, Vertex AI, Mistral adapters with rate limiting.
- **Use when**: Adding new LLM provider, updating models, optimizing prompts, implementing new LLM features
- **Example**: "Add Anthropic Claude provider" or "Implement streaming responses"

**rag-engineer** - Vector search, document chunking, embeddings
- **Status**: NOT implemented. Only stubs exist.
- **Use when**: Implementing RAG functionality (Wave 3+)
- **Example**: "Implement pgvector integration" or "Add document chunking pipeline"

**tools-engineer** - Business logic tools (like weather tool)
- **Status**: Weather tool complete. Legacy tools are stubs.
- **Use when**: Building new tools for agents to use
- **Example**: "Implement GitHub API tool" or "Create Slack integration tool"
- **Follow**: Weather tool pattern in `app/tools/weather/`

**flows-engineer** - Agent flows and orchestration
- **Status**: Weather agent complete (reference template). Legacy flows are stubs.
- **Use when**: Building new conversational agents
- **Example**: "Implement GitHub agent" or "Create sales agent flow"
- **Follow**: WeatherAgentFlow pattern in `app/flows/agents/weather_agent.py`

#### 🔒 Security & Quality

**security-engineer** - Authentication, authorization, security audit
- **Status**: Rate limiter implemented. Auth/API keys NOT implemented.
- **Use when**: Implementing authentication, adding security features, conducting security reviews
- **Example**: "Implement API key authentication" or "Add tenant isolation middleware"

**qa-engineer** - Testing, test coverage, test infrastructure
- **Status**: Minimal tests exist (~8-15% coverage).
- **Use when**: Writing tests, improving coverage, setting up test infrastructure
- **Example**: "Achieve 80% coverage for flows/" or "Add integration tests for database layer"

**code-reviewer** - Code review, quality checks, production readiness
- **Status**: Available for use.
- **Use when**: Before major releases, after completing waves, for quality assessments
- **Example**: "Review code before production deployment" or "Assess Wave 3 deliverables"

#### 📚 Documentation & Maintenance

**docs-engineer** - Documentation maintenance and synchronization
- **Status**: Core docs complete (ARCHITECTURE.md, DEVELOPER_GUIDE.md, QUICKSTART.md).
- **Use when**: After implementing features, infrastructure changes, or completing waves
- **Example**: "Update docs after implementing GitHub agent" or "Document new authentication system"
- **Remember**: Update existing docs rather than creating new ones

**code-cleaner** - Cleanup unused code, consolidate scripts
- **Status**: Repository cleaned after Wave 1.
- **Use when**: At end of each wave, before major releases, or when repo becomes cluttered
- **Example**: "Clean up after Wave 3" or "Remove obsolete migration scripts"

---

## Documentation Guidelines

### 🎯 Documentation Philosophy: Rationalized & Consolidated

**ALWAYS follow these principles:**

#### 1. Update Existing, Don't Create New
- ✅ **DO**: Update `docs/ARCHITECTURE.md` when adding new components
- ✅ **DO**: Update `docs/DEVELOPER_GUIDE.md` when changing development workflow
- ❌ **DON'T**: Create `ARCHITECTURE_v2.md` or separate doc per feature
- ❌ **DON'T**: Create extensive review documents that duplicate information

#### 2. Keep Documentation DRY (Don't Repeat Yourself)
- Each concept should be documented **once**, in the logical place
- Link between documents rather than duplicating content
- Example: Authentication details → `docs/DEVELOPER_GUIDE.md` (not scattered across multiple files)

#### 3. Rationalized Documentation Structure
\`\`\`
docs/
├── QUICKSTART.md          # 5-minute setup guide (minimal)
├── DEVELOPER_GUIDE.md     # Complete development guide (main reference)
├── ARCHITECTURE.md        # Technical architecture and design decisions
└── wave1/                 # Wave-specific artifacts (historical reference)
    └── README.md          # Consolidated Wave 1 summary
\`\`\`

**Do NOT create:**
- `FEATURE_X_GUIDE.md` - Add to DEVELOPER_GUIDE.md instead
- `API_DOCUMENTATION.md` - Use OpenAPI/Swagger at `/docs`
- Multiple architecture docs - One ARCHITECTURE.md is enough
- Per-feature documentation files - Consolidate into existing docs

#### 4. When Documentation IS Needed
Create new docs only when:
- Starting a new Wave (e.g., `docs/wave3/` folder)
- Adding completely new system (e.g., if adding authentication, update DEVELOPER_GUIDE.md with auth section)
- Historical reference (e.g., migration guides, postmortem documents)

#### 5. Documentation Maintenance Triggers
Update docs immediately after:
- ✅ Completing a development wave
- ✅ Adding new agent flows or tools
- ✅ Infrastructure changes (new services, deployment regions)
- ✅ Database schema changes
- ✅ Breaking changes to APIs or interfaces

---

## Repository Good Practices

See `docs/DEVELOPER_GUIDE.md` for complete coding standards and patterns.

### Key Principles:
- **Type hints**: All functions must have type hints
- **Error handling**: Use specific exceptions, structured logging
- **Database**: Always use `ConversationRepository`, never direct database access
- **LLM**: Always use `get_llm_adapter()` factory, never import providers directly
- **Testing**: Aim for 80%+ coverage
- **Documentation**: Update existing docs, don't create new ones

---

## Quick Commands Reference

\`\`\`bash
# Development
./dev db-proxy                    # Start PostgreSQL
./dev server  # Start API

# Testing
pytest -v                                  # Run all tests
pytest --cov=app --cov-report=html        # With coverage

# Code Quality
ruff format app tests                      # Format code
ruff check app tests --fix                 # Lint and auto-fix

# Database
alembic revision --autogenerate -m "msg"   # Create migration
alembic upgrade head                       # Apply migrations
alembic current                            # Check current version

# Deployment
git push origin main                       # Auto-deploys to Cloud Run
\`\`\`

---

## Key Documentation Files

- **`docs/QUICKSTART.md`** - 5-minute setup guide
- **`docs/DEVELOPER_GUIDE.md`** - Complete development guide (main reference)
- **`docs/ARCHITECTURE.md`** - Technical architecture and design decisions

## Reference Implementations

- **`app/flows/agents/weather_agent.py`** - ⭐ Reference agent flow (study this!)
- **`app/tools/weather/`** - ⭐ Reference tool implementation (follow this pattern!)

---

## Production URLs

- **API**: https://ai-agency-4ebxrg4hdq-ew.a.run.app
- **Docs**: https://ai-agency-4ebxrg4hdq-ew.a.run.app/docs
- **Region**: europe-west1 (Belgium)

---

## Summary

This is a **multi-tenant conversational AI platform** with:
- ✅ Multi-flow conversation architecture (single schema for all agents)
- ✅ Provider-agnostic LLM integration (OpenAI, Vertex AI, Mistral)
- ✅ Weather agent as reference implementation
- ✅ Cloud-native deployment on GCP Cloud Run
- ⚠️ Testing, RAG, and authentication to be implemented

**When in doubt:**
1. Look at Weather agent implementation (`app/flows/agents/weather_agent.py`)
2. Read `docs/DEVELOPER_GUIDE.md`
3. Follow the patterns established in Wave 1 & 2
4. Update existing docs, don't create new ones
5. Use the specialized agents in `.claude/agents/` for guidance
