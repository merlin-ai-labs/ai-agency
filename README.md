# AI Consulting Agency Platform

AI-powered consulting platform for maturity assessments and use case grooming.

**Status:** 🏗️ Wave 1 Complete - Foundation Ready

## Features

- 📊 AI Maturity Assessments with rubric-based scoring
- 🎯 Use Case Grooming and prioritization
- 🤖 Multi-LLM support (OpenAI, Vertex AI)
- 🔍 RAG-powered document analysis with pgvector
- 🔐 Secure multi-tenant architecture
- ⚡ Fast async API with FastAPI
- 📈 Database-backed queue (no Pub/Sub overhead)

## Tech Stack

- **Backend:** FastAPI + Python 3.11+
- **Database:** PostgreSQL 15 + pgvector
- **LLM Providers:** OpenAI, Google Vertex AI
- **Storage:** Google Cloud Storage
- **Testing:** pytest with 80%+ coverage target
- **Deployment:** Docker, Google Cloud Run (multi-cloud ready)

## Quick Start

### 1. Local Development

```bash
# Clone and setup
git clone <repo>
cd ConsultingAgency
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Start PostgreSQL with pgvector
docker-compose up -d db

# Run migrations
DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/ai_agency" alembic upgrade head

# Start the API
uvicorn app.main:app --reload
```

Access the API at: http://localhost:8080/docs

### 2. Run Tests

```bash
# Activate venv if not already
source venv/bin/activate

# Run all tests
pytest -v

# Run with coverage
pytest --cov=app --cov-report=html
open htmlcov/index.html
```

### 3. Code Quality Checks

```bash
# Linting and formatting
ruff check app tests
ruff format app tests

# Type checking
mypy app
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for comprehensive technical architecture with diagrams:
- System overview and component architecture
- Data flow diagrams
- Database schema (ER diagram)
- Deployment architecture (local, GCP, multi-cloud)
- Design patterns and security

## API Endpoints

### Assessments
- `POST /api/v1/assessments` - Create maturity assessment
- `GET /api/v1/assessments/{run_id}` - Get assessment results

### Use Cases
- `POST /api/v1/use-cases/groom` - Groom and prioritize use cases
- `GET /api/v1/use-cases/{run_id}` - Get grooming results

### Health Check
- `GET /health` - System health status

Full API documentation: http://localhost:8080/docs

## Project Structure

```
ConsultingAgency/
├── app/
│   ├── api/                 # FastAPI routes and endpoints
│   ├── core/                # ✅ Base classes, exceptions, decorators
│   ├── db/                  # ✅ Database models, repositories, migrations
│   ├── flows/               # Flow orchestration (maturity, grooming)
│   ├── llm/                 # LLM adapters (OpenAI, Vertex AI)
│   ├── rag/                 # RAG engine with pgvector
│   ├── tools/               # Business logic tools (5 core tools)
│   ├── security/            # Authentication and authorization
│   └── main.py              # FastAPI application entry point
├── tests/                   # Test suite (pytest)
├── docs/                    # ✅ Documentation
│   ├── ARCHITECTURE.md      # Technical architecture (diagrams)
│   ├── CODING_STANDARDS.md  # Python coding standards
│   ├── DEPLOYMENT.md        # Multi-cloud deployment guide
│   ├── CODE_REVIEW_CHECKLIST.md
│   └── WAVE1_REVIEW.md      # Wave 1 implementation review
├── scripts/                 # ✅ Setup and utility scripts
├── .github/workflows/       # ✅ CI/CD pipeline
├── docker-compose.yml       # ✅ Local development environment
├── alembic.ini              # ✅ Database migration config
└── pyproject.toml           # ✅ Dependencies and tool config
```

**Legend:** ✅ = Implemented | 🚧 = In Progress | ⏳ = Pending

## Development Workflow

### Wave-Based Development

The project is being built in 6 coordinated waves using specialized AI agents:

- **Wave 1** ✅ - Foundation (tech-lead, devops-engineer)
- **Wave 2** 🚧 - Core Services (database-engineer, llm-engineer)
- **Wave 3** ⏳ - RAG Implementation (rag-engineer)
- **Wave 4** ⏳ - Business Logic (tools-engineer, flows-engineer)
- **Wave 5** ⏳ - Quality & Security (qa-engineer, security-engineer, docs-engineer)
- **Wave 6** ⏳ - Final Review (code-reviewer)

See [docs/WAVE1_REVIEW.md](docs/WAVE1_REVIEW.md) for Wave 1 completion summary.

## Database Portability

The platform supports multiple database providers via simple environment variable configuration:

```bash
# GCP Cloud SQL
DATABASE_URL="postgresql+psycopg://user:pass@/dbname?host=/cloudsql/project:region:instance"

# AWS RDS
DATABASE_URL="postgresql+psycopg://user:pass@host.rds.amazonaws.com:5432/dbname?sslmode=require"

# Azure Database
DATABASE_URL="postgresql+psycopg://user@server:pass@host.postgres.database.azure.com:5432/dbname?sslmode=require"

# Local Docker
DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/ai_agency"
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed multi-cloud setup.

## Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Complete technical architecture with diagrams
- **[CODING_STANDARDS.md](docs/CODING_STANDARDS.md)** - Python coding standards
- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Multi-cloud deployment guide
- **[CODE_REVIEW_CHECKLIST.md](docs/CODE_REVIEW_CHECKLIST.md)** - Code review guidelines
- **[WAVE1_REVIEW.md](docs/WAVE1_REVIEW.md)** - Wave 1 implementation review

## Environment Variables

Required environment variables:

```bash
# Database
DATABASE_URL=postgresql+psycopg://user:pass@host:5432/dbname

# LLM Provider
LLM_PROVIDER=openai  # or vertex
OPENAI_API_KEY=sk-...  # if using OpenAI
GCP_PROJECT_ID=...     # if using Vertex AI

# Storage
GCS_BUCKET=your-bucket-name

# Optional
LOG_LEVEL=INFO
ENVIRONMENT=development
```

See `.env.example` for complete list.

## CI/CD Pipeline

GitHub Actions runs on every PR and push to main/develop:

- ✅ Ruff linting and formatting
- ✅ mypy type checking (continue-on-error initially)
- ✅ pytest with 70% coverage requirement
- ✅ PostgreSQL + pgvector integration tests
- ✅ Alembic migration tests
- ✅ Codecov integration

See [.github/workflows/ci.yml](.github/workflows/ci.yml)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow coding standards in [docs/CODING_STANDARDS.md](docs/CODING_STANDARDS.md)
4. Write tests (maintain 80%+ coverage)
5. Run pre-commit hooks: `ruff check && ruff format && mypy app`
6. Submit a pull request

## License

[Your License]

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- See documentation in [docs/](docs/)
- Check API docs at http://localhost:8080/docs
