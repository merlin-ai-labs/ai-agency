# AI Consulting Agency Platform

AI-powered consulting platform for maturity assessments and use case grooming with multi-LLM support and RAG capabilities.

## Quick Start

```bash
# 1. Clone repository
git clone <repo-url>
cd ConsultingAgency

# 2. Start Cloud SQL Proxy (in a separate terminal)
cloud-sql-proxy merlin-notebook-lm:europe-west1:ai-agency-db --port 5433

# 3. One-command setup
./dev setup

# 4. Update .env with your Cloud SQL password and API keys

# 5. Start server
./dev server
```

Visit http://localhost:8000/docs for the interactive API documentation.

See all available commands: `./dev help`

**For detailed setup and development workflow, see [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)**

## Features

- 📊 AI Maturity Assessments with rubric-based scoring
- 🎯 Use Case Grooming and prioritization
- 🤖 Multi-LLM support (OpenAI, Vertex AI)
- 🔍 RAG-powered document analysis with pgvector
- 🔐 Secure multi-tenant architecture
- ⚡ Fast async API with FastAPI

## Tech Stack

- **Backend**: FastAPI + Python 3.11+
- **Database**: PostgreSQL 15 + pgvector (Cloud SQL)
- **LLM Providers**: OpenAI, Google Vertex AI
- **Storage**: Google Cloud Storage
- **Testing**: pytest (current: 8% coverage, target: 80%)
- **Deployment**: Google Cloud Run (containerized)
- **CI/CD**: GitHub Actions (auto-deploy on push to main)

## Production Deployment

**Live Service**: https://ai-agency-4ebxrg4hdq-ew.a.run.app

- **Region**: europe-west1 (Belgium)
- **Database**: Cloud SQL PostgreSQL 15
- **Auto-deployment**: Enabled via GitHub Actions

## Project Structure

```
ConsultingAgency/
├── app/
│   ├── api/              # API endpoints
│   ├── core/             # Base classes, exceptions, decorators
│   ├── db/               # Database models and sessions
│   ├── flows/            # Business workflows (maturity, grooming)
│   ├── adapters/         # LLM provider adapters
│   ├── rag/              # RAG ingestion and retrieval
│   ├── tools/            # Business logic tools
│   └── main.py           # FastAPI application
├── tests/                # Test suite
├── scripts/              # Setup and utility scripts
├── docs/                 # Documentation
├── alembic/              # Database migrations
└── dev                   # Development CLI tool
```

## Development

### Running Tests

```bash
pytest -v --cov=app
```

### Code Quality

```bash
ruff check app tests --fix
ruff format app tests
mypy app
```

### Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head
```

## Documentation

### Getting Started
- **[QUICKSTART.md](docs/QUICKSTART.md)** - 5-minute setup guide

### Development
- **[DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)** - Complete development guide
- **[CODING_STANDARDS.md](docs/CODING_STANDARDS.md)** - Best practices, decorators, patterns (READ THIS FIRST!)
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Technical architecture and design decisions

## API Endpoints

### Core Endpoints
- `POST /runs` - Create flow execution
- `GET /runs/{run_id}` - Get run status
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

### Use Cases
- Maturity assessment flow: `flow_name: "maturity_assessment"`
- Use case grooming flow: `flow_name: "usecase_grooming"`

Example:
```bash
curl -X POST http://localhost:8080/runs \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "my-company",
    "flow_name": "maturity_assessment",
    "input_data": {}
  }'
```

## Environment Configuration

Required environment variables:

```bash
# Database (requires Cloud SQL Proxy on port 5433)
DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5433/ai_agency

# LLM Provider
LLM_PROVIDER=openai  # or "vertex"
OPENAI_API_KEY=sk-...

# Storage (for production)
GCS_BUCKET=your-bucket-name

# Optional
LOG_LEVEL=INFO
ENVIRONMENT=development
```

See `.env.example` for complete configuration.

## CI/CD Pipeline

### Testing (`ci.yml`)
Runs on every PR and push:
- Linting and formatting (Ruff)
- Type checking (mypy)
- Tests with coverage (pytest)
- PostgreSQL + pgvector integration tests

### Deployment (`deploy.yml`)
Auto-deploys on push to main:
- Build Docker image
- Push to Artifact Registry
- Deploy to Cloud Run
- Run smoke tests

## Contributing

1. Create a feature branch
2. Read [CODING_STANDARDS.md](docs/CODING_STANDARDS.md) for patterns and best practices
3. Write tests (maintain 80%+ coverage target)
4. Run quality checks: `ruff check && ruff format && mypy app && pytest`
5. Submit a pull request

## Development Phases

- **Wave 1** ✅ - Foundation (completed, deployed)
- **Wave 2** 🚧 - Core Services & Multi-LLM (in progress)
- **Wave 3** ⏳ - RAG Implementation
- **Wave 4** ⏳ - Business Logic
- **Wave 5** ⏳ - Quality & Security
- **Wave 6** ⏳ - Final Review

## Support

- Start with [QUICKSTART.md](docs/QUICKSTART.md) for setup
- Read [CODING_STANDARDS.md](docs/CODING_STANDARDS.md) before coding
- Check [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for common tasks
- Review [ARCHITECTURE.md](docs/ARCHITECTURE.md) for technical decisions
- Open an issue for bugs or feature requests
