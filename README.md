# AI Consulting Agency Platform

AI-powered consulting platform for maturity assessments and use case grooming with multi-LLM support and RAG capabilities.

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url>
cd ConsultingAgency
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# 2. Configure environment
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# 3. Start database
docker-compose up -d db

# 4. Run migrations
./scripts/run_migrations.sh

# 5. Start API
uvicorn app.main:app --reload --port 8080
```

Visit http://localhost:8080/docs for the interactive API documentation.

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
- **Database**: PostgreSQL 15 + pgvector
- **LLM Providers**: OpenAI, Google Vertex AI
- **Storage**: Google Cloud Storage
- **Testing**: pytest (current: 8% coverage, target: 80%)
- **Deployment**: Docker + Google Cloud Run
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
└── docker-compose.yml    # Local development services
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

- **[DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)** - Complete development guide
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Technical architecture
- **[CODING_STANDARDS.md](docs/CODING_STANDARDS.md)** - Code style guidelines
- **[TESTING_GUIDE.md](docs/TESTING_GUIDE.md)** - Testing reference
- **[CHANGELOG.md](docs/CHANGELOG.md)** - Version history

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
# Database
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/ai_agency

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
2. Follow [coding standards](docs/CODING_STANDARDS.md)
3. Write tests (maintain 80%+ coverage target)
4. Run quality checks: `ruff check && ruff format && mypy app && pytest`
5. Submit a pull request

## Development Phases

- **Wave 1** ✅ - Foundation (completed, deployed)
- **Wave 2** 🚧 - Core Services (in progress)
- **Wave 3** ⏳ - RAG Implementation
- **Wave 4** ⏳ - Business Logic
- **Wave 5** ⏳ - Quality & Security
- **Wave 6** ⏳ - Final Review

## Support

- Check [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) for common tasks
- Review existing code for examples
- Open an issue for bugs or feature requests
