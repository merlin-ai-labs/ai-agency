# DevOps Quick Start Guide

**Fast track to get the AI Agency Platform running locally and deployed to the cloud.**

---

## Prerequisites Checklist

- [ ] Python 3.11+ installed
- [ ] Docker and Docker Compose installed
- [ ] Git installed
- [ ] Code editor (VS Code, PyCharm, etc.)
- [ ] OpenAI API key (for LLM features)

---

## Local Development (5 Minutes)

### Option 1: Automated Setup (Recommended)

```bash
# 1. Clone and navigate to repository
cd /Users/mathiascara/ConsultingAgency

# 2. Run automated setup
./scripts/setup_dev.sh

# 3. Configure environment
cp .env.local .env
# Edit .env and add your OPENAI_API_KEY

# 4. Start all services
docker-compose up

# 5. Open browser
# http://localhost:8080/docs
```

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Start PostgreSQL
docker-compose up -d db

# 4. Run migrations
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_agency
alembic upgrade head

# 5. Start application
uvicorn app.main:app --reload

# 6. Open browser
# http://localhost:8080/docs
```

---

## Verify Installation

```bash
# Run validation script
./scripts/validate_setup.sh

# Expected output: All checks passed ✓
```

---

## Common Development Tasks

### Database Operations

```bash
# Create new migration
./scripts/run_migrations.sh revision "Add user table"

# Apply migrations
./scripts/run_migrations.sh upgrade head

# Rollback one migration
./scripts/run_migrations.sh downgrade -1

# Show current version
./scripts/run_migrations.sh current

# Show history
./scripts/run_migrations.sh history
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test
pytest tests/test_flows.py::test_maturity_assessment

# Run integration tests only
pytest -m integration
```

### Code Quality

```bash
# Format code
ruff format app tests

# Lint code
ruff check app tests

# Type check
mypy app

# Run all pre-commit hooks
pre-commit run --all-files
```

### Docker Operations

```bash
# Start all services
docker-compose up

# Start in background
docker-compose up -d

# Stop all services
docker-compose down

# Stop and remove volumes (fresh start)
docker-compose down -v

# View logs
docker-compose logs -f app

# Restart application only
docker-compose restart app
```

---

## Cloud Deployment

### Choose Your Provider

- [GCP Cloud Run](#gcp-deployment) (Recommended)
- [AWS App Runner/ECS](#aws-deployment)
- [Azure Container Instances](#azure-deployment)

### GCP Deployment

```bash
# 1. Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# 2. Set up Cloud SQL
gcloud sql instances create ai-agency-db \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=us-central1 \
  --database-flags=cloudsql.enable_pgvector=on

# 3. Create database
gcloud sql databases create ai_agency --instance=ai-agency-db

# 4. Build and push Docker image
gcloud builds submit --tag gcr.io/PROJECT_ID/ai-agency

# 5. Deploy to Cloud Run
gcloud run deploy ai-agency \
  --image gcr.io/PROJECT_ID/ai-agency \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

# See docs/DEPLOYMENT.md for complete instructions
```

### AWS Deployment

```bash
# 1. Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier ai-agency-db \
  --engine postgres \
  --engine-version 15.4

# 2. Install pgvector extension
psql $DATABASE_URL -c "CREATE EXTENSION vector;"

# 3. Push to ECR and deploy to App Runner
# See docs/DEPLOYMENT.md for complete instructions
```

### Azure Deployment

```bash
# 1. Create PostgreSQL Flexible Server
az postgres flexible-server create \
  --name ai-agency-db \
  --version 15

# 2. Enable pgvector
az postgres flexible-server parameter set \
  --name azure.extensions \
  --value vector

# 3. Deploy to Container Instances
# See docs/DEPLOYMENT.md for complete instructions
```

---

## Database Portability

**Switching providers is easy - just change the DATABASE_URL!**

### Connection Strings

```bash
# Local
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_agency

# GCP Cloud SQL
DATABASE_URL=postgresql://user:pass@/cloudsql/project:region:instance/db

# AWS RDS
DATABASE_URL=postgresql://user:pass@host.rds.amazonaws.com:5432/db?sslmode=require

# Azure
DATABASE_URL=postgresql://user:pass@host.postgres.database.azure.com:5432/db?sslmode=require
```

### Test Connection

```bash
# Export new DATABASE_URL
export DATABASE_URL="your-connection-string"

# Test connection
psql "$DATABASE_URL" -c "SELECT version();"

# Run migrations
alembic upgrade head

# Verify pgvector
psql "$DATABASE_URL" -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

---

## CI/CD Pipeline

### Enable GitHub Actions

1. **Configure Secrets** (Settings > Secrets and variables > Actions)
   ```
   OPENAI_API_KEY          # Your OpenAI API key
   GCP_PROJECT_ID          # Your GCP project ID (if using GCP)
   GCP_SA_KEY              # Service account JSON (if using GCP)
   DATABASE_URL            # Production database URL
   CODECOV_TOKEN           # Codecov token (optional)
   ```

2. **CI Pipeline** (Runs automatically on PRs)
   - `.github/workflows/ci.yml` is already active
   - Tests, linting, type checking, coverage

3. **Deployment Pipeline** (Enable manually)
   - Edit `.github/workflows/deploy.yml`
   - Change `if: false` to `if: true`
   - Push to main branch to trigger deployment

---

## Troubleshooting

### Database Connection Issues

```bash
# Check PostgreSQL is running
docker-compose ps

# Check database logs
docker-compose logs db

# Restart database
docker-compose restart db

# Test connection manually
psql postgresql://postgres:postgres@localhost:5432/ai_agency
```

### Migration Issues

```bash
# Check current migration version
./scripts/run_migrations.sh current

# Show migration history
./scripts/run_migrations.sh history

# Force stamp database (if out of sync)
./scripts/run_migrations.sh stamp head

# Rollback and reapply
./scripts/run_migrations.sh downgrade -1
./scripts/run_migrations.sh upgrade head
```

### Docker Issues

```bash
# Clean Docker cache
docker system prune -a

# Rebuild containers
docker-compose build --no-cache

# Reset everything (CAUTION: deletes data)
docker-compose down -v
./scripts/setup_dev.sh
```

### Pre-commit Issues

```bash
# Reinstall hooks
pre-commit clean
pre-commit install

# Update hook versions
pre-commit autoupdate

# Run manually to debug
pre-commit run --all-files --verbose
```

---

## Environment Variables Reference

### Required
- `DATABASE_URL`: PostgreSQL connection string
- `OPENAI_API_KEY`: OpenAI API key (or use Vertex AI)

### Optional
- `GCP_PROJECT_ID`: GCP project ID (for GCS, Vertex AI)
- `GCS_BUCKET`: GCS bucket name for artifacts
- `LLM_PROVIDER`: `openai` or `vertex` (default: `openai`)
- `LOG_LEVEL`: `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`)
- `ENVIRONMENT`: `development`, `staging`, `production`

### Example .env Files
- `.env.local` - Local development
- `.env.gcp` - GCP Cloud Run
- `.env.aws` - AWS deployment
- `.env.azure` - Azure deployment

---

## Performance Tips

### Development
- Use `--reload` flag for hot reload during development
- Mount code as volume in docker-compose for instant changes
- Use `pytest -n auto` for parallel test execution

### Production
- Enable connection pooling in SQLAlchemy
- Use read replicas for read-heavy workloads
- Configure appropriate worker count: `workers = (2 * CPU) + 1`
- Set reasonable timeouts: `--timeout 300`
- Use health checks: `/health` endpoint

### Database
- Run `ANALYZE` after bulk data imports
- Tune IVFFlat index: `lists = sqrt(rows)` for optimal performance
- Monitor slow queries: Enable `log_min_duration_statement`
- Use `EXPLAIN ANALYZE` to optimize queries

---

## Additional Resources

### Documentation
- [Full Deployment Guide](DEPLOYMENT.md)
- [Implementation Summary](DEVOPS_IMPLEMENTATION_SUMMARY.md)
- [Architecture Documentation](REPO-ARCHITECTURE-LEAN-EN.md)

### External Resources
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)

### Support
- Check troubleshooting section in [DEPLOYMENT.md](DEPLOYMENT.md)
- Review logs: `docker-compose logs -f`
- Enable debug mode: `LOG_LEVEL=DEBUG`

---

## Quick Command Reference

```bash
# Setup
./scripts/setup_dev.sh                    # Automated setup
./scripts/validate_setup.sh               # Verify installation

# Development
docker-compose up                         # Start services
uvicorn app.main:app --reload            # Start with hot reload
pytest --cov=app                         # Run tests with coverage

# Database
./scripts/run_migrations.sh upgrade head # Apply migrations
./scripts/run_migrations.sh revision "msg" # Create migration
psql $DATABASE_URL                       # Connect to database

# Code Quality
ruff format app tests                    # Format code
ruff check app tests                     # Lint code
mypy app                                 # Type check
pre-commit run --all-files              # Run all hooks

# Docker
docker-compose logs -f app               # Follow logs
docker-compose restart app               # Restart app
docker-compose down -v                   # Clean reset
```

---

**Ready to start?** Run `./scripts/setup_dev.sh` and you'll be up and running in minutes!

**Last Updated**: 2025-10-27
