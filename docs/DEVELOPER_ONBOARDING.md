# Developer Onboarding Guide

Welcome to the AI Agency Platform development team! This guide will get you up and running in **under 10 minutes**.

---

## 📋 Prerequisites

Before you start, make sure you have:

1. **Python 3.11+** installed
2. **Docker** installed (Docker Desktop or Colima)
3. **Git** installed
4. **gcloud CLI** installed ([Install here](https://cloud.google.com/sdk/docs/install))
5. **Access to merlin-notebook-lm GCP project** (ask admin if you don't have it)

---

## 🚀 Quick Start (5 Minutes)

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/merlin-ai-labs/ai-agency.git
cd ai-agency

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
# Get your key from: https://platform.openai.com/api-keys
nano .env  # or use your preferred editor
```

**Required in `.env`:**
```bash
OPENAI_API_KEY=sk-your-actual-key-here
```

### 3. Start Local Database

```bash
# Start PostgreSQL with pgvector
docker-compose up -d db

# Wait for it to be healthy (10 seconds)
sleep 10

# Run migrations
DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/ai_agency" \
  alembic upgrade head
```

### 4. Verify Setup

```bash
# Run validation script
./scripts/validate_wave1.sh
```

**Expected output:** All 10 checks should pass ✅

### 5. Start Development Server

```bash
# Activate venv if not already
source venv/bin/activate

# Start the API
uvicorn app.main:app --reload
```

**Access the API:**
- Swagger UI: http://localhost:8080/docs
- Health check: http://localhost:8080/health

---

## 🔐 GCP Authentication (One-Time Setup)

If you'll be deploying to GCP or using Vertex AI:

```bash
# 1. Authenticate with your Google account
gcloud auth login

# 2. Set the project
gcloud config set project merlin-notebook-lm

# 3. Configure Docker for Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

# 4. Verify access
gcloud projects describe merlin-notebook-lm
```

**If you get permission errors:**
- Ask the admin to grant you these roles:
  - `roles/artifactregistry.writer`
  - `roles/storage.objectAdmin`
  - `roles/aiplatform.user`
  - `roles/run.developer`

---

## 🏗️ Project Structure

```
ai-agency/
├── app/
│   ├── api/              # FastAPI routes
│   ├── core/             # ✅ Base classes, exceptions, decorators
│   ├── db/               # ✅ Models, repositories, migrations
│   ├── flows/            # Flow orchestration
│   ├── llm/              # LLM adapters (OpenAI, Vertex AI)
│   ├── rag/              # RAG engine with pgvector
│   ├── tools/            # Business logic tools
│   └── main.py           # FastAPI app
├── tests/                # Test suite
├── docs/                 # Documentation
│   ├── ARCHITECTURE.md   # System architecture
│   ├── CODING_STANDARDS.md
│   └── ...
├── scripts/              # Utility scripts
├── .env.example          # Environment template
└── docker-compose.yml    # Local development
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=app --cov-report=html
open htmlcov/index.html

# Run specific test file
pytest tests/test_main.py -v

# Run tests matching pattern
pytest -k "test_health" -v
```

---

## 🎨 Code Quality

### Before Committing

```bash
# Format code
ruff format app tests

# Check linting
ruff check app tests

# Fix auto-fixable issues
ruff check --fix app tests

# Type checking
mypy app
```

### Pre-commit Hooks (Recommended)

```bash
# Install pre-commit hooks
pre-commit install

# Now these checks run automatically on git commit
```

---

## 🐳 Docker Development

### Build and Run

```bash
# Build the container
docker build -t ai-agency:dev .

# Run locally
docker run -p 8080:8080 \
  -e DATABASE_URL="postgresql+psycopg://postgres:postgres@host.docker.internal:5432/ai_agency" \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  ai-agency:dev
```

### Docker Compose (Full Stack)

```bash
# Start everything
docker-compose up -d

# Check logs
docker-compose logs -f app

# Stop everything
docker-compose down
```

---

## 🚀 Deploying to GCP

### First Time Setup (Admin Only)

```bash
# Run the GCP setup script (admin only)
./scripts/setup_gcp.sh
```

### Build and Deploy

```bash
# 1. Build and push container
docker build -t us-central1-docker.pkg.dev/merlin-notebook-lm/ai-agency/app:latest .
docker push us-central1-docker.pkg.dev/merlin-notebook-lm/ai-agency/app:latest

# 2. Deploy to Cloud Run
gcloud run services replace clouddeploy.yaml --region=us-central1

# 3. Get the URL
gcloud run services describe ai-agency \
  --region=us-central1 \
  --format='value(status.url)'
```

### Quick Redeploy

```bash
# Build, push, and deploy in one command
./scripts/deploy.sh
```

---

## 🔧 Common Tasks

### Update Dependencies

```bash
# Add a new dependency
pip install <package>

# Update pyproject.toml
# Then reinstall
pip install -e ".[dev]"
```

### Create a New Migration

```bash
# After modifying models in app/db/models.py
alembic revision --autogenerate -m "Description of changes"

# Review the generated migration
# Then apply it
alembic upgrade head
```

### Reset Database

```bash
# Warning: This deletes all data!
docker-compose down -v
docker-compose up -d db
sleep 10
DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/ai_agency" \
  alembic upgrade head
```

---

## 🎯 Development Workflow

### Feature Development

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Make changes
# ... code ...

# 3. Run tests
pytest -v

# 4. Run linting
ruff check app tests
ruff format app tests

# 5. Commit
git add .
git commit -m "feat: add my feature"

# 6. Push and create PR
git push origin feature/my-feature
```

### Pull Request Checklist

- [ ] All tests pass locally
- [ ] Code is formatted (ruff format)
- [ ] No linting errors (ruff check)
- [ ] Type hints added where applicable
- [ ] Documentation updated if needed
- [ ] PR description explains the changes

---

## 🆘 Troubleshooting

### Docker Issues

**Problem:** `docker-compose` not found
```bash
# Install Docker Compose
pip install docker-compose
```

**Problem:** PostgreSQL container not healthy
```bash
# Check logs
docker-compose logs db

# Restart
docker-compose restart db
```

### Migration Issues

**Problem:** Migration fails
```bash
# Check current revision
alembic current

# Downgrade one revision
alembic downgrade -1

# Or reset to base
alembic downgrade base

# Then upgrade again
alembic upgrade head
```

### Import Errors

**Problem:** Can't import app modules
```bash
# Reinstall in editable mode
pip install -e .
```

### GCP Permission Issues

**Problem:** Can't push to Artifact Registry
```bash
# Re-authenticate Docker
gcloud auth configure-docker us-central1-docker.pkg.dev

# Check your permissions
gcloud projects get-iam-policy merlin-notebook-lm \
  --flatten="bindings[].members" \
  --filter="bindings.members:user:$(gcloud config get-value account)"
```

---

## 📚 Additional Resources

### Documentation
- [Architecture Diagrams](ARCHITECTURE.md) - System design
- [Coding Standards](CODING_STANDARDS.md) - Python best practices
- [Deployment Guide](DEPLOYMENT.md) - Multi-cloud deployment
- [API Documentation](http://localhost:8080/docs) - When running locally

### External Links
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLModel Documentation](https://sqlmodel.tiangolo.com/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)

---

## 🎓 Learning Path

### Week 1: Setup & Exploration
- [ ] Complete this onboarding guide
- [ ] Run the application locally
- [ ] Explore the codebase structure
- [ ] Read ARCHITECTURE.md
- [ ] Run tests and understand coverage

### Week 2: First Contribution
- [ ] Pick a "good first issue" from GitHub
- [ ] Create a feature branch
- [ ] Make your changes
- [ ] Submit your first PR

### Week 3: Deep Dive
- [ ] Understand the flow orchestration
- [ ] Learn about LLM adapters
- [ ] Explore the RAG implementation
- [ ] Review database schema

---

## 🤝 Getting Help

- **Slack:** #ai-agency-dev
- **GitHub Issues:** [Report bugs or request features](https://github.com/merlin-ai-labs/ai-agency/issues)
- **Team Lead:** Ask in team channel
- **Documentation:** Check docs/ folder first

---

## 🎉 Welcome to the Team!

You're now ready to start developing AI agents! If you have any questions or run into issues, don't hesitate to ask in the team channel.

**Happy coding!** 🚀
