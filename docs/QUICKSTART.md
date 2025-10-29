# Quick Start

Get the AI Consulting Agency Platform running locally in 5 minutes.

## Prerequisites

- Python 3.11+
- Cloud SQL Proxy ([installation guide](https://cloud.google.com/sql/docs/postgres/sql-proxy))
- Git
- GCP account with access to the project

## Setup Steps

```bash
# 1. Clone repository
git clone <repo-url>
cd ConsultingAgency

# 2. Install Cloud SQL Proxy (macOS)
brew install cloud-sql-proxy

# 3. Start Cloud SQL Proxy (in a separate terminal - keep it running)
cloud-sql-proxy merlin-notebook-lm:europe-west1:ai-agency-db --port 5433

# 4. Run setup (creates venv, installs deps, runs migrations)
./dev setup

# 5. Update .env with your credentials
# Edit .env and add:
#   - DATABASE_URL with Cloud SQL password (get from: gcloud secrets versions access latest --secret='cloud-sql-password')
#   - OPENAI_API_KEY

# 6. Start development server
./dev server
```

## Test It Works

Visit http://localhost:8000/docs - you should see the FastAPI interactive documentation.

Try the health endpoint:
```bash
curl http://localhost:8000/health
```

Try the weather agent:
```bash
curl -X POST http://localhost:8000/weather-chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the weather in London?",
    "tenant_id": "test-user"
  }'
```

## Common Commands

```bash
./dev help        # Show all available commands
./dev db-check    # Verify Cloud SQL Proxy is running
./dev db-migrate  # Run database migrations
./dev db-seed     # Seed database with test data
./dev test        # Run all tests
./dev quality     # Run linting, formatting, type checks
./dev clean       # Clean generated files
```

## Next Steps

For detailed development workflows, building flows, and deployment, see:
- [CODING_STANDARDS.md](CODING_STANDARDS.md) - **READ THIS FIRST!** Best practices and patterns
- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - Complete development guide
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and technical decisions

## Troubleshooting

**Database connection failed?**
```bash
# Check if Cloud SQL Proxy is running
./dev db-check

# If not running, start it in a separate terminal:
cloud-sql-proxy merlin-notebook-lm:europe-west1:ai-agency-db --port 5433
```

**Port 8000 already in use?**
```bash
lsof -i :8000  # Find process
kill -9 <PID>  # Kill it
```

**Port 5433 already in use?**
```bash
lsof -i :5433  # Find process using port 5433
kill -9 <PID>  # Kill it, then restart Cloud SQL Proxy
```

**Import errors?**
```bash
./dev setup  # Re-run setup
```
