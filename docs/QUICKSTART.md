# Quick Start Guide

Complete setup guide to get the AI Consulting Agency Platform running locally in 10 minutes.

## Prerequisites

- **Python 3.11+** - Check with `python3 --version`
- **Git** - Check with `git --version`
- **GCP Access** - You need access to the `merlin-notebook-lm` project
- **gcloud CLI** - [Install guide](https://cloud.google.com/sdk/docs/install)

## Step 1: GCP Authentication

First, authenticate with Google Cloud to access Cloud SQL and secrets:

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

## Step 2: Install Cloud SQL Proxy

The Cloud SQL Proxy allows you to connect to the production database securely from your local machine.

```bash
# macOS
brew install cloud-sql-proxy

# Verify installation
cloud-sql-proxy --version
```

**Other platforms:** See [official install guide](https://cloud.google.com/sql/docs/postgres/sql-proxy#install)

## Step 3: Get Database Password

Retrieve the Cloud SQL database password from Google Secret Manager:

```bash
# Get the password (save this, you'll need it for .env)
gcloud secrets versions access latest --secret='cloud-sql-password'
```

**Save this password** - you'll use it in Step 6.

## Step 4: Start Cloud SQL Proxy

Open a **separate terminal window** and start the proxy (keep it running):

```bash
cloud-sql-proxy merlin-notebook-lm:europe-west1:ai-agency-db --port 5433
```

You should see:
```
Ready for new connections
```

**Keep this terminal open!** The proxy needs to run continuously while you develop.

## Step 5: Clone and Setup Project

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
- ✅ Run database migrations
- ✅ Create `.env` file from `.env.example`

## Step 6: Configure Environment Variables

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

## Step 7: Verify Setup

Check that everything is configured correctly:

```bash
# Verify Cloud SQL Proxy is connected
./dev db-check
```

You should see:
```
✓ Cloud SQL Proxy running on port 5433
```

If you see an error, go back to Step 4 and ensure the proxy is running.

## Step 8: Start Development Server

```bash
./dev server
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

## Step 9: Test It Works

Open your browser or use curl to test the API:

**1. Health Check:**
```bash
curl http://localhost:8000/health
# Expected: {"status":"healthy"}
```

**2. Interactive API Docs:**

Visit: http://localhost:8000/docs

You should see the FastAPI interactive documentation (Swagger UI).

**3. Try the Weather Agent:**
```bash
curl -X POST http://localhost:8000/weather/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the weather in London?",
    "tenant_id": "test-user"
  }'
```

Expected response:
```json
{
  "response": "The current weather in London is...",
  "conversation_id": "uuid-here"
}
```

## Common Commands

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

## Daily Development Workflow

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

## Next Steps

Now that you're set up, here's what to read next:

1. **[CODING_STANDARDS.md](CODING_STANDARDS.md)** - **READ THIS FIRST!** Essential patterns and best practices
2. **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** - How to build agents, flows, and tools
3. **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design and technical decisions

## Quick Reference

**GCP Commands:**
```bash
gcloud auth login                                              # Login to GCP
gcloud config set project merlin-notebook-lm                   # Set project
gcloud secrets versions access latest --secret='cloud-sql-password'  # Get DB password
```

**Cloud SQL Proxy:**
```bash
cloud-sql-proxy merlin-notebook-lm:europe-west1:ai-agency-db --port 5433  # Start proxy
```

**Development:**
```bash
./dev setup      # One-time setup
./dev server     # Start API server
./dev test       # Run tests
./dev quality    # Code quality checks
```

**API Endpoints:**
- Health: http://localhost:8000/health
- Docs: http://localhost:8000/docs
- Weather: `POST /weather/chat`

---

**Need help?** Open an issue or check existing issues on GitHub.
