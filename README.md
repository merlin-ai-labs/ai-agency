# AI Agency — Lean Monorepo for AI Agents/Flows

Minimalist Python platform for AI agent workflows on GCP. No frontend, no over-engineering.

## Quick Start (Local)
```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # Edit with your values
python scripts/seed.py
uvicorn app.main:app --reload
```

## Quick Start (GCP Cloud Run)
```bash
gcloud run deploy ai-agency --source . --region us-central1 \
  --set-env-vars DATABASE_URL=...,GCS_BUCKET=...,LLM_PROVIDER=vertex
```

## Architecture
See [docs/REPO-ARCHITECTURE-LEAN-EN.md](docs/REPO-ARCHITECTURE-LEAN-EN.md) for full details.

## Flows
- **Maturity Assessment**: Parse docs → Score → Recommendations → `assessment.json`
- **Use-Case Grooming**: Consume assessment → Prioritize (RICE/WSJF) → Backlog

## Testing
```bash
pytest
```

**Status**: 🏗️ Skeleton only — no business logic implemented yet.
