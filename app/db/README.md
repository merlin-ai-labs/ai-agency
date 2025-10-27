# Database Module

This module contains database models and migration management.

## Setup

### Local Development (SQLite)
```bash
export DATABASE_URL=sqlite:///./dev.db
python scripts/seed.py
```

### Production (Postgres + pgvector)
```bash
# Enable pgvector extension
psql $DATABASE_URL -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Run migrations (TODO: Set up Alembic)
# alembic upgrade head
```

## Models

- **Run**: Flow execution records (queue table)
- **DocumentChunk**: Text chunks with embeddings for RAG
- **Tenant**: Multi-tenant isolation

## TODO

- [ ] Set up Alembic for migrations
- [ ] Add pgvector support for embeddings
- [ ] Add indexes for performance
- [ ] Add connection pooling
- [ ] Add database session management (FastAPI dependency injection)
