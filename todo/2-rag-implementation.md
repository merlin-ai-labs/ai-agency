# RAG Implementation with pgvector

**Estimated Effort**: 29-37 hours (~1 week for 1 developer, 2-3 days for 2 developers in parallel)
**Priority**: HIGH (enables knowledge-based agents)
**Status**: Not Started

## Overview

Implement complete RAG (Retrieval Augmented Generation) pipeline with pgvector integration. This includes document ingestion, embeddings generation, vector storage, semantic search, and context retrieval for LLMs.

**Architecture**:
- Multi-LLM adapter pattern for embeddings (OpenAI, Vertex AI)
- Repository pattern for data access
- SQLModel + Alembic for database migrations
- Async-first with comprehensive error handling
- Multi-tenant isolation
- 80%+ test coverage

---

## Phase 1: Database Foundation (3-4 hours)

### Step 1.1: Create Alembic Migration
**File**: `app/db/migrations/versions/002_add_pgvector_embeddings.py`
**Estimated**: 2 hours
**Dependencies**: pgvector extension enabled

- [ ] Create new Alembic migration file
- [ ] Add `embedding` column to `document_chunks` table
  ```sql
  ALTER TABLE document_chunks ADD COLUMN embedding vector(1536);
  ```

- [ ] Add metadata columns
  ```sql
  ALTER TABLE document_chunks ADD COLUMN embedding_model VARCHAR(100);
  ALTER TABLE document_chunks ADD COLUMN embedding_generated_at TIMESTAMP;
  ALTER TABLE document_chunks ADD COLUMN chunk_index INTEGER;
  ALTER TABLE document_chunks ADD COLUMN chunk_size INTEGER;
  ```

- [ ] Create IVFFlat index for vector search
  ```sql
  CREATE INDEX document_chunks_embedding_idx
  ON document_chunks USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);
  ```

- [ ] Create composite index for tenant-filtered searches
  ```sql
  CREATE INDEX document_chunks_tenant_embedding_idx
  ON document_chunks(tenant_id)
  WHERE embedding IS NOT NULL;
  ```

- [ ] Run migration: `alembic upgrade head`

### Step 1.2: Update DocumentChunk Model
**File**: `app/db/models.py`
**Estimated**: 1 hour
**Dependencies**: Step 1.1 complete

- [ ] Install `pgvector[sqlalchemy]` dependency

- [ ] Import Vector type from pgvector
  ```python
  from pgvector.sqlalchemy import Vector
  ```

- [ ] Add embedding field to `DocumentChunk` model
  ```python
  embedding: list[float] | None = Field(
      default=None,
      sa_column=Column(Vector(1536), nullable=True)
  )
  ```

- [ ] Add metadata fields
  - `embedding_model: str | None = None`
  - `embedding_generated_at: datetime | None = None`
  - `chunk_index: int = Field(default=0)`
  - `chunk_size: int = Field(default=0)`

- [ ] Update `chunk_metadata` field to use JSON column

- [ ] Add model validation for embedding dimensions

---

## Phase 2: Embeddings Layer (4-5 hours)

### Step 2.1: Create Base Embeddings Adapter
**File**: `app/adapters/embeddings_base.py`
**Estimated**: 1 hour
**Dependencies**: None

- [ ] Create abstract `BaseEmbeddingsAdapter` class
  - Follow pattern from `app/adapters/llm_base.py`
  - Add `provider_name`, `model_name`, `dimension` attributes
  - Add request counting and telemetry

- [ ] Define abstract method: `generate_embedding(text: str) -> list[float]`

- [ ] Define abstract method: `generate_embeddings_batch(texts: list[str], batch_size: int) -> list[list[float]]`

- [ ] Add `get_dimension() -> int` method

- [ ] Add docstrings and type hints

### Step 2.2: Implement OpenAI Embeddings Adapter
**File**: `app/adapters/embeddings_openai.py`
**Estimated**: 1.5 hours
**Dependencies**: Step 2.1 complete

- [ ] Create `OpenAIEmbeddingsAdapter` class
  - Default model: `text-embedding-3-small` (1536 dimensions)
  - Use existing `AsyncOpenAI` client pattern

- [ ] Implement `generate_embedding()` method
  - Call OpenAI embeddings API
  - Handle errors and retries

- [ ] Implement `generate_embeddings_batch()` method
  - Batch size: 100 texts per request
  - Handle OpenAI batch limits (max 2048)

- [ ] Add rate limiting using `TokenBucketRateLimiter`
  - Reuse pattern from `llm_openai.py`

- [ ] Add retry logic with exponential backoff
  - Use `@retry` decorator

- [ ] Add timeout handling

- [ ] Handle text truncation (max 8191 tokens)

### Step 2.3: Implement Vertex AI Embeddings Adapter
**File**: `app/adapters/embeddings_vertex.py`
**Estimated**: 1.5 hours
**Dependencies**: Step 2.1 complete

- [ ] Create `VertexEmbeddingsAdapter` class
  - Default model: `text-embedding-004` (768 dimensions)
  - Use `vertexai` library pattern from `llm_vertex.py`

- [ ] Implement `generate_embedding()` method

- [ ] Implement `generate_embeddings_batch()` method
  - Batch size: 250 texts per request
  - Handle Vertex AI batch limits

- [ ] Add retry logic for transient failures

- [ ] Handle authentication via Application Default Credentials

- [ ] Handle text truncation (max 20,000 characters)

### Step 2.4: Create Embeddings Factory
**File**: `app/adapters/embeddings_factory.py`
**Estimated**: 1 hour
**Dependencies**: Steps 2.2, 2.3 complete

- [ ] Create `get_embeddings_adapter()` factory function
  - Follow pattern from `llm_factory.py`
  - Support provider selection via config

- [ ] Add configuration to `app/config.py`
  ```python
  embeddings_provider: Literal["openai", "vertex"] = "openai"
  embeddings_model: str | None = None
  openai_embeddings_model: str = "text-embedding-3-small"
  vertex_embeddings_model: str = "text-embedding-004"
  ```

- [ ] Implement provider selection logic

- [ ] Add fallback logic if primary provider fails

- [ ] Cache adapter instances per provider

---

## Phase 3: Document Processing (5-6 hours)

### Step 3.1: Implement Text Chunking Strategy
**File**: `app/rag/chunking.py`
**Estimated**: 2 hours
**Dependencies**: None

- [ ] Create `DocumentChunker` class

- [ ] Implement recursive text splitting algorithm
  1. Split by paragraphs (`\n\n`)
  2. If paragraph > chunk_size, split by sentences
  3. If sentence > chunk_size, split by words
  4. Add overlap by including last N characters from previous chunk
  5. Track chunk indices and positions

- [ ] Add configuration parameters
  - Default chunk_size: 1000 characters (~250 tokens)
  - Default overlap: 200 characters
  - Min chunk_size: 100 characters
  - Max chunk_size: 4000 characters

- [ ] Implement semantic chunking (preserve sentence boundaries)

- [ ] Implement markdown-aware chunking (preserve headers)

- [ ] Add overlap management

- [ ] Preserve metadata during chunking (page numbers, sections)

- [ ] Handle edge cases (empty docs, very small/large chunks)

### Step 3.2: Implement Document Loader
**File**: `app/rag/document_loader.py`
**Estimated**: 2 hours
**Dependencies**: None

- [ ] Install dependencies
  - Add `PyPDF2>=3.0.0` to pyproject.toml
  - Add `python-docx>=1.1.0` to pyproject.toml

- [ ] Create `DocumentLoader` class

- [ ] Implement PDF parsing
  - Use `PyPDF2` or `pdfplumber`
  - Extract text and page numbers

- [ ] Implement DOCX parsing
  - Use `python-docx`
  - Extract text and structure

- [ ] Implement TXT/MD parsing
  - Handle encoding issues (UTF-8, latin-1 fallback)

- [ ] Extract document metadata (headings, sections)

- [ ] Add file type validation

- [ ] Add file size validation

- [ ] Implement error handling
  - Raise `ValidationError` for unsupported formats
  - Handle corrupted files gracefully
  - Log warnings for extraction issues

### Step 3.3: Rewrite Document Ingestion Service
**File**: `app/rag/ingestion.py`
**Estimated**: 2 hours
**Dependencies**: Steps 3.1, 3.2, Phase 2 complete

- [ ] Complete `DocumentIngestion` class implementation

- [ ] Implement ingestion pipeline orchestration
  1. Load document → DocumentLoader
  2. Validate and parse → extract text + metadata
  3. Chunk text → DocumentChunker
  4. Generate embeddings (batch) → EmbeddingsAdapter
  5. Store chunks + embeddings → DocumentChunkRepository
  6. Return ingestion stats

- [ ] Add batch processing
  - Chunk batch size: 100 chunks per embedding request
  - Database insert batch size: 500 chunks
  - Commit every 1000 chunks for large documents

- [ ] Implement idempotency (skip already-ingested docs)

- [ ] Add transaction management for atomic operations

- [ ] Add progress tracking and logging

- [ ] Include error recovery (continue on single doc failure)

- [ ] Return ingestion statistics (success/failure counts)

---

## Phase 4: Retrieval System (4-5 hours)

### Step 4.1: Create DocumentChunk Repository
**File**: `app/db/repositories/document_chunk_repository.py`
**Estimated**: 2 hours
**Dependencies**: Phase 1 complete

- [ ] Create `DocumentChunkRepository` class
  - Follow pattern from `conversation_repository.py`
  - Inherit from `BaseRepository`

- [ ] Implement CRUD operations
  - `create_chunk()`: Insert single chunk
  - `create_chunks_batch()`: Bulk insert with COPY
  - `get_by_id()`: Retrieve chunk
  - `delete_by_document_id()`: Delete all chunks for document
  - `get_by_document_id()`: Get all chunks for document

- [ ] Add tenant isolation to all queries

- [ ] Include soft delete support (add `is_deleted` flag)

- [ ] Optimize bulk inserts
  - Use PostgreSQL COPY for 10x faster inserts
  - Batch commits every 500 rows
  - Use `RETURNING *` to get inserted IDs

### Step 4.2: Rewrite Vector Retriever
**File**: `app/rag/retriever.py`
**Estimated**: 2 hours
**Dependencies**: Step 4.1, Phase 2 complete

- [ ] Complete `VectorRetriever` class implementation

- [ ] Implement `search()` method
  - Generate query embedding internally
  - Call `search_by_embedding()`

- [ ] Implement `search_by_embedding()` method
  - Query with pre-computed embedding
  - Use cosine similarity: `ORDER BY embedding <=> query_embedding`
  - Add tenant isolation
  - Filter by `is_deleted = FALSE`

- [ ] Implement `search_with_filters()` method
  - Add metadata filtering
  - Support multiple filter conditions

- [ ] Add similarity threshold filtering (default: 0.7)

- [ ] Implement pagination support (limit/offset)

- [ ] Add result re-ranking (optional: cross-encoder)

- [ ] Optimize query performance
  - Use IVFFlat index
  - Add `EXPLAIN ANALYZE` logging in debug mode
  - Cache frequently-used query embeddings

### Step 4.3: Implement Hybrid Retriever (Optional)
**File**: `app/rag/hybrid_retriever.py`
**Estimated**: 2 hours (optional)
**Dependencies**: Step 4.2 complete

- [ ] Create `HybridRetriever` class

- [ ] Add full-text search using PostgreSQL tsvector
  - Create GIN index on `to_tsvector(content)`
  - Use `ts_rank()` for keyword relevance

- [ ] Combine vector and keyword scores
  - Formula: `final_score = α × vector_score + β × keyword_score`
  - Default weights: α=0.7 (vector), β=0.3 (keyword)

- [ ] Implement Reciprocal Rank Fusion (RRF)

### Step 4.4: Create RAG Context Manager
**File**: `app/rag/context_manager.py`
**Estimated**: 1.5 hours
**Dependencies**: Step 4.2 complete

- [ ] Create `ContextManager` class

- [ ] Retrieve relevant chunks and format for LLM
  - Add clear separators and metadata
  - Include document source and chunk index

- [ ] Implement token counting (estimate: 4 chars = 1 token)
  - Optional: Use `tiktoken` for accurate counting

- [ ] Manage context window size (default: 8000 tokens)

- [ ] Add citation tracking (source document + chunk index)

- [ ] Format output with structure
  ```
  Context from documents:

  [Document: report.pdf, Page 3, Chunk 1]
  {chunk content}

  [Document: report.pdf, Page 3, Chunk 2]
  {chunk content}
  ```

---

## Phase 5: Integration & API (3-4 hours)

### Step 5.1: Create RAG Service Orchestrator
**File**: `app/rag/rag_service.py`
**Estimated**: 1.5 hours
**Dependencies**: All RAG components complete

- [ ] Create high-level `RAGService` class

- [ ] Implement `ingest_document()` method
  - Full ingestion pipeline
  - Return ingestion statistics

- [ ] Implement `query()` method
  - Retrieve relevant context for query
  - Return formatted context with citations

- [ ] Implement `update_document()` method
  - Re-ingest updated document
  - Handle version tracking

- [ ] Implement `delete_document()` method
  - Remove document and all chunks
  - Soft delete option

- [ ] Add caching for expensive operations

- [ ] Include comprehensive logging and metrics

- [ ] Add async context manager support

### Step 5.2: Create API Endpoints
**File**: `app/api/rag_endpoints.py`
**Estimated**: 1.5 hours
**Dependencies**: Step 5.1 complete

- [ ] Create FastAPI router for RAG operations

- [ ] Implement `POST /api/v1/documents/ingest`
  - Upload and ingest document
  - File upload validation (size, type)

- [ ] Implement `POST /api/v1/documents/query`
  - Semantic search endpoint
  - Return relevant chunks with scores

- [ ] Implement `GET /api/v1/documents/{document_id}/chunks`
  - List chunks for document

- [ ] Implement `DELETE /api/v1/documents/{document_id}`
  - Delete document and chunks

- [ ] Implement `GET /api/v1/documents/{document_id}/status`
  - Check ingestion status

- [ ] Create Pydantic request/response models
  ```python
  class DocumentIngestRequest(BaseModel):
      file: UploadFile
      document_id: str
      metadata: dict[str, Any] = {}

  class DocumentQueryRequest(BaseModel):
      query: str
      top_k: int = 5
      filters: dict[str, Any] = {}
      similarity_threshold: float = 0.7
  ```

- [ ] Add tenant authentication (use existing auth)

- [ ] Add rate limiting per tenant

- [ ] Include comprehensive error handling

### Step 5.3: Create Example RAG Agent Flow
**File**: `app/flows/rag_qa_flow.py`
**Estimated**: 1.5 hours
**Dependencies**: Step 5.1 complete

- [ ] Create `RAGQuestionAnsweringFlow` class
  - Inherit from `BaseFlow`

- [ ] Implement multi-step flow
  1. Receive user question
  2. Retrieve relevant context via RAG
  3. Generate answer using LLM with context
  4. Include citations in response

- [ ] Add conversation history support

- [ ] Include error handling and fallbacks

- [ ] Add telemetry (query latency, retrieval quality)

- [ ] Create example usage in docstring

---

## Phase 6: Testing (6-8 hours)

### Step 6.1: Unit Tests - Embeddings Adapters
**Files**: `tests/test_adapters/test_embeddings_*.py`
**Estimated**: 2 hours
**Coverage Target**: 85%+

- [ ] Create `tests/test_adapters/test_embeddings_openai.py`
  - Test generate single embedding (dimension check)
  - Test generate batch embeddings (verify batch size)
  - Test rate limiting (mock rate limiter)
  - Test API errors (retry logic)
  - Test timeout scenarios
  - Test with empty/null inputs

- [ ] Create `tests/test_adapters/test_embeddings_vertex.py`
  - Similar tests for Vertex AI adapter
  - Test batch size limits (250)

- [ ] Use `pytest-mock` to mock API calls

### Step 6.2: Unit Tests - Document Processing
**Files**: `tests/test_rag/test_*.py`
**Estimated**: 2.5 hours
**Coverage Target**: 80%+

- [ ] Create `tests/test_rag/test_chunking.py`
  - Test various text sizes
  - Test overlap logic
  - Test edge cases (empty, very small, very large)
  - Test markdown-aware chunking

- [ ] Create `tests/test_rag/test_document_loader.py`
  - Test PDF parsing
  - Test DOCX parsing
  - Test TXT parsing
  - Test error handling (corrupted files, unsupported formats)

- [ ] Create `tests/test_rag/test_ingestion.py`
  - Test end-to-end pipeline with mocked embeddings
  - Test idempotency
  - Test error recovery

- [ ] Add sample test files to `tests/fixtures/`
  - Sample PDF, DOCX, TXT files

### Step 6.3: Unit Tests - Retrieval
**Files**: `tests/test_rag/test_retriever.py`, `tests/test_rag/test_context_manager.py`
**Estimated**: 2 hours
**Coverage Target**: 85%+

- [ ] Create `tests/test_rag/test_retriever.py`
  - Test vector search (similarity scoring, top-k results)
  - Test filtering (tenant isolation, metadata filters)
  - Test pagination (limit/offset)
  - Test edge cases (no results, identical embeddings)

- [ ] Create `tests/test_rag/test_context_manager.py`
  - Test context building
  - Test token limits
  - Test formatting
  - Test citation tracking

- [ ] Use test database with sample embeddings

### Step 6.4: Integration Tests - End-to-End RAG
**File**: `tests/integration/test_rag_e2e.py`
**Estimated**: 1.5 hours
**Coverage Target**: 75%+

- [ ] Test full pipeline: ingest → query → answer

- [ ] Test multi-document scenarios

- [ ] Verify tenant isolation

- [ ] Performance benchmarks
  - Ingest time
  - Query latency

- [ ] Test concurrent operations

- [ ] Test database transaction handling

- [ ] Use test PostgreSQL with pgvector enabled

### Step 6.5: Repository Tests
**File**: `tests/test_db/test_document_chunk_repository.py`
**Estimated**: 1.5 hours
**Coverage Target**: 90%+

- [ ] Test CRUD operations

- [ ] Test bulk inserts (performance validation)

- [ ] Test tenant isolation

- [ ] Test vector search queries

- [ ] Test soft delete functionality

- [ ] Follow pattern from `test_weather_repository.py`

---

## Phase 7: Performance & Production (4-5 hours)

### Step 7.1: Add Monitoring & Observability
**File**: `app/rag/telemetry.py`
**Estimated**: 1.5 hours

- [ ] Add metrics collection
  - Ingestion rate (docs/min, chunks/min)
  - Query latency (p50, p95, p99)
  - Embedding generation time
  - Cache hit rates

- [ ] Add structured logging with correlation IDs

- [ ] Add performance counters

- [ ] Include cost tracking (API calls × pricing)

- [ ] Use `structlog` (already in dependencies)

### Step 7.2: Performance Optimization
**Estimated**: 2 hours

- [ ] Database tuning
  - Tune IVFFlat index lists parameter (default: 100)
  - Add partial indexes for common queries
  - Enable query plan caching
  - Tune PostgreSQL work_mem for vector ops

- [ ] Caching implementation
  - Cache query embeddings (Redis or in-memory)
  - Cache frequently-accessed chunks
  - Implement cache warming for popular queries

- [ ] Batching optimization
  - Optimize batch sizes for embeddings (100-250)
  - Use database connection pooling
  - Implement bulk operations everywhere

- [ ] Add configuration to `app/config.py`

### Step 7.3: Add Data Migration Script
**File**: `scripts/migrate_embeddings.py`
**Estimated**: 1.5 hours

- [ ] Create script to backfill embeddings for existing chunks

- [ ] Process in batches to avoid memory issues

- [ ] Add progress tracking and resume capability

- [ ] Handle failures gracefully (skip and log)

- [ ] Generate migration report (success/failure counts)

- [ ] Add CLI arguments
  - `--tenant-id`: Target tenant
  - `--batch-size`: Batch size
  - `--resume`: Resume from last position

### Step 7.4: Documentation
**Files**: `docs/rag/*.md`
**Estimated**: 2 hours

- [ ] Create `docs/rag/README.md`
  - Architecture overview
  - Component descriptions
  - Architecture diagrams

- [ ] Create `docs/rag/USAGE.md`
  - API documentation with examples
  - Code examples for ingestion and query
  - Common use cases

- [ ] Create `docs/rag/PERFORMANCE.md`
  - Tuning guide
  - Performance benchmarks
  - Cost estimation guide

- [ ] Add troubleshooting guide
  - Common issues and solutions
  - Debugging tips

---

## Dependencies

### New Python Packages
Add to `pyproject.toml`:
```toml
dependencies = [
    # ... existing
    "PyPDF2>=3.0.0",           # PDF parsing
    "python-docx>=1.1.0",      # DOCX parsing
    "tiktoken>=0.5.0",         # Token counting (optional)
    "pgvector[sqlalchemy]",    # pgvector support
]
```

### Database Requirements
- PostgreSQL 12+ with pgvector extension
- Recommended: PostgreSQL 15+ for better vector performance

### External Services
- OpenAI API (embeddings + LLM)
- OR Vertex AI (embeddings + LLM)

---

## Parallel Work Streams

To accelerate development, work can be parallelized:

**Stream A** (Database/Backend):
- Phase 1: Database migrations
- Phase 4: Repository & retrieval
- Phase 6.5: Repository tests

**Stream B** (ML/Embeddings):
- Phase 2: Embeddings adapters
- Phase 6.1: Embeddings tests

**Stream C** (Document Processing):
- Phase 3: Chunking & loading
- Phase 6.2: Processing tests

**Stream D** (Integration):
- Phase 5: API & orchestration (depends on A, B, C)
- Phase 6.4: E2E tests

**Stream E** (Production):
- Phase 7: Monitoring & optimization (can start in parallel)

---

## Risk Mitigation

### High-Risk Areas

**1. pgvector Performance**
- **Risk**: Index tuning may require iteration
- **Mitigation**: Start with IVFFlat, benchmark, consider HNSW for scale

**2. Embedding Costs**
- **Risk**: OpenAI embeddings expensive at scale
- **Mitigation**: Cache aggressively, use batch APIs, consider open-source alternatives

**3. Chunking Strategy**
- **Risk**: Finding optimal chunk size is application-specific
- **Mitigation**: Make configurable, run A/B tests, monitor retrieval quality

**4. Multi-Tenancy**
- **Risk**: Ensuring perfect tenant isolation
- **Mitigation**: Add tenant_id to all queries, test isolation thoroughly

### Medium-Risk Areas

**1. Document Parsing**
- **Risk**: PDFs can be complex (scanned images, tables)
- **Mitigation**: Use OCR fallback if needed, handle failures gracefully

**2. Query Performance**
- **Risk**: Large embeddings datasets can slow down
- **Mitigation**: Proper indexing, query optimization, caching

---

## Success Criteria

### Functional Requirements
- [ ] Ingest documents (PDF, DOCX, TXT) successfully
- [ ] Generate embeddings using OpenAI and Vertex AI
- [ ] Store embeddings in pgvector
- [ ] Retrieve relevant chunks with >0.7 similarity
- [ ] Build context windows for LLM
- [ ] API endpoints working with proper auth
- [ ] Example RAG flow functioning end-to-end

### Non-Functional Requirements
- [ ] 80%+ test coverage achieved
- [ ] All tests passing
- [ ] Tenant isolation verified
- [ ] Query latency < 200ms (p95)
- [ ] Ingestion throughput > 100 chunks/min
- [ ] No memory leaks or resource exhaustion
- [ ] Documentation complete

---

## Post-Implementation Enhancements

Future improvements (not in initial scope):

- [ ] Advanced chunking: Semantic chunking using sentence transformers
- [ ] Re-ranking: Add cross-encoder for result re-ranking
- [ ] Hybrid search: Combine vector + keyword search
- [ ] Query expansion: Use LLM to expand queries
- [ ] Feedback loop: Collect user feedback to improve retrieval
- [ ] Multi-modal: Support images, tables from PDFs
- [ ] Incremental updates: Smart re-embedding of changed documents
- [ ] Query analytics: Track popular queries and optimize
