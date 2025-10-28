# Coding Standards - AI Agency Platform

## Overview

This document defines the coding standards for the AI Agency platform. All code must adhere to these standards to ensure consistency, maintainability, and quality across the codebase.

## Table of Contents

1. [Type Hints](#type-hints)
2. [Docstrings](#docstrings)
3. [Async/Await Patterns](#asyncawait-patterns)
4. [Error Handling](#error-handling)
5. [Naming Conventions](#naming-conventions)
6. [Import Organization](#import-organization)
7. [Code Structure](#code-structure)
8. [Testing Requirements](#testing-requirements)
9. [Performance Guidelines](#performance-guidelines)

---

## Type Hints

### Requirements

- **ALL** public functions, methods, and class attributes MUST have type hints
- Use Python 3.11+ union syntax: `str | None` instead of `Optional[str]`
- Use `from typing import Protocol` for interface definitions
- Use generic types from `collections.abc` when possible

### Examples

```python
# Good - Complete type hints
async def process_document(
    doc_id: str,
    metadata: dict[str, str | int],
    options: ProcessingOptions | None = None
) -> ProcessingResult:
    """Process a document with given options."""
    pass

# Good - Protocol for duck typing
from typing import Protocol

class LLMProvider(Protocol):
    """Protocol for LLM provider implementations."""

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7
    ) -> str:
        """Generate completion from messages."""
        ...

# Bad - Missing type hints
def process_document(doc_id, metadata, options=None):
    pass

# Bad - Old-style Optional
from typing import Optional
def get_user(user_id: str) -> Optional[User]:
    pass
```

### Type Alias Definitions

Define complex types as aliases for clarity:

```python
from typing import TypeAlias

MessageHistory: TypeAlias = list[dict[str, str]]
MetadataDict: TypeAlias = dict[str, str | int | float | bool]
JSONValue: TypeAlias = str | int | float | bool | None | dict[str, "JSONValue"] | list["JSONValue"]
```

---

## Docstrings

### Style: Google Style

Use Google-style docstrings for all public modules, classes, and functions.

### Requirements

- **ALL** public functions and classes MUST have docstrings
- Include type information in docstrings (complementing type hints)
- Document exceptions that can be raised
- Include usage examples for complex functions

### Examples

```python
async def retrieve_similar_documents(
    query: str,
    top_k: int = 5,
    filters: dict[str, str] | None = None
) -> list[Document]:
    """Retrieve documents similar to the query using vector search.

    Performs semantic search over the document collection using embeddings
    to find the most relevant documents matching the query.

    Args:
        query: The search query text to find similar documents for.
        top_k: Maximum number of documents to return. Defaults to 5.
        filters: Optional metadata filters to apply to the search.
            Keys are field names, values are filter values.

    Returns:
        A list of Document objects ordered by relevance (highest first).
        Returns empty list if no documents match.

    Raises:
        DatabaseError: If the database query fails.
        ValidationError: If query is empty or top_k is invalid.

    Example:
        >>> docs = await retrieve_similar_documents(
        ...     query="machine learning best practices",
        ...     top_k=3,
        ...     filters={"category": "technical"}
        ... )
        >>> for doc in docs:
        ...     print(f"{doc.title}: {doc.similarity_score}")
    """
    pass

class DocumentProcessor:
    """Processes and transforms documents for storage and retrieval.

    The DocumentProcessor handles document ingestion, text extraction,
    chunking, and metadata enrichment. It supports multiple document
    formats and provides async processing for high throughput.

    Attributes:
        chunk_size: Maximum size of text chunks in characters.
        chunk_overlap: Number of overlapping characters between chunks.
        supported_formats: Set of file extensions that can be processed.

    Example:
        >>> processor = DocumentProcessor(chunk_size=1000, chunk_overlap=100)
        >>> chunks = await processor.process("document.pdf")
        >>> print(f"Created {len(chunks)} chunks")
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """Initialize the document processor.

        Args:
            chunk_size: Maximum characters per chunk. Must be > 0.
            chunk_overlap: Characters to overlap between chunks. Must be < chunk_size.

        Raises:
            ValidationError: If chunk_size or chunk_overlap are invalid.
        """
        pass
```

---

## Async/Await Patterns

### Requirements

- Use `async def` for **ALL** I/O operations (database, API calls, file I/O)
- Use `await` for all async function calls - never use `.result()` or blocking calls
- Use `asyncio.gather()` for concurrent operations when possible
- Use async context managers (`async with`) for resource management

### Examples

```python
# Good - Async I/O operations
async def fetch_user_data(user_id: str) -> UserData:
    """Fetch user data from database."""
    async with get_db_session() as session:
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

# Good - Concurrent operations
async def fetch_all_dependencies(doc_ids: list[str]) -> list[Document]:
    """Fetch multiple documents concurrently."""
    tasks = [fetch_document(doc_id) for doc_id in doc_ids]
    return await asyncio.gather(*tasks)

# Good - Async context manager
class DatabaseSession:
    """Async database session manager."""

    async def __aenter__(self) -> "DatabaseSession":
        """Enter the context and acquire connection."""
        self.connection = await self.pool.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and release connection."""
        await self.connection.close()

# Bad - Blocking I/O in async function
async def bad_fetch_user(user_id: str) -> UserData:
    # Don't do this - blocks the event loop
    response = requests.get(f"/api/users/{user_id}")
    return response.json()

# Good - Use async HTTP client
async def good_fetch_user(user_id: str) -> UserData:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"/api/users/{user_id}")
        return response.json()
```

### Background Tasks

For long-running operations, use background tasks:

```python
from fastapi import BackgroundTasks

@router.post("/documents/process")
async def process_document(
    document_id: str,
    background_tasks: BackgroundTasks
) -> dict[str, str]:
    """Start document processing in background."""
    background_tasks.add_task(process_document_async, document_id)
    return {"status": "processing", "document_id": document_id}
```

---

## Error Handling

### Requirements

- **ALWAYS** use custom exception classes from `app.core.exceptions`
- Never use bare `except:` - always specify exception types
- Include context in exception messages (IDs, operation names)
- Use retry decorators for transient failures
- Log exceptions with structured context

### Custom Exception Hierarchy

```python
from app.core.exceptions import (
    AIAgencyError,        # Base exception
    DatabaseError,        # Database operations
    LLMError,             # LLM API calls
    FlowError,            # Flow execution
    ToolError,            # Tool execution
    ValidationError,      # Input validation
    AuthError,            # Authentication
    RateLimitError,       # Rate limiting
)
```

### Examples

```python
# Good - Specific exception handling
async def get_document(doc_id: str) -> Document:
    """Retrieve document by ID."""
    try:
        async with get_db_session() as session:
            result = await session.execute(
                select(Document).where(Document.id == doc_id)
            )
            doc = result.scalar_one_or_none()

            if doc is None:
                raise ValidationError(
                    f"Document not found: {doc_id}",
                    details={"document_id": doc_id}
                )

            return doc

    except SQLAlchemyError as e:
        logger.error(
            "Database error fetching document",
            extra={"document_id": doc_id, "error": str(e)}
        )
        raise DatabaseError(
            f"Failed to fetch document {doc_id}",
            operation="fetch_document"
        ) from e

# Good - Retry for transient failures
from app.core.decorators import retry

@retry(max_attempts=3, backoff_type="exponential", exceptions=(LLMError,))
async def call_llm(prompt: str) -> str:
    """Call LLM with retry logic."""
    try:
        response = await llm_client.complete(prompt)
        return response.text
    except Exception as e:
        raise LLMError(f"LLM call failed: {e}", provider="openai") from e

# Bad - Bare except
try:
    process_data()
except:  # Don't do this
    pass

# Bad - Generic exception without context
try:
    doc = fetch_document(doc_id)
except Exception:
    raise Exception("Error")  # Not helpful
```

### Validation

Use Pydantic for input validation:

```python
from pydantic import BaseModel, Field, field_validator

class ProcessingRequest(BaseModel):
    """Request to process a document."""

    document_id: str = Field(..., min_length=1, description="Document ID")
    chunk_size: int = Field(1000, ge=100, le=10000, description="Chunk size")

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, v: str) -> str:
        """Validate document ID format."""
        if not v.startswith("doc_"):
            raise ValidationError("Document ID must start with 'doc_'")
        return v
```

---

## Naming Conventions

### Python Naming Standards (PEP 8)

- **Modules**: `lowercase_with_underscores.py`
- **Classes**: `PascalCase`
- **Functions/Methods**: `snake_case`
- **Constants**: `UPPERCASE_WITH_UNDERSCORES`
- **Private**: `_leading_underscore`
- **Protected**: `_single_leading_underscore`
- **Dunder**: `__double_leading_underscore__` (avoid creating custom ones)

### Descriptive Naming

Use clear, descriptive names that reveal intent:

```python
# Good - Clear and descriptive
async def calculate_document_similarity_score(
    doc1: Document,
    doc2: Document
) -> float:
    """Calculate cosine similarity between two documents."""
    pass

user_authentication_token: str
max_retry_attempts: int = 3
is_document_processed: bool

# Bad - Unclear abbreviations
async def calc_sim(d1, d2) -> float:
    pass

usr_tok: str
max_ret: int = 3
is_proc: bool
```

### Naming Patterns

```python
# Boolean variables/functions
is_valid: bool
has_permission: bool
can_process: bool

# Collections
documents: list[Document]
user_ids: set[str]
metadata_by_id: dict[str, Metadata]

# Factory functions
def create_llm_client() -> LLMClient:
    pass

# Processing functions
async def process_document() -> ProcessingResult:
    pass
async def transform_data() -> TransformedData:
    pass

# Retrieval functions
async def get_document() -> Document:
    pass
async def fetch_users() -> list[User]:
    pass
async def retrieve_similar() -> list[Document]:
    pass
```

---

## Import Organization

### Order (PEP 8)

1. Standard library imports
2. Third-party imports
3. Local application imports

Separate each group with a blank line.

### Examples

```python
# Standard library
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Protocol

# Third-party
import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import select

# Local
from app.core.base import BaseTool
from app.core.decorators import retry, timeout
from app.core.exceptions import DatabaseError, ValidationError
from app.db.session import get_db_session
from app.models.document import Document

logger = logging.getLogger(__name__)
```

### Import Guidelines

- Use explicit imports: `from module import Class` not `from module import *`
- Group related imports together
- Sort imports alphabetically within each group
- Use absolute imports from project root

---

## Code Structure

### File Organization

```python
"""Module docstring describing purpose and contents.

This module implements document processing functionality including
text extraction, chunking, and metadata enrichment.
"""

# Imports
import logging
from typing import Protocol

# Constants
MAX_CHUNK_SIZE: int = 2000
DEFAULT_OVERLAP: int = 200

# Module-level logger
logger = logging.getLogger(__name__)

# Type definitions
ProcessingResult = dict[str, str | int]

# Classes
class DocumentProcessor:
    """Main class implementation."""
    pass

# Functions
async def process_document() -> None:
    """Helper functions."""
    pass

# Entry point (if applicable)
if __name__ == "__main__":
    pass
```

### Function Length

- Keep functions focused on a single responsibility
- Aim for < 50 lines per function
- Extract complex logic into helper functions
- Use early returns to reduce nesting

```python
# Good - Single responsibility, early returns
async def validate_and_process_document(doc_id: str) -> ProcessingResult:
    """Validate and process a document."""
    if not doc_id:
        raise ValidationError("Document ID required")

    doc = await fetch_document(doc_id)
    if doc is None:
        raise ValidationError(f"Document not found: {doc_id}")

    if doc.status == "processed":
        return ProcessingResult(status="already_processed")

    result = await process_document_content(doc)
    await save_processing_result(result)
    return result
```

### Class Design

- Use composition over inheritance
- Keep classes focused (Single Responsibility Principle)
- Use protocols for interface definitions
- Implement `__repr__` for debugging

```python
from typing import Protocol

# Define interfaces with protocols
class StorageProvider(Protocol):
    """Protocol for storage implementations."""

    async def save(self, key: str, data: bytes) -> str:
        """Save data and return storage URL."""
        ...

    async def load(self, key: str) -> bytes:
        """Load data from storage."""
        ...

# Implementation
class GCSStorageProvider:
    """Google Cloud Storage implementation."""

    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name

    async def save(self, key: str, data: bytes) -> str:
        """Save to GCS bucket."""
        # Implementation
        pass

    async def load(self, key: str) -> bytes:
        """Load from GCS bucket."""
        # Implementation
        pass

    def __repr__(self) -> str:
        return f"GCSStorageProvider(bucket={self.bucket_name})"
```

---

## Testing Requirements

### Coverage Requirements

- **Minimum 80% code coverage** for all new code
- 100% coverage for critical paths (authentication, payment, data loss)
- Test both success and failure cases

### Test Organization

```
tests/
├── unit/              # Unit tests (isolated)
├── integration/       # Integration tests (with dependencies)
├── e2e/              # End-to-end tests
├── fixtures/         # Shared test fixtures
└── conftest.py       # Pytest configuration
```

### Testing Patterns

```python
import pytest
from unittest.mock import AsyncMock, patch

from app.core.exceptions import DatabaseError, ValidationError
from app.services.document_service import DocumentService

class TestDocumentService:
    """Tests for DocumentService."""

    @pytest.fixture
    def mock_db_session(self):
        """Provide mock database session."""
        session = AsyncMock()
        return session

    @pytest.fixture
    def document_service(self, mock_db_session):
        """Provide DocumentService instance."""
        return DocumentService(session=mock_db_session)

    @pytest.mark.asyncio
    async def test_get_document_success(self, document_service, mock_db_session):
        """Test successful document retrieval."""
        # Arrange
        doc_id = "doc_123"
        mock_doc = Document(id=doc_id, title="Test")
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_doc

        # Act
        result = await document_service.get_document(doc_id)

        # Assert
        assert result == mock_doc
        mock_db_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, document_service, mock_db_session):
        """Test document not found raises ValidationError."""
        # Arrange
        doc_id = "nonexistent"
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None

        # Act & Assert
        with pytest.raises(ValidationError, match="Document not found"):
            await document_service.get_document(doc_id)

    @pytest.mark.asyncio
    async def test_get_document_database_error(self, document_service, mock_db_session):
        """Test database error is properly handled."""
        # Arrange
        doc_id = "doc_123"
        mock_db_session.execute.side_effect = Exception("Connection failed")

        # Act & Assert
        with pytest.raises(DatabaseError, match="Failed to fetch document"):
            await document_service.get_document(doc_id)
```

### Test Naming

- Test files: `test_<module_name>.py`
- Test functions: `test_<function>_<scenario>` (e.g., `test_get_document_success`)
- Use descriptive names that explain what is being tested

### Fixtures

Use pytest fixtures for test setup:

```python
@pytest.fixture
async def db_session():
    """Provide database session for tests."""
    async with get_test_db_session() as session:
        yield session
        await session.rollback()

@pytest.fixture
def sample_document() -> Document:
    """Provide sample document for testing."""
    return Document(
        id="doc_test_123",
        title="Test Document",
        content="Sample content",
        metadata={"category": "test"}
    )
```

---

## Performance Guidelines

### Async Best Practices

- Use `asyncio.gather()` for concurrent operations
- Avoid blocking operations in async functions
- Use connection pooling for database and HTTP clients
- Implement proper timeout handling

### Database Optimization

```python
# Good - Efficient query with select loading
async def get_documents_with_metadata(limit: int = 100) -> list[Document]:
    """Fetch documents with related metadata."""
    async with get_db_session() as session:
        result = await session.execute(
            select(Document)
            .options(selectinload(Document.metadata))  # Eager load
            .limit(limit)
        )
        return result.scalars().all()

# Bad - N+1 query problem
async def get_documents_with_metadata_bad(limit: int = 100) -> list[Document]:
    """Fetch documents (inefficient)."""
    docs = await session.execute(select(Document).limit(limit))
    for doc in docs:
        # This creates N additional queries
        doc.metadata  # Lazy loading triggers query per document
    return docs
```

### Caching

Use caching for expensive operations:

```python
from functools import lru_cache
from cachetools import TTLCache

# Sync function caching
@lru_cache(maxsize=128)
def get_config(key: str) -> str:
    """Get configuration value (cached)."""
    return expensive_config_lookup(key)

# Async function caching with TTL
cache = TTLCache(maxsize=100, ttl=300)  # 5 minutes

async def get_embedding(text: str) -> list[float]:
    """Get text embedding (cached)."""
    if text in cache:
        return cache[text]

    embedding = await call_embedding_api(text)
    cache[text] = embedding
    return embedding
```

### Resource Management

Always use context managers for resource cleanup:

```python
# Good - Automatic resource cleanup
async def process_file(file_path: str) -> None:
    """Process file with proper resource management."""
    async with aiofiles.open(file_path, 'r') as f:
        content = await f.read()
        await process_content(content)

# Database sessions
async def query_database() -> list[Record]:
    """Query with automatic session cleanup."""
    async with get_db_session() as session:
        result = await session.execute(select(Record))
        return result.scalars().all()
    # Session automatically closed
```

---

## Tools and Enforcement

### Ruff (Linting and Formatting)

Run before committing:

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### Mypy (Type Checking)

Run type checking:

```bash
mypy app/
```

### Pytest (Testing)

Run tests with coverage:

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=app --cov-report=html --cov-report=term

# Run specific test file
pytest tests/unit/test_document_service.py
```

---

## Summary Checklist

Before committing code, ensure:

- [ ] All functions have type hints
- [ ] All public functions have docstrings (Google style)
- [ ] I/O operations use async/await
- [ ] Custom exceptions are used for error handling
- [ ] Variable and function names are descriptive
- [ ] Imports are organized (stdlib, third-party, local)
- [ ] Code passes `ruff check` and `ruff format`
- [ ] Code passes `mypy` type checking
- [ ] Tests are written and pass
- [ ] Test coverage is >= 80%

---

## References

- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/async/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
