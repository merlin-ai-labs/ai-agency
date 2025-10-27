# Tests

Minimal test suite for AI Agency.

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_main.py

# Run with verbose output
pytest -v
```

## TODO

- [ ] Add integration tests for flows
- [ ] Add tests for database operations
- [ ] Add tests for LLM adapters (with mocking)
- [ ] Add tests for RAG ingestion and retrieval
- [ ] Add end-to-end tests
- [ ] Set up CI/CD with GitHub Actions
