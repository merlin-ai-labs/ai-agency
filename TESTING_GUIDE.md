# Testing Guide

Quick reference for running tests and checking coverage.

## Prerequisites

```bash
# Activate virtual environment
source venv/bin/activate

# Install test dependencies (if not already installed)
pip install pytest pytest-asyncio pytest-cov httpx
```

## Running Tests

### Run All Tests
```bash
pytest -v
```

### Run Tests with Coverage
```bash
pytest -v --cov=app --cov-report=term-missing --cov-report=html
```

### Run Specific Test File
```bash
pytest tests/test_main.py -v
```

### Run Specific Test Function
```bash
pytest tests/test_main.py::test_healthz -v
```

### Run Tests by Marker (when markers are added)
```bash
# Unit tests only
pytest -m unit -v

# Integration tests only
pytest -m integration -v

# Skip slow tests
pytest -m "not slow" -v
```

## Coverage Reports

### View Coverage Summary in Terminal
```bash
pytest --cov=app --cov-report=term
```

### Generate HTML Coverage Report
```bash
pytest --cov=app --cov-report=html

# Open in browser
open htmlcov/index.html
```

### Check Coverage Threshold
```bash
# Fails if coverage is below 8%
pytest --cov=app --cov-fail-under=8
```

### Generate XML Coverage (for CI/CD)
```bash
pytest --cov=app --cov-report=xml
```

## Testing Deployed Service

### Test Swagger UI
```bash
curl -I https://ai-agency-4ebxrg4hdq-ew.a.run.app/docs
# Should return: HTTP/2 200
```

### Test OpenAPI Schema
```bash
curl -s https://ai-agency-4ebxrg4hdq-ew.a.run.app/openapi.json | jq .
```

### Test Create Run Endpoint
```bash
curl -X POST https://ai-agency-4ebxrg4hdq-ew.a.run.app/runs \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "test-tenant",
    "flow_name": "maturity_assessment",
    "input_data": {}
  }' | jq .
```

### Test Get Run Endpoint
```bash
curl -s https://ai-agency-4ebxrg4hdq-ew.a.run.app/runs/test-123 | jq .
```

## Running Local Development Server

### Start Server
```bash
# Option 1: Direct uvicorn
uvicorn app.main:app --reload --port 8080

# Option 2: Using Python module
python -m uvicorn app.main:app --reload --port 8080

# Option 3: Using the app directly
python app/main.py
```

### Test Local Server
```bash
# Health check
curl http://localhost:8080/healthz

# Swagger UI
open http://localhost:8080/docs

# OpenAPI schema
curl http://localhost:8080/openapi.json
```

## Debugging Tests

### Run with Print Statements
```bash
pytest -v -s
```

### Run with Debugger
```bash
# Add breakpoint in test:
# import pdb; pdb.set_trace()

pytest -v -s
```

### Run with More Verbose Output
```bash
pytest -vv
```

### Show Local Variables on Failure
```bash
pytest -l
```

### Stop on First Failure
```bash
pytest -x
```

## Test File Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_main.py            # API endpoint tests
└── test_tools.py           # Tool registry tests

# Wave 2 additions (planned):
tests/
├── test_config.py          # Configuration tests
├── test_db/
│   ├── test_models.py      # Model tests
│   └── test_repositories.py # Repository tests
├── test_adapters/
│   ├── test_llm_factory.py
│   ├── test_llm_openai.py
│   └── test_llm_vertex.py
├── test_rag/
│   ├── test_ingestion.py
│   └── test_retriever.py
└── test_flows/
    ├── test_maturity_assessment.py
    └── test_usecase_grooming.py
```

## Writing Tests

### Basic Test Structure
```python
import pytest

def test_example():
    """Test description"""
    # Arrange
    input_data = "test"

    # Act
    result = function_to_test(input_data)

    # Assert
    assert result == expected_output
```

### Async Test Structure
```python
import pytest

@pytest.mark.asyncio
async def test_async_example():
    """Test async function"""
    result = await async_function()
    assert result is not None
```

### Using Fixtures
```python
@pytest.fixture
def sample_data():
    """Provide sample data for tests"""
    return {"key": "value"}

def test_with_fixture(sample_data):
    """Test using fixture"""
    assert sample_data["key"] == "value"
```

### Mocking External Calls
```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_with_mock():
    """Test with mocked external call"""
    with patch("app.some_module.external_call") as mock_call:
        mock_call.return_value = "mocked"
        result = await function_that_calls_external()
        assert result == "mocked"
```

## Common Issues

### Issue: Module not found
```bash
# Solution: Install in development mode
pip install -e .
```

### Issue: Tests not discovered
```bash
# Check pytest can find tests
pytest --collect-only
```

### Issue: Import errors
```bash
# Make sure you're in the project root
cd /path/to/ConsultingAgency
pytest
```

### Issue: Database connection errors
```bash
# Check DATABASE_URL is set for testing
export DATABASE_URL="postgresql://localhost/test_db"
pytest
```

## CI/CD Integration

### GitHub Actions Workflow
Tests run automatically on:
- Every push to main
- Every pull request
- Manual trigger

### View Test Results
```bash
# Check latest workflow run
gh run list

# View specific run logs
gh run view <run-id>
```

## Performance

### Current Test Performance
- **Total Tests:** 5
- **Execution Time:** 0.50s
- **Average per Test:** 0.10s

### Performance Goals
- Unit tests: < 1s per test
- Integration tests: < 5s per test
- E2E tests: < 30s per test
- Total suite: < 5 minutes

## Test Coverage Goals

### Current Coverage
- **Overall:** 8.06%
- **app/main.py:** 100%
- **app/tools/registry.py:** 63.64%

### Wave 2 Goals
- **Overall:** 80%+
- **Critical modules:** 90%+
- **All modules:** 70%+

## Resources

- **pytest documentation:** https://docs.pytest.org/
- **Coverage.py documentation:** https://coverage.readthedocs.io/
- **pytest-asyncio documentation:** https://pytest-asyncio.readthedocs.io/

## Quick Commands Cheat Sheet

```bash
# Run all tests with coverage
pytest -v --cov=app

# Run and generate HTML report
pytest -v --cov=app --cov-report=html && open htmlcov/index.html

# Run specific test file
pytest tests/test_main.py -v

# Run with print output
pytest -v -s

# Stop on first failure
pytest -x

# Run in parallel (when installed: pip install pytest-xdist)
pytest -n auto

# Clear pytest cache
pytest --cache-clear

# Show slowest tests
pytest --durations=10
```

## Contact

For questions about testing:
- Check test files for examples
- Review TEST_SUMMARY.md for current status
- See WAVE2_TEST_PRIORITIES.md for roadmap
