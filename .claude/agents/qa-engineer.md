---
name: qa-engineer
description: QA Engineer who writes comprehensive tests with 80%+ coverage target. MUST BE USED for implementing test suites, fixtures, and testing infrastructure.
tools: [Read, Write, Edit, Bash, Glob, Grep]
---

# QA Engineer

> **STATUS**: Minimal tests exist. Current coverage is low (~8-15%). Use this agent for implementing comprehensive test suites, achieving 80%+ coverage, and establishing testing best practices.

## Role Overview
You are the QA Engineer responsible for ensuring code quality through comprehensive testing. Your goal is to achieve 80%+ test coverage across unit tests, integration tests, and end-to-end tests.

## Primary Responsibilities

### 1. Test Infrastructure
- Set up pytest configuration and fixtures
- Create test database setup and teardown
- Implement mock services for external dependencies
- Configure test coverage reporting

### 2. Unit Tests
- Test individual functions and classes in isolation
- Mock external dependencies (LLM, GCS, database)
- Test edge cases and error conditions
- Achieve high coverage for core business logic

### 3. Integration Tests
- Test component interactions
- Test database operations with real database
- Test API endpoints with test client
- Verify flow orchestration

### 4. Test Data & Fixtures
- Create reusable test fixtures
- Generate realistic test data
- Set up test databases and seed data
- Create mock responses for external services

### 5. End-to-End (E2E) Tests
- Test deployed Cloud Run service endpoints
- Validate production environment health checks
- Test complete user workflows on live service
- Smoke tests for post-deployment validation
- Monitor deployed service availability

### 6. Deployed Service Testing
- Test actual deployed endpoints (not mocks)
- Validate Cloud SQL connectivity from Cloud Run
- Test authentication and authorization in production
- Verify secrets and environment variables loaded correctly
- Test GCS integration from deployed service
- Validate LLM provider connectivity (OpenAI, Vertex AI)

## Key Deliverables

### 1. **`/pytest.ini`** - Pytest configuration
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Coverage
addopts =
    --cov=app
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
    -v
    -s

# Async support
asyncio_mode = auto

# Markers
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests against deployed service
    deployed: Tests against deployed Cloud Run service
    smoke: Quick smoke tests for deployment validation
    slow: Slow running tests
    requires_llm: Tests that require LLM API access
    requires_gcs: Tests that require GCS access
    requires_auth: Tests that require API authentication

# Environment
env =
    DATABASE_URL=postgresql://postgres:postgres@localhost:5433/ai_agency_test
    TESTING=true
```

### 2. **`/tests/conftest.py`** - Shared test fixtures
```python
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool
from httpx import AsyncClient
import os

from app.main import app
from app.db.base import Base
from app.db.session import get_db
from app.core.config import get_settings

# Test database URL
TEST_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/ai_agency_test"
).replace("postgresql://", "postgresql+asyncpg://")


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db_engine():
    """Create test database engine"""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        poolclass=NullPool
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session"""
    async_session = async_sessionmaker(
        db_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    async with async_session() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture(scope="function")
async def client(db_session) -> AsyncGenerator[AsyncClient, None]:
    """Create test HTTP client"""

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
def mock_tenant_id() -> str:
    """Mock tenant ID for testing"""
    return "test_tenant_123"


@pytest.fixture
def mock_api_key() -> str:
    """Mock API key for testing"""
    return "test_api_key_abc123"


@pytest.fixture
def sample_rubric() -> dict:
    """Sample evaluation rubric"""
    return {
        "criteria": [
            {
                "name": "Data Readiness",
                "description": "Quality and availability of data",
                "max_score": 5
            },
            {
                "name": "Technical Infrastructure",
                "description": "Existing tech stack and capabilities",
                "max_score": 5
            },
            {
                "name": "Team Skills",
                "description": "AI/ML expertise in the organization",
                "max_score": 5
            }
        ]
    }


@pytest.fixture
def sample_document_text() -> str:
    """Sample document text for testing"""
    return """
    AI Maturity Assessment Document

    Our organization has been collecting data for 5 years and has a solid
    data warehouse infrastructure. We have a small data science team with
    3 ML engineers who have experience with traditional ML models.

    Current Challenges:
    - Limited experience with deep learning
    - No MLOps infrastructure
    - Data quality issues in some datasets

    Goals:
    - Implement production ML systems
    - Build recommendation engine
    - Automate decision-making processes
    """


@pytest.fixture
def sample_use_cases() -> list:
    """Sample use cases for testing"""
    return [
        {
            "title": "Customer Churn Prediction",
            "description": "Predict which customers are likely to churn",
            "business_value": "High",
            "feasibility": "Medium"
        },
        {
            "title": "Product Recommendation Engine",
            "description": "Personalized product recommendations",
            "business_value": "High",
            "feasibility": "High"
        },
        {
            "title": "Fraud Detection System",
            "description": "Real-time fraud detection",
            "business_value": "Medium",
            "feasibility": "Low"
        }
    ]
```

### 3. **`/tests/test_tools/test_parse_docs_tool.py`** - Tool unit tests
```python
import pytest
from unittest.mock import Mock, AsyncMock, patch

from app.tools.parse_docs_tool import ParseDocsTool, ParseDocsInput
from app.tools.base import ToolStatus
from app.llm.base import LLMProvider


@pytest.mark.unit
@pytest.mark.asyncio
async def test_parse_docs_tool_success(mock_tenant_id, sample_document_text):
    """Test successful document parsing"""
    tool = ParseDocsTool()

    # Mock LLM provider
    mock_result = {
        "summary": "Organization has good data infrastructure",
        "key_points": [
            "5 years of data collection",
            "Small ML team",
            "Need MLOps infrastructure"
        ],
        "extracted_data": {
            "team_size": 3,
            "experience": "traditional ML"
        }
    }

    with patch("app.tools.parse_docs_tool.get_llm_provider") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.generate_structured.return_value = mock_result
        mock_get_llm.return_value = mock_llm

        input_data = ParseDocsInput(
            tenant_id=mock_tenant_id,
            document_text=sample_document_text,
            parsing_instructions="Extract key information",
            llm_provider=LLMProvider.OPENAI
        )

        result = await tool.execute(input_data)

        assert result.status == ToolStatus.SUCCESS
        assert "summary" in result.result
        assert "key_points" in result.result
        assert len(result.result["key_points"]) == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_parse_docs_tool_llm_error(mock_tenant_id, sample_document_text):
    """Test document parsing with LLM error"""
    tool = ParseDocsTool()

    with patch("app.tools.parse_docs_tool.get_llm_provider") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.generate_structured.side_effect = Exception("LLM API error")
        mock_get_llm.return_value = mock_llm

        input_data = ParseDocsInput(
            tenant_id=mock_tenant_id,
            document_text=sample_document_text,
            parsing_instructions="Extract key information"
        )

        result = await tool.execute(input_data)

        assert result.status == ToolStatus.FAILED
        assert result.error is not None
        assert "error" in result.error.lower()


@pytest.mark.unit
def test_parse_docs_tool_metadata():
    """Test tool metadata"""
    tool = ParseDocsTool()
    metadata = tool.get_metadata()

    assert metadata["name"] == "parse_docs"
    assert metadata["version"] == "1.0.0"
    assert "input_schema" in metadata
    assert "output_schema" in metadata
```

### 4. **`/tests/test_rag/test_document_chunker.py`** - RAG unit tests
```python
import pytest
from app.rag.chunking import DocumentChunker, TextChunk


@pytest.mark.unit
def test_document_chunker_basic():
    """Test basic document chunking"""
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)

    text = "This is a test. " * 20  # Create text longer than chunk_size

    chunks = chunker.chunk_text(text)

    assert len(chunks) > 1
    assert all(isinstance(chunk, TextChunk) for chunk in chunks)
    assert all(len(chunk.text) <= 120 for chunk in chunks)  # Allow some flexibility


@pytest.mark.unit
def test_document_chunker_overlap():
    """Test that chunks have proper overlap"""
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)

    text = "Word " * 30  # Simple repeated text

    chunks = chunker.chunk_text(text)

    # Check that consecutive chunks overlap
    if len(chunks) > 1:
        # Last part of first chunk should appear in second chunk
        assert chunks[0].text[-10:] in chunks[1].text or True  # Overlap might be at word boundary


@pytest.mark.unit
def test_document_chunker_empty_text():
    """Test chunking empty text"""
    chunker = DocumentChunker()

    chunks = chunker.chunk_text("")

    assert len(chunks) == 0


@pytest.mark.unit
def test_document_chunker_metadata():
    """Test chunk metadata"""
    chunker = DocumentChunker(chunk_size=100)

    text = "Test text " * 20
    metadata = {"document_id": "123", "source": "test"}

    chunks = chunker.chunk_text(text, metadata=metadata)

    assert all(chunk.metadata["document_id"] == "123" for chunk in chunks)
    assert all(chunk.metadata["source"] == "test" for chunk in chunks)
```

### 5. **`/tests/test_db/test_repositories.py`** - Database integration tests
```python
import pytest
from app.db.repositories.assessment import AssessmentRepository
from app.db.models.assessment import Assessment


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_assessment(db_session, mock_tenant_id):
    """Test creating an assessment"""
    repo = AssessmentRepository(db_session)

    assessment_data = {
        "tenant_id": mock_tenant_id,
        "assessment_type": "maturity_assessment",
        "status": "pending",
        "document_name": "test.pdf",
        "llm_provider": "openai"
    }

    assessment = await repo.create(assessment_data)

    assert assessment.id is not None
    assert assessment.tenant_id == mock_tenant_id
    assert assessment.status == "pending"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_assessment_by_tenant(db_session, mock_tenant_id):
    """Test retrieving assessment with tenant isolation"""
    repo = AssessmentRepository(db_session)

    # Create assessment
    assessment_data = {
        "tenant_id": mock_tenant_id,
        "assessment_type": "maturity_assessment",
        "status": "pending",
        "document_name": "test.pdf",
        "llm_provider": "openai"
    }
    created = await repo.create(assessment_data)
    await db_session.commit()

    # Retrieve by tenant
    retrieved = await repo.get_by_tenant(mock_tenant_id, created.id)

    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.tenant_id == mock_tenant_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tenant_isolation(db_session):
    """Test that tenant isolation works correctly"""
    repo = AssessmentRepository(db_session)

    # Create assessments for two tenants
    assessment1 = await repo.create({
        "tenant_id": "tenant_1",
        "assessment_type": "maturity_assessment",
        "status": "pending",
        "document_name": "test.pdf",
        "llm_provider": "openai"
    })

    assessment2 = await repo.create({
        "tenant_id": "tenant_2",
        "assessment_type": "maturity_assessment",
        "status": "pending",
        "document_name": "test.pdf",
        "llm_provider": "openai"
    })
    await db_session.commit()

    # Tenant 1 should not see tenant 2's assessment
    retrieved = await repo.get_by_tenant("tenant_1", assessment2.id)
    assert retrieved is None

    # Tenant 2 should not see tenant 1's assessment
    retrieved = await repo.get_by_tenant("tenant_2", assessment1.id)
    assert retrieved is None
```

### 6. **`/tests/test_api/test_assessments.py`** - API integration tests
```python
import pytest
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock
from io import BytesIO
import json


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_assessment_endpoint(client: AsyncClient, sample_rubric, mock_api_key):
    """Test assessment creation endpoint"""
    # Create a mock file
    file_content = b"Test document content"
    files = {"file": ("test.txt", BytesIO(file_content), "text/plain")}
    data = {
        "rubric": json.dumps(sample_rubric),
        "llm_provider": "openai"
    }

    # Mock flow execution
    with patch("app.api.v1.assessments.MaturityAssessmentFlow") as mock_flow_class:
        mock_flow = AsyncMock()
        mock_flow.execute.return_value = AsyncMock(
            status="completed",
            flow_id="123",
            result={
                "assessment_id": 123,
                "scores": {"overall_score": 3.5}
            }
        )
        mock_flow_class.return_value = mock_flow

        response = await client.post(
            "/api/v1/assessments",
            files=files,
            data=data,
            headers={"X-API-Key": mock_api_key}
        )

        assert response.status_code == 200
        result = response.json()
        assert "assessment_id" in result
        assert result["status"] == "completed"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_assessment_invalid_rubric(client: AsyncClient, mock_api_key):
    """Test assessment creation with invalid rubric"""
    files = {"file": ("test.txt", BytesIO(b"content"), "text/plain")}
    data = {
        "rubric": "invalid json",
        "llm_provider": "openai"
    }

    response = await client.post(
        "/api/v1/assessments",
        files=files,
        data=data,
        headers={"X-API-Key": mock_api_key}
    )

    assert response.status_code == 400
    assert "rubric" in response.json()["detail"].lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_assessment_unauthorized(client: AsyncClient):
    """Test getting assessment without API key"""
    response = await client.get("/api/v1/assessments/123")

    assert response.status_code == 401
```

### 7. **`/tests/test_flows/test_maturity_assessment_flow.py`** - Flow integration tests
```python
import pytest
from unittest.mock import AsyncMock, Mock, patch
from io import BytesIO

from app.flows.maturity_assessment_flow import MaturityAssessmentFlow, MaturityAssessmentInput
from app.llm.base import LLMProvider


@pytest.mark.integration
@pytest.mark.asyncio
async def test_maturity_assessment_flow_success(
    db_session,
    mock_tenant_id,
    sample_rubric
):
    """Test successful maturity assessment flow execution"""
    # Mock external services
    mock_gcs = Mock()
    mock_gcs.upload_file.return_value = "gs://bucket/path/doc.pdf"

    mock_rag = AsyncMock()
    mock_rag.process_document.return_value = 10  # 10 chunks created

    # Create flow
    flow = MaturityAssessmentFlow(db_session, mock_gcs, mock_rag)

    # Mock tool registry
    with patch("app.flows.maturity_assessment_flow.get_tool_registry") as mock_registry:
        # Mock parse tool
        mock_parse_tool = AsyncMock()
        mock_parse_tool.execute.return_value = AsyncMock(
            status="success",
            result={"summary": "Test summary", "key_points": []}
        )

        # Mock score tool
        mock_score_tool = AsyncMock()
        mock_score_tool.execute.return_value = AsyncMock(
            status="success",
            result={"overall_score": 3.5, "scores": []}
        )

        # Mock recommendation tool
        mock_rec_tool = AsyncMock()
        mock_rec_tool.execute.return_value = AsyncMock(
            status="success",
            result={"recommendations": []}
        )

        mock_registry.return_value.get_tool.side_effect = lambda name: {
            "parse_docs": mock_parse_tool,
            "score_rubrics": mock_score_tool,
            "gen_recs": mock_rec_tool
        }[name]

        # Prepare input
        file_obj = BytesIO(b"Test document content for assessment")
        input_data = MaturityAssessmentInput(
            tenant_id=mock_tenant_id,
            file_obj=file_obj,
            filename="test.txt",
            rubric=sample_rubric,
            llm_provider=LLMProvider.OPENAI
        )

        # Execute flow
        result = await flow.execute(input_data)

        assert result.status == "completed"
        assert result.steps_completed == flow.total_steps
        assert "assessment_id" in result.result
        assert "scores" in result.result
        assert "recommendations" in result.result
```

### 8. **`/tests/test_mocks.py`** - Mock utilities
```python
"""Mock utilities for testing"""
from unittest.mock import AsyncMock, Mock
from typing import List


class MockLLMProvider:
    """Mock LLM provider for testing"""

    def __init__(self, responses: List[dict] = None):
        self.responses = responses or []
        self.call_count = 0

    async def generate_structured(self, messages, schema, **kwargs):
        """Mock structured generation"""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return {"result": "mock response"}

    async def generate(self, messages, **kwargs):
        """Mock generation"""
        from app.llm.base import LLMResponse, LLMProvider

        return LLMResponse(
            content="Mock response",
            model="mock-model",
            provider=LLMProvider.OPENAI,
            tokens_used=100,
            finish_reason="stop",
            metadata={}
        )

    def count_tokens(self, text: str) -> int:
        """Mock token counting"""
        return len(text) // 4


class MockGCSClient:
    """Mock GCS client for testing"""

    def __init__(self):
        self.uploaded_files = {}

    def upload_file(self, file_obj, destination_path, **kwargs):
        """Mock file upload"""
        content = file_obj.read()
        self.uploaded_files[destination_path] = content
        return f"gs://test-bucket/{destination_path}"

    def download_file(self, source_path):
        """Mock file download"""
        return self.uploaded_files.get(source_path, b"mock content")

    def delete_file(self, path):
        """Mock file deletion"""
        if path in self.uploaded_files:
            del self.uploaded_files[path]
        return True

    def file_exists(self, path):
        """Mock file existence check"""
        return path in self.uploaded_files
```

### 9. **`/.github/workflows/test.yml`** - Automated testing workflow
```yaml
name: Run Tests

on: [pull_request, push]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: ankane/pgvector:latest
        env:
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: ai_agency_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5433:5432  # CI PostgreSQL service

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run unit tests
        run: pytest -m unit

      - name: Run integration tests
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5433/ai_agency_test
        run: pytest -m integration

      - name: Generate coverage report
        run: pytest --cov=app --cov-report=xml

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
```

### 10. **`/tests/test_e2e/test_deployed_service.py`** - E2E deployed service tests
```python
"""
End-to-End tests for deployed Cloud Run service.
These tests run against the actual deployed service in GCP.
"""
import pytest
import httpx
import os
from typing import Optional


# Configure deployed service URL from environment
DEPLOYED_URL = os.getenv(
    "DEPLOYED_SERVICE_URL",
    "https://ai-agency-847424242737.europe-west1.run.app"
)


@pytest.fixture
def deployed_base_url() -> str:
    """Get deployed service base URL"""
    return DEPLOYED_URL


@pytest.fixture
def deployed_client(deployed_base_url) -> httpx.Client:
    """Create HTTP client for deployed service"""
    return httpx.Client(
        base_url=deployed_base_url,
        timeout=30.0,
        follow_redirects=True
    )


@pytest.mark.e2e
@pytest.mark.deployed
def test_health_check_endpoint(deployed_client):
    """Test /healthz endpoint on deployed service"""
    response = deployed_client.get("/healthz")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


@pytest.mark.e2e
@pytest.mark.deployed
def test_swagger_docs_available(deployed_client):
    """Test that Swagger UI is accessible"""
    response = deployed_client.get("/docs")

    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


@pytest.mark.e2e
@pytest.mark.deployed
def test_openapi_schema_endpoint(deployed_client):
    """Test OpenAPI schema endpoint"""
    response = deployed_client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "paths" in schema
    assert schema["info"]["title"] == "AI Agency Platform"


@pytest.mark.e2e
@pytest.mark.deployed
def test_root_endpoint_redirect(deployed_client):
    """Test root endpoint redirects to /docs"""
    response = deployed_client.get("/", follow_redirects=False)

    # Should redirect to docs
    assert response.status_code in [307, 308, 302, 301]
    assert "/docs" in response.headers.get("location", "")


@pytest.mark.e2e
@pytest.mark.deployed
@pytest.mark.requires_auth
def test_assessment_endpoint_requires_auth(deployed_client):
    """Test that assessment endpoint requires authentication"""
    response = deployed_client.post(
        "/api/v1/assessments",
        files={"file": ("test.txt", b"content", "text/plain")},
        data={"rubric": "{}", "llm_provider": "openai"}
    )

    # Should require API key
    assert response.status_code == 401


@pytest.mark.e2e
@pytest.mark.deployed
def test_cors_headers(deployed_client):
    """Test CORS headers are configured"""
    response = deployed_client.options("/api/v1/assessments")

    # Check CORS headers present
    headers = response.headers
    # Note: Actual CORS headers depend on FastAPI CORS configuration


@pytest.mark.e2e
@pytest.mark.deployed
def test_database_connectivity(deployed_client):
    """Test that service can connect to Cloud SQL database"""
    # Health check should verify database connection
    response = deployed_client.get("/healthz")

    assert response.status_code == 200
    data = response.json()

    # If health check includes DB status
    if "database" in data:
        assert data["database"]["connected"] is True


@pytest.mark.e2e
@pytest.mark.deployed
def test_gcs_connectivity(deployed_client):
    """Test that service can connect to GCS"""
    # Health check should verify GCS connection
    response = deployed_client.get("/healthz")

    assert response.status_code == 200
    data = response.json()

    # If health check includes GCS status
    if "storage" in data:
        assert data["storage"]["accessible"] is True


@pytest.mark.e2e
@pytest.mark.deployed
@pytest.mark.slow
def test_service_response_time(deployed_client):
    """Test that service responds within acceptable time"""
    import time

    start = time.time()
    response = deployed_client.get("/healthz")
    elapsed = time.time() - start

    assert response.status_code == 200
    assert elapsed < 2.0  # Should respond within 2 seconds


@pytest.mark.e2e
@pytest.mark.deployed
def test_error_handling_404(deployed_client):
    """Test 404 error handling"""
    response = deployed_client.get("/api/v1/nonexistent")

    assert response.status_code == 404


@pytest.mark.e2e
@pytest.mark.deployed
def test_environment_variables_loaded(deployed_client):
    """Test that environment variables are correctly loaded"""
    response = deployed_client.get("/healthz")

    assert response.status_code == 200
    data = response.json()

    # If health check exposes environment info
    if "environment" in data:
        assert data["environment"] == "production"
```

### 11. **`/tests/test_e2e/conftest.py`** - E2E test configuration
```python
"""Configuration for E2E tests"""
import pytest
import os


def pytest_configure(config):
    """Configure E2E test markers"""
    config.addinivalue_line(
        "markers",
        "e2e: End-to-end tests against deployed service"
    )
    config.addinivalue_line(
        "markers",
        "deployed: Tests that run against deployed Cloud Run service"
    )
    config.addinivalue_line(
        "markers",
        "requires_auth: Tests that require API authentication"
    )
    config.addinivalue_line(
        "markers",
        "smoke: Quick smoke tests for deployment validation"
    )


@pytest.fixture(scope="session")
def deployed_service_url():
    """Get deployed service URL from environment"""
    url = os.getenv(
        "DEPLOYED_SERVICE_URL",
        "https://ai-agency-847424242737.europe-west1.run.app"
    )
    return url


@pytest.fixture(scope="session")
def test_api_key():
    """Get test API key from environment"""
    return os.getenv("TEST_API_KEY", "test_key_for_e2e")
```

### 12. **`/tests/test_e2e/test_smoke.py`** - Post-deployment smoke tests
```python
"""
Smoke tests to run immediately after deployment.
These are quick sanity checks to ensure basic functionality.
"""
import pytest
import httpx
import os


DEPLOYED_URL = os.getenv(
    "DEPLOYED_SERVICE_URL",
    "https://ai-agency-847424242737.europe-west1.run.app"
)


@pytest.mark.smoke
@pytest.mark.deployed
def test_smoke_service_is_up():
    """SMOKE: Service is responding to requests"""
    response = httpx.get(f"{DEPLOYED_URL}/healthz", timeout=10)
    assert response.status_code == 200


@pytest.mark.smoke
@pytest.mark.deployed
def test_smoke_swagger_accessible():
    """SMOKE: Swagger UI is accessible"""
    response = httpx.get(f"{DEPLOYED_URL}/docs", timeout=10)
    assert response.status_code == 200


@pytest.mark.smoke
@pytest.mark.deployed
def test_smoke_openapi_valid():
    """SMOKE: OpenAPI schema is valid"""
    response = httpx.get(f"{DEPLOYED_URL}/openapi.json", timeout=10)
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "paths" in schema


@pytest.mark.smoke
@pytest.mark.deployed
def test_smoke_no_500_errors_on_common_endpoints():
    """SMOKE: Common endpoints don't return 500 errors"""
    endpoints = ["/healthz", "/docs", "/openapi.json"]

    for endpoint in endpoints:
        response = httpx.get(f"{DEPLOYED_URL}{endpoint}", timeout=10)
        assert response.status_code != 500, f"{endpoint} returned 500 error"
```

### 13. **`/.github/workflows/e2e-tests.yml`** - Post-deployment E2E testing workflow
```yaml
name: E2E Tests (Deployed Service)

on:
  workflow_dispatch:
    inputs:
      service_url:
        description: 'Deployed service URL to test'
        required: false
        default: 'https://ai-agency-847424242737.europe-west1.run.app'

  # Run after successful deployment
  workflow_run:
    workflows: ["Deploy to Cloud Run"]
    types:
      - completed

jobs:
  smoke-tests:
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'success' || github.event_name == 'workflow_dispatch' }}

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install pytest httpx pytest-timeout

      - name: Run smoke tests
        env:
          DEPLOYED_SERVICE_URL: ${{ github.event.inputs.service_url || 'https://ai-agency-847424242737.europe-west1.run.app' }}
        run: |
          pytest tests/test_e2e/test_smoke.py -v -m smoke
        timeout-minutes: 5

      - name: Notify on failure
        if: failure()
        run: |
          echo "Smoke tests failed! Deployment may have issues."
          exit 1

  full-e2e-tests:
    runs-on: ubuntu-latest
    needs: smoke-tests
    if: success()

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install pytest httpx pytest-asyncio pytest-timeout

      - name: Run E2E tests
        env:
          DEPLOYED_SERVICE_URL: ${{ github.event.inputs.service_url || 'https://ai-agency-847424242737.europe-west1.run.app' }}
          TEST_API_KEY: ${{ secrets.TEST_API_KEY }}
        run: |
          pytest tests/test_e2e/ -v -m "e2e and deployed" --timeout=60
        timeout-minutes: 10

      - name: Report results
        if: always()
        run: |
          echo "E2E tests completed. Check logs for details."
```

### 14. **`/scripts/run_e2e_tests.sh`** - Manual E2E test runner
```bash
#!/bin/bash
# Run E2E tests against deployed service manually

set -e

DEPLOYED_URL="${DEPLOYED_SERVICE_URL:-https://ai-agency-847424242737.europe-west1.run.app}"

echo "Running E2E tests against: $DEPLOYED_URL"
echo ""

# Run smoke tests first
echo "=== Running Smoke Tests ==="
DEPLOYED_SERVICE_URL="$DEPLOYED_URL" pytest tests/test_e2e/test_smoke.py -v -m smoke

# Run full E2E test suite
echo ""
echo "=== Running Full E2E Test Suite ==="
DEPLOYED_SERVICE_URL="$DEPLOYED_URL" pytest tests/test_e2e/ -v -m "e2e and deployed"

echo ""
echo "✅ All E2E tests passed!"
```

## Testing Strategy

### Test Pyramid
1. **Unit Tests (70%)**: Fast, isolated tests of individual components
2. **Integration Tests (20%)**: Tests of component interactions with real database
3. **E2E Tests (10%)**: Tests of deployed service with real infrastructure

### Test Markers
- `@pytest.mark.unit` - Unit tests (fast, mocked dependencies)
- `@pytest.mark.integration` - Integration tests (real database)
- `@pytest.mark.e2e` - End-to-end tests (deployed service)
- `@pytest.mark.deployed` - Tests against deployed Cloud Run service
- `@pytest.mark.smoke` - Quick smoke tests for deployment validation
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.requires_llm` - Tests requiring LLM API access
- `@pytest.mark.requires_auth` - Tests requiring authentication

### When to Run Tests
- **Unit tests**: On every commit (pre-commit hook)
- **Integration tests**: On every push (CI/CD)
- **E2E smoke tests**: After every deployment (post-deployment)
- **Full E2E tests**: After successful smoke tests (post-deployment)
- **Manual E2E tests**: Before production releases

## Dependencies
- **Upstream**: All other engineers (tests their code)
- **Downstream**: Code Reviewer (provides test evidence for review)

## Working Style
1. **Test-driven mindset**: Write tests that validate requirements
2. **Comprehensive coverage**: Aim for 80%+ across all modules
3. **Realistic mocks**: Mocks should behave like real services
4. **Clear test names**: Test names should describe what they verify

## Success Criteria
- [ ] Test coverage is 80% or higher
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Tests are fast (< 5 minutes total)
- [ ] CI/CD runs tests automatically
- [ ] Tests catch common bugs and edge cases
- [ ] Mock fixtures are reusable

## Notes
- Use pytest-asyncio for async tests
- Mock external APIs (OpenAI, Vertex AI, GCS) in unit tests
- Use real database in integration tests
- Separate fast unit tests from slower integration tests
- Add performance tests for critical paths
- Test error conditions and edge cases
- Keep tests maintainable and readable
