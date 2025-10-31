# Coding Standards

This document outlines the coding standards and best practices for the AI Agency codebase. Follow these patterns when building new features.

---

## Table of Contents

1. [Decorators](#decorators)
2. [Base Classes](#base-classes)
3. [Exception Handling](#exception-handling)
4. [Type Hints](#type-hints)
5. [Async Patterns](#async-patterns)
6. [Rate Limiting](#rate-limiting)
7. [Database Patterns](#database-patterns)
8. [Testing Patterns](#testing-patterns)
9. [Code Organization](#code-organization)

---

## 1. Decorators

We provide production-ready decorators in `app/core/decorators.py`. Use these instead of writing custom implementations.

### @retry

**Use when:** Making external API calls, database operations, or any operation that might fail transiently.

**Example:**
```python
from app.core.decorators import retry
from app.core.exceptions import LLMError

@retry(
    max_attempts=3,
    backoff_type="exponential",
    min_wait=1.0,
    max_wait=10.0,
    exceptions=(httpx.HTTPError,)
)
async def call_external_api(self):
    response = await client.post(url, json=payload)
    return response.json()
```

**Parameters:**
- `max_attempts`: Number of retry attempts (default: 3)
- `backoff_type`: "exponential" or "fixed" (default: "exponential")
- `min_wait`: Minimum wait in seconds (default: 1.0)
- `max_wait`: Maximum wait in seconds (default: 60.0)
- `exceptions`: Tuple of exceptions to retry on (default: (Exception,))

**When NOT to use:**
- User input validation errors (don't retry user mistakes)
- Authentication failures (wrong credentials won't fix themselves)
- 404 errors (resource won't suddenly appear)

### @timeout

**Use when:** Operations that might hang indefinitely (API calls, database queries).

**Example:**
```python
from app.core.decorators import timeout

@timeout(seconds=30.0)
async def long_running_operation(self):
    # This will raise TimeoutError if it takes > 30 seconds
    result = await slow_api_call()
    return result
```

**Parameters:**
- `seconds`: Maximum execution time (float)

**Best practices:**
- LLM calls: 60-120 seconds
- Database queries: 10-30 seconds
- HTTP requests: 30 seconds
- File operations: 10 seconds

### @log_execution

**Use when:** You want automatic logging of function entry, exit, duration, and errors.

**Example:**
```python
from app.core.decorators import log_execution

@log_execution
async def process_data(self, data: dict):
    # Automatically logs:
    # - Function entry with args
    # - Execution duration
    # - Function exit
    # - Any exceptions raised
    result = await transform(data)
    return result
```

**Use for:**
- All public API methods
- Flow orchestration methods
- Critical business logic

**Skip for:**
- Private helper methods
- Simple getters/setters
- Functions called in tight loops

### @measure_execution

**Use when:** You need detailed performance metrics without logs.

**Example:**
```python
from app.core.decorators import measure_execution

@measure_execution
async def expensive_computation(self):
    # Returns (result, duration_seconds)
    result = await heavy_computation()
    return result

result, duration = await expensive_computation()
logger.info(f"Computation took {duration:.2f}s")
```

**Use for:**
- Performance-critical code
- Operations you want to profile
- Code that needs timing metrics

### Combining Decorators

**Order matters!** Apply decorators from innermost (closest to function) to outermost:

```python
@log_execution          # 4. Outermost: log everything
@timeout(seconds=120.0) # 3. Apply timeout
@retry(max_attempts=3)  # 2. Retry on failures
async def call_llm(self, prompt: str):  # 1. Innermost: actual function
    return await llm.complete(prompt)
```

**Recommended order:**
1. Function definition
2. `@retry` (if needed)
3. `@timeout` (if needed)
4. `@log_execution` (if needed)

---

## 2. Base Classes

### BaseFlow

**Use when:** Building any agent flow or orchestration logic.

**Location:** `app/core/base.py`

**Example:**
```python
from app.core.base import BaseFlow
from app.core.decorators import log_execution, timeout

class YourAgentFlow(BaseFlow):
    """Your agent description."""

    def __init__(self):
        super().__init__(name="your_agent")
        # Initialize your dependencies
        self.llm = LLMFactory.create()
        self.repo = ConversationRepository(session)

    @log_execution
    @timeout(seconds=120.0)
    async def run(
        self,
        user_message: str,
        tenant_id: str = "default",
    ) -> dict:
        """Execute the flow."""
        # 1. Create/load conversation
        conversation_id = self.repo.create_conversation(
            tenant_id=tenant_id,
            flow_type="your_flow_type"
        )

        # 2. Save user message
        self.repo.save_message(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            flow_type="your_flow_type",
            role="user",
            content=user_message,
        )

        # 3. Get LLM response
        response = await self.llm.complete(messages, tenant_id=tenant_id)

        # 4. Save assistant message
        self.repo.save_message(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            flow_type="your_flow_type",
            role="assistant",
            content=response,
        )

        return {
            "response": response,
            "conversation_id": conversation_id,
        }
```

**Must implement:**
- `run()` method (main orchestration logic)

**Inherit:**
- `name` property (set in `__init__`)
- Logging capabilities

### BaseTool

**Use when:** Building tools that LLMs can call (function calling).

**Location:** `app/core/base.py`

**Example:**
```python
from app.core.base import BaseTool
from app.core.types import ToolMetadata

class CalculatorTool(BaseTool):
    """Perform calculations."""

    def __init__(self):
        super().__init__(name="calculator")

    def metadata(self) -> ToolMetadata:
        """Return OpenAI function calling schema."""
        return {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform arithmetic calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to evaluate",
                        },
                    },
                    "required": ["expression"],
                },
            },
        }

    async def execute(self, **kwargs) -> dict:
        """Execute the tool."""
        expression = kwargs["expression"]
        result = eval(expression)  # Use a safe evaluator in production!
        return {"result": result}
```

**Must implement:**
- `metadata()` - Returns OpenAI function calling schema
- `execute(**kwargs)` - Executes the tool

---

## 3. Exception Handling

### Custom Exceptions

**Location:** `app/core/exceptions.py`

**Use these exceptions:**

```python
from app.core.exceptions import (
    LLMError,           # LLM API failures
    RateLimitError,     # Rate limit exceeded
    TimeoutError,       # Operation timeout
    ValidationError,    # Input validation failed
    DatabaseError,      # Database operations failed
    ConfigurationError, # Configuration issues
)
```

### When to use each exception

**LLMError:**
```python
# Use for LLM API failures
try:
    response = await client.post(url, json=payload)
    response.raise_for_status()
except httpx.HTTPError as e:
    raise LLMError(
        f"OpenAI API call failed: {str(e)}",
        details={
            "provider": "openai",
            "model": self.model_name,
            "error_type": type(e).__name__,
        },
        original_error=e,
    ) from e
```

**RateLimitError:**
```python
# Use when rate limits are exceeded
if response.status_code == 429:
    raise RateLimitError(
        "OpenAI API rate limit exceeded",
        details={"status_code": 429},
    )
```

**TimeoutError:**
```python
# Use when operations timeout
try:
    result = await asyncio.wait_for(operation(), timeout=30.0)
except asyncio.TimeoutError as e:
    raise TimeoutError(
        f"Operation timed out after 30s",
        details={"operation": "llm_completion"},
    ) from e
```

**ValidationError:**
```python
# Use for input validation
if not api_key:
    raise ValidationError(
        "API key is required",
        details={"field": "api_key"},
    )
```

### Exception patterns

**Always provide context:**
```python
# Bad
raise LLMError("API call failed")

# Good
raise LLMError(
    f"OpenAI API call failed: {str(e)}",
    details={
        "provider": "openai",
        "model": self.model_name,
        "error_type": type(e).__name__,
    },
    original_error=e,
)
```

**Always chain exceptions:**
```python
# Bad
except httpx.HTTPError as e:
    raise LLMError("Failed")

# Good
except httpx.HTTPError as e:
    raise LLMError("Failed") from e
```

---

## 4. Type Hints

### Always use type hints

```python
# Bad
def process_data(data):
    return data["result"]

# Good
def process_data(data: dict[str, Any]) -> str:
    return data["result"]
```

### Modern Python 3.11+ syntax

```python
# Use | for Optional
def get_user(user_id: str) -> User | None:
    return user_repo.get(user_id)

# Use lowercase generic types
def process_items(items: list[str]) -> dict[str, int]:
    return {item: len(item) for item in items}
```

### TypedDict for structured data

```python
from typing import TypedDict

class LLMResponse(TypedDict):
    content: str
    model: str
    tokens_used: int
    finish_reason: str

async def complete(messages: list) -> LLMResponse:
    return {
        "content": "Hello",
        "model": "gpt-4",
        "tokens_used": 10,
        "finish_reason": "stop",
    }
```

---

## 5. Async Patterns

### Always use async for I/O

```python
# Bad
def call_api():
    response = requests.post(url, json=data)
    return response.json()

# Good
async def call_api():
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data)
        return response.json()
```

### Proper context manager usage

```python
# Good
async with httpx.AsyncClient() as client:
    response = await client.post(url, json=payload)
    return response.json()
```

### Concurrent operations

```python
# Run multiple operations concurrently
results = await asyncio.gather(
    call_api_1(),
    call_api_2(),
    call_api_3(),
)
```

---

## 6. Rate Limiting

### Always use rate limiter for LLM calls

**Location:** `app/core/rate_limiter.py`

```python
from app.core.rate_limiter import TokenBucketRateLimiter

class OpenAIAdapter:
    def __init__(self):
        self.rate_limiter = TokenBucketRateLimiter(
            tokens_per_minute=90000,   # OpenAI TPM limit
            tokens_per_hour=5000000,   # OpenAI TPH limit
        )

    async def complete(self, messages, tenant_id: str = "default"):
        # Estimate tokens
        estimated_tokens = self._estimate_tokens(messages)

        # Wait if rate limited
        await self.rate_limiter.wait_if_needed(tenant_id, estimated_tokens)

        # Make API call
        response = await self._call_api(messages)
        return response
```

**Key points:**
- Use per-tenant rate limiting
- Estimate tokens before calling API
- Use `wait_if_needed()` for automatic waiting
- Configure limits per provider

---

## 7. Database Patterns

### Connection Pooling

The application uses **psycopg ConnectionPool** for all database operations. This is handled automatically by `app/db/base.py` - you don't need to manage connections directly.

```python
# Good - use Session from get_session()
from app.db.base import get_session
from sqlmodel import Session

with Session(get_session()) as session:
    # Connection automatically obtained from pool
    repo = ConversationRepository(session)
    result = repo.create_conversation(tenant_id, flow_type)
    # Connection automatically returned to pool
```

**Key points:**
- Two separate pools: main application (20 connections) and LangGraph checkpointer (10 connections)
- Connection pooling is automatic - just use `get_session()`
- Never create connections directly with `psycopg.connect()`
- Cloud SQL Proxy required for local development
- See `docs/ARCHITECTURE.md` for detailed connection pool architecture

### Always use repositories

```python
# Bad - direct database access
conversation = session.query(Conversation).filter_by(id=conv_id).first()

# Good - use repository
conversation = repo.get_conversation(conv_id)
```

### Always pass flow_type

```python
# Bad - missing flow_type
repo.create_conversation(tenant_id="user123")

# Good - include flow_type
repo.create_conversation(
    tenant_id="user123",
    flow_type="weather"
)
```

### Use context managers for sessions

```python
from app.db.base import get_session

async with get_session() as session:
    repo = ConversationRepository(session)
    result = repo.create_conversation(tenant_id, flow_type)
```

---

## 8. Testing Patterns

### Test organization

```
tests/
├── test_adapters/          # Unit tests for adapters
│   ├── test_llm_openai.py
│   └── test_llm_mistral.py
├── integration/            # Integration tests
│   └── test_weather_agent.py
└── conftest.py             # Shared fixtures
```

### Use pytest-mock for mocking

```python
@pytest.mark.asyncio
async def test_llm_call(mocker):
    # Mock the API call
    mock_post = mocker.patch("httpx.AsyncClient.post")
    mock_post.return_value.json.return_value = {
        "choices": [{"message": {"content": "Hello"}}]
    }

    # Test
    adapter = OpenAIAdapter(api_key="test")
    result = await adapter.complete(messages=[])

    assert result == "Hello"
    mock_post.assert_called_once()
```

### Async tests

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await my_async_function()
    assert result == expected
```

---

## 9. Code Organization

### Project structure

```
app/
├── core/                   # Base classes, decorators, exceptions
│   ├── base.py            # BaseFlow, BaseTool
│   ├── decorators.py      # @retry, @timeout, @log_execution
│   ├── exceptions.py      # Custom exceptions
│   ├── rate_limiter.py    # Token bucket rate limiter
│   └── types.py           # TypedDicts and type aliases
│
├── adapters/               # LLM provider adapters
│   ├── llm_factory.py     # Provider factory
│   ├── llm_openai.py      # OpenAI adapter
│   ├── llm_vertex.py      # Vertex AI adapter
│   └── llm_mistral.py     # Mistral adapter
│
├── flows/                  # Agent flows
│   └── agents/
│       └── weather_agent.py
│
├── db/                     # Database layer
│   ├── models.py          # SQLModel tables
│   ├── base.py            # Session management
│   └── repositories/      # Data access layer
│       └── conversation_repository.py
│
└── tools/                  # LLM-callable tools
    └── weather/
        ├── client.py      # Weather API client
        └── v1.py          # Weather tool
```

### File naming

- Use snake_case for files: `weather_agent.py`
- Use PascalCase for classes: `WeatherAgentFlow`
- Use snake_case for functions: `get_weather_data()`
- Use UPPER_CASE for constants: `MAX_RETRIES = 3`

### Import organization

```python
# Standard library
import asyncio
import logging
from typing import Any

# Third-party
import httpx
from fastapi import FastAPI

# Local
from app.core.base import BaseFlow
from app.core.decorators import log_execution, retry, timeout
from app.core.exceptions import LLMError
```

---

## Summary Checklist

When building a new feature, ensure:

- [ ] Use `@retry` for external API calls
- [ ] Use `@timeout` for long-running operations
- [ ] Use `@log_execution` for important methods
- [ ] Extend `BaseFlow` for agent flows
- [ ] Extend `BaseTool` for LLM-callable tools
- [ ] Use custom exceptions with context
- [ ] Add type hints to all functions
- [ ] Use async/await for I/O operations
- [ ] Use rate limiter for LLM calls
- [ ] Use repositories for database access
- [ ] Always pass `flow_type` to repositories
- [ ] Write tests with pytest-mock
- [ ] Follow project structure conventions
- [ ] Use snake_case for files and functions
- [ ] Use PascalCase for classes
