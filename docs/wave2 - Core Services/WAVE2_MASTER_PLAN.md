# Wave 2 Master Plan - Weather Agent Foundation

**Goal:** Build a fully working weather agent that demonstrates the complete platform capabilities and serves as the template for all future agents.

**Status:** Planning → Implementation
**Target:** Production-ready example agent with multi-LLM support, flows, and loops

---

## Overview

Build a simple but complete agent that:
- ✅ Answers weather questions using external API
- ✅ Works locally and on GCP
- ✅ Uses the database for conversation history
- ✅ Supports multiple LLM providers (OpenAI, Gemini, Mistral)
- ✅ Demonstrates flow orchestration
- ✅ Implements basic loop/retry logic
- ✅ Has comprehensive tests (80%+ coverage)
- ✅ Serves as template for real use case agents

---

## Success Criteria

### Functional Requirements
- [ ] User asks "What's the weather in Paris?" → Agent calls weather API → Returns answer
- [ ] Conversation history stored in database
- [ ] Switch LLM provider via environment variable (no code changes)
- [ ] Works identically on local and GCP
- [ ] Agent can retry failed API calls (loop logic)
- [ ] Agent can handle multi-turn conversations (flow logic)

### Technical Requirements
- [ ] OpenAI models: gpt-4, gpt-3.5-turbo, gpt-4-turbo
- [ ] Gemini models via Vertex AI: gemini-pro, gemini-1.5-pro
- [ ] Mistral models: mistral-large, mistral-medium
- [ ] Weather API integration (e.g., OpenWeatherMap)
- [ ] Database schema for conversations and turns
- [ ] Test coverage: 80%+ for all new code
- [ ] Documentation with example usage

---

## Architecture

```
User Question
    ↓
API Endpoint (/weather/chat)
    ↓
Weather Agent Flow
    ↓
┌─────────────────────┐
│  1. Parse Intent    │ ← LLM (any provider)
└─────────────────────┘
    ↓
┌─────────────────────┐
│  2. Call Weather    │ ← External API
│     API (with retry)│
└─────────────────────┘
    ↓
┌─────────────────────┐
│  3. Format Response │ ← LLM (any provider)
└─────────────────────┘
    ↓
┌─────────────────────┐
│  4. Save to DB      │ ← PostgreSQL
└─────────────────────┘
    ↓
Return to User
```

---

## Implementation Plan

### Phase 1: Multi-LLM Foundation (Days 1-2)

**Goal:** Make LLM adapters work with real API calls

#### 1.1 LLM Adapter Interface
```python
# app/adapters/base_llm.py
class BaseLLMAdapter(ABC):
    @abstractmethod
    async def generate_completion(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Generate completion from messages."""
        pass

    @abstractmethod
    async def available_models(self) -> list[str]:
        """List available models for this provider."""
        pass
```

#### 1.2 OpenAI Adapter
```python
# app/adapters/llm_openai.py
class OpenAIAdapter(BaseLLMAdapter):
    """OpenAI adapter with latest models (as of October 2025)."""

    supported_models = [
        # GPT-4.1 Series (Latest - October 2025)
        "gpt-4.1",           # 1M context, best performance
        "gpt-4.1-mini",      # Fast, cost-effective, beats gpt-4o
        "gpt-4.1-nano",      # Fastest, cheapest

        # GPT-4o Series (Still available)
        "gpt-4o",            # Multimodal, audio support
        "gpt-4o-mini",       # Being replaced by 4.1-mini

        # O-Series Reasoning Models
        "o4-mini",           # Fast reasoning for math/coding

        # Legacy (still available)
        "gpt-4-turbo",
        "gpt-3.5-turbo"
    ]

    # Default model per use case
    DEFAULT_MODEL = "gpt-4.1-mini"  # Best price/performance
    REASONING_MODEL = "o4-mini"     # For complex reasoning
    FAST_MODEL = "gpt-4.1-nano"     # For simple tasks

    async def generate_completion(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        # Real OpenAI API call with retry logic
        pass
```

#### 1.3 Gemini/Vertex AI Adapter
```python
# app/adapters/llm_vertex.py
class VertexAIAdapter(BaseLLMAdapter):
    """Google Vertex AI adapter with latest Gemini models (as of October 2025)."""

    supported_models = [
        # Gemini 2.5 Series (Latest - October 2025)
        "gemini-2.5-pro",              # Best for complex tasks
        "gemini-2.5-flash",            # Best price/performance
        "gemini-2.5-flash-lite",       # Most cost-effective
        "gemini-2.5-computer-use",     # Preview: tool use

        # Gemini 2.0 Series
        "gemini-2.0-flash",            # Multimodal, latest features
        "gemini-2.0-flash-lite",       # Low latency

        # Legacy (still available)
        "gemini-1.5-pro",
        "gemini-1.5-flash"
    ]

    # Default model per use case
    DEFAULT_MODEL = "gemini-2.5-flash"       # Best price/performance
    PRO_MODEL = "gemini-2.5-pro"             # For complex tasks
    FAST_MODEL = "gemini-2.5-flash-lite"     # For simple tasks

    async def generate_completion(...) -> str:
        # Real Vertex AI API call
        pass
```

#### 1.4 Mistral Adapter
```python
# app/adapters/llm_mistral.py
class MistralAdapter(BaseLLMAdapter):
    """Mistral AI adapter with latest models (as of October 2025)."""

    supported_models = [
        # Latest Models (2025)
        "mistral-medium-3",           # Frontier performance, cost-effective
        "magistral",                  # Reasoning specialist (June 2025)
        "codestral-2",                # Code generation specialist
        "mistral-small-3.1",          # Multimodal, 128K context

        # Legacy
        "mistral-large-latest",
        "mistral-small-latest"
    ]

    # Default model per use case
    DEFAULT_MODEL = "mistral-medium-3"    # Best price/performance
    REASONING_MODEL = "magistral"         # For complex reasoning
    CODE_MODEL = "codestral-2"            # For code generation

    async def generate_completion(...) -> str:
        # Real Mistral API call
        pass
```

#### 1.5 LLM Factory (Enhanced)
```python
# app/adapters/llm_factory.py
class LLMFactory:
    @staticmethod
    async def create(
        provider: str | None = None,
        model: str | None = None
    ) -> BaseLLMAdapter:
        """
        Create LLM adapter based on config.

        provider: "openai" | "vertex" | "mistral"
        model: specific model name (optional)

        Reads from env:
        - LLM_PROVIDER (default provider)
        - LLM_MODEL (default model)
        - OPENAI_API_KEY
        - GCP_PROJECT_ID (for Vertex)
        - MISTRAL_API_KEY
        """
        pass
```

#### 1.6 Configuration
```bash
# .env
LLM_PROVIDER=openai  # or "vertex" or "mistral"
LLM_MODEL=gpt-4      # optional, uses provider default if not set

# OpenAI
OPENAI_API_KEY=sk-...

# Vertex AI
GCP_PROJECT_ID=your-project
GCP_REGION=europe-west1

# Mistral
MISTRAL_API_KEY=...
```

#### 1.7 Rate Limiting Implementation
```python
# app/core/rate_limiter.py
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio

class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for LLM API calls.

    Best practices (2025):
    - Track tokens_per_minute, not just requests_per_minute
    - Multi-dimensional limits (user, tenant, model)
    - Sliding window for precision
    - Exponential backoff on 429 errors
    """

    def __init__(
        self,
        tokens_per_minute: int = 90000,    # OpenAI GPT-4.1 default
        tokens_per_hour: int = 5000000,    # Hourly limit
        burst_multiplier: float = 1.5      # Allow short bursts
    ):
        self.tokens_per_minute = tokens_per_minute
        self.tokens_per_hour = tokens_per_hour
        self.burst_size = int(tokens_per_minute * burst_multiplier)

        # Per-tenant tracking
        self.tenant_buckets = defaultdict(lambda: {
            "minute_tokens": self.burst_size,
            "minute_reset": datetime.now(),
            "hour_tokens": tokens_per_hour,
            "hour_reset": datetime.now()
        })

    async def acquire(
        self,
        tenant_id: str,
        estimated_tokens: int
    ) -> tuple[bool, int]:
        """
        Try to acquire tokens for request.

        Returns:
            (allowed, wait_seconds)
        """
        bucket = self.tenant_buckets[tenant_id]
        now = datetime.now()

        # Refill buckets if time windows expired
        if now >= bucket["minute_reset"]:
            bucket["minute_tokens"] = self.burst_size
            bucket["minute_reset"] = now + timedelta(minutes=1)

        if now >= bucket["hour_reset"]:
            bucket["hour_tokens"] = self.tokens_per_hour
            bucket["hour_reset"] = now + timedelta(hours=1)

        # Check if we have enough tokens
        if (bucket["minute_tokens"] >= estimated_tokens and
            bucket["hour_tokens"] >= estimated_tokens):

            bucket["minute_tokens"] -= estimated_tokens
            bucket["hour_tokens"] -= estimated_tokens
            return True, 0

        # Calculate wait time
        wait_seconds = (bucket["minute_reset"] - now).total_seconds()
        return False, max(1, int(wait_seconds))

    async def wait_if_needed(
        self,
        tenant_id: str,
        estimated_tokens: int,
        max_wait: int = 60
    ):
        """
        Wait for rate limit to allow request.

        Raises:
            RateLimitError: If wait time exceeds max_wait
        """
        allowed, wait_time = await self.acquire(tenant_id, estimated_tokens)

        if not allowed:
            if wait_time > max_wait:
                raise RateLimitError(
                    f"Rate limit exceeded. Try again in {wait_time}s"
                )

            logger.info(f"Rate limit reached, waiting {wait_time}s")
            await asyncio.sleep(wait_time)

            # Retry after wait
            allowed, _ = await self.acquire(tenant_id, estimated_tokens)
            if not allowed:
                raise RateLimitError("Rate limit still exceeded after wait")


# app/adapters/base_llm.py (updated)
class BaseLLMAdapter(ABC):
    def __init__(self):
        self.rate_limiter = TokenBucketRateLimiter(
            tokens_per_minute=self.get_rate_limit(),
            tokens_per_hour=self.get_hourly_limit()
        )

    @abstractmethod
    def get_rate_limit(self) -> int:
        """Get tokens per minute for this provider."""
        pass

    @abstractmethod
    def estimate_tokens(self, messages: list[dict]) -> int:
        """Estimate tokens for request."""
        pass

    async def generate_completion(
        self,
        messages: list[dict],
        tenant_id: str = "default",
        **kwargs
    ) -> str:
        # Estimate tokens
        estimated_tokens = self.estimate_tokens(messages)

        # Wait for rate limit
        await self.rate_limiter.wait_if_needed(tenant_id, estimated_tokens)

        # Make API call with retry on 429
        return await self._make_api_call_with_retry(messages, **kwargs)

    @retry(
        max_attempts=3,
        on_exceptions=(RateLimitError, httpx.HTTPStatusError),
        backoff_strategy="exponential"
    )
    async def _make_api_call_with_retry(self, messages, **kwargs):
        """Make API call with exponential backoff on 429."""
        try:
            response = await self._api_call(messages, **kwargs)
            return response
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                # Respect Retry-After header
                retry_after = e.response.headers.get("Retry-After", 60)
                logger.warning(f"429 error, waiting {retry_after}s")
                await asyncio.sleep(int(retry_after))
                raise  # Retry decorator will handle retry
            raise
```

#### Rate Limit Configuration
```python
# app/config.py
class RateLimitSettings(BaseSettings):
    """Rate limit configuration per provider."""

    # OpenAI (as of October 2025)
    openai_tokens_per_minute: int = 90000      # GPT-4.1
    openai_tokens_per_hour: int = 5000000

    # Vertex AI (generous limits)
    vertex_tokens_per_minute: int = 300000     # Gemini 2.5
    vertex_tokens_per_hour: int = 10000000

    # Mistral
    mistral_tokens_per_minute: int = 100000
    mistral_tokens_per_hour: int = 3000000

    # Global settings
    enable_rate_limiting: bool = True
    rate_limit_burst_multiplier: float = 1.5
    max_wait_time: int = 60  # seconds
```

#### Database Schema for Rate Limit Tracking
```python
# app/db/models.py
class APIUsage(SQLModel, table=True):
    """Track API usage for monitoring and billing."""
    __tablename__ = "api_usage"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    tenant_id: str = Field(index=True)
    provider: str  # "openai", "vertex", "mistral"
    model: str
    endpoint: str  # "/weather/chat", etc.

    # Token usage
    input_tokens: int
    output_tokens: int
    total_tokens: int

    # Timing
    latency_ms: int
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    # Cost estimation (computed)
    estimated_cost_usd: float | None = None
```

#### Tests (Phase 1)
- [ ] Test OpenAI adapter with all supported models
- [ ] Test Vertex AI adapter with Gemini models
- [ ] Test Mistral adapter
- [ ] Test factory switching between providers
- [ ] Test retry logic on API failures
- [ ] Test error handling for invalid API keys
- [ ] Test rate limiter: token bucket algorithm
- [ ] Test rate limiter: multi-tenant isolation
- [ ] Test rate limiter: burst capacity
- [ ] Test rate limiter: wait and retry behavior
- [ ] Test rate limiter: exponential backoff on 429
- [ ] Test API usage tracking and logging
- [ ] Mock tests (don't hit real APIs)
- [ ] Integration tests (hit real APIs with test keys)

---

### Phase 2: Weather API Tool (Days 3-4)

**Goal:** Create a reusable tool for weather API calls

#### 2.1 Weather Tool
```python
# app/tools/weather_tool.py
class WeatherTool(BaseTool):
    """
    Tool for fetching weather data.
    Uses OpenWeatherMap API (free tier).
    """

    def __init__(self):
        self.api_key = get_settings().weather_api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"

    async def execute(self, location: str, units: str = "metric") -> dict:
        """
        Fetch weather for location.

        Args:
            location: City name or coordinates
            units: "metric" or "imperial"

        Returns:
            {
                "temperature": 15.5,
                "description": "partly cloudy",
                "humidity": 65,
                "wind_speed": 3.5,
                "location": "Paris, FR"
            }

        Raises:
            ToolError: If API call fails after retries
        """
        pass

    @retry(max_attempts=3, backoff=2.0)
    async def _call_api(self, params: dict) -> dict:
        """API call with retry logic."""
        pass
```

#### 2.2 Tool Registry Update
```python
# app/tools/registry.py
AVAILABLE_TOOLS = {
    "weather": WeatherTool,
    # Future tools:
    # "search": SearchTool,
    # "calculator": CalculatorTool,
}

def get_tool(tool_name: str) -> BaseTool:
    """Get tool instance by name."""
    pass
```

#### Tests (Phase 2)
- [ ] Test weather API calls (mock responses)
- [ ] Test retry logic on failures
- [ ] Test error handling (invalid API key, unknown location)
- [ ] Test different units (metric/imperial)
- [ ] Integration test with real API
- [ ] Test tool registry

---

### Phase 3: Database Schema for Conversations (Day 5)

**Goal:** Store agent conversations in database

#### 3.1 Database Models
```python
# app/db/models.py

class Conversation(SQLModel, table=True):
    """A conversation session with the agent."""
    __tablename__ = "conversations"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    tenant_id: str = Field(index=True)
    user_id: str | None = Field(default=None, index=True)
    agent_type: str = Field(default="weather")  # "weather", "maturity", etc.
    status: str = Field(default="active")  # "active", "completed", "failed"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = Field(default_factory=dict, sa_column=Column(JSON))


class ConversationTurn(SQLModel, table=True):
    """A single turn (user message + agent response) in a conversation."""
    __tablename__ = "conversation_turns"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    conversation_id: UUID = Field(foreign_key="conversations.id", index=True)
    turn_number: int  # 1, 2, 3, ...

    # User input
    user_message: str

    # Agent processing
    llm_provider: str  # "openai", "vertex", "mistral"
    llm_model: str  # "gpt-4", "gemini-pro", etc.
    tools_used: list[str] = Field(default_factory=list, sa_column=Column(JSON))

    # Agent output
    agent_response: str

    # Metadata
    tokens_used: int | None = None
    latency_ms: int | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = Field(default_factory=dict, sa_column=Column(JSON))
```

#### 3.2 Alembic Migration
```bash
alembic revision --autogenerate -m "Add conversation tables for agents"
alembic upgrade head
```

#### 3.3 Repository Pattern
```python
# app/db/repositories/conversation_repository.py
class ConversationRepository:
    async def create_conversation(
        self,
        tenant_id: str,
        user_id: str | None = None,
        agent_type: str = "weather"
    ) -> Conversation:
        """Create new conversation."""
        pass

    async def add_turn(
        self,
        conversation_id: UUID,
        user_message: str,
        agent_response: str,
        llm_provider: str,
        llm_model: str,
        tools_used: list[str],
        tokens_used: int | None = None,
        latency_ms: int | None = None
    ) -> ConversationTurn:
        """Add turn to conversation."""
        pass

    async def get_conversation_history(
        self,
        conversation_id: UUID,
        limit: int = 50
    ) -> list[ConversationTurn]:
        """Get conversation turns."""
        pass
```

#### Tests (Phase 3)
- [ ] Test conversation creation
- [ ] Test adding turns
- [ ] Test retrieving history
- [ ] Test conversation completion
- [ ] Test querying by tenant_id
- [ ] Test querying by user_id

---

### Phase 4: Weather Agent Flow (Days 6-7)

**Goal:** Orchestrate LLM + Tool + Database into a complete flow

#### 4.1 Flow Definition
```python
# app/flows/weather_agent/flow.py
class WeatherAgentFlow(BaseFlow):
    """
    Complete weather agent flow.

    Steps:
    1. Parse user intent with LLM
    2. Call weather API if needed
    3. Generate response with LLM
    4. Save to database
    """

    def __init__(
        self,
        llm_provider: str | None = None,
        llm_model: str | None = None
    ):
        self.llm = await LLMFactory.create(llm_provider, llm_model)
        self.weather_tool = WeatherTool()
        self.conversation_repo = ConversationRepository()

    async def run(
        self,
        user_message: str,
        conversation_id: UUID | None = None,
        tenant_id: str = "default"
    ) -> dict:
        """
        Process user message and return response.

        Returns:
            {
                "conversation_id": "...",
                "response": "The weather in Paris is...",
                "tools_used": ["weather"],
                "llm_provider": "openai",
                "llm_model": "gpt-4"
            }
        """
        start_time = time.time()

        # 1. Create or get conversation
        if not conversation_id:
            conversation = await self.conversation_repo.create_conversation(
                tenant_id=tenant_id,
                agent_type="weather"
            )
            conversation_id = conversation.id

        # 2. Parse intent with LLM
        intent = await self._parse_intent(user_message)

        # 3. Call weather API if needed
        tools_used = []
        weather_data = None
        if intent.get("needs_weather"):
            location = intent.get("location")
            weather_data = await self.weather_tool.execute(location)
            tools_used.append("weather")

        # 4. Generate response with LLM
        response = await self._generate_response(
            user_message=user_message,
            weather_data=weather_data
        )

        # 5. Save to database
        latency_ms = int((time.time() - start_time) * 1000)
        await self.conversation_repo.add_turn(
            conversation_id=conversation_id,
            user_message=user_message,
            agent_response=response,
            llm_provider=self.llm.provider_name,
            llm_model=self.llm.current_model,
            tools_used=tools_used,
            latency_ms=latency_ms
        )

        return {
            "conversation_id": str(conversation_id),
            "response": response,
            "tools_used": tools_used,
            "llm_provider": self.llm.provider_name,
            "llm_model": self.llm.current_model,
            "latency_ms": latency_ms
        }

    async def _parse_intent(self, user_message: str) -> dict:
        """Use LLM to parse user intent."""
        system_prompt = """You are an intent parser. Determine if the user is asking about weather.

        Return JSON:
        {
            "needs_weather": true/false,
            "location": "city name or null"
        }
        """
        # Call LLM, parse response
        pass

    async def _generate_response(
        self,
        user_message: str,
        weather_data: dict | None
    ) -> str:
        """Generate natural language response."""
        if weather_data:
            system_prompt = f"""You are a helpful weather assistant.

            User asked: {user_message}
            Weather data: {weather_data}

            Generate a natural, friendly response about the weather.
            """
        else:
            system_prompt = "You are a helpful assistant. Respond to the user."

        # Call LLM
        pass
```

#### 4.2 Loop Logic Example
```python
# app/flows/weather_agent/flow.py (continued)

async def run_with_retry(
    self,
    user_message: str,
    max_retries: int = 3,
    **kwargs
) -> dict:
    """
    Run flow with retry logic.

    Example use case:
    - LLM call fails → retry
    - Weather API fails → retry
    - Exponential backoff
    """
    for attempt in range(1, max_retries + 1):
        try:
            return await self.run(user_message, **kwargs)
        except (LLMError, ToolError) as e:
            if attempt == max_retries:
                raise

            wait_time = 2 ** attempt  # Exponential backoff
            logger.warning(
                f"Flow attempt {attempt} failed, retrying in {wait_time}s",
                exc_info=True
            )
            await asyncio.sleep(wait_time)

    raise FlowError("Max retries exceeded")
```

#### Tests (Phase 4)
- [ ] Test complete flow end-to-end
- [ ] Test with different LLM providers
- [ ] Test with weather API working
- [ ] Test with weather API failing (retry logic)
- [ ] Test database persistence
- [ ] Test multi-turn conversations
- [ ] Test loop/retry logic
- [ ] Mock all external calls for fast tests

---

### Phase 5: API Endpoints (Day 8)

**Goal:** Expose weather agent via REST API

#### 5.1 API Endpoints
```python
# app/api/weather_agent.py
router = APIRouter(prefix="/api/v1/weather", tags=["weather-agent"])

class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    tenant_id: str = "default"
    llm_provider: str | None = None  # Override default
    llm_model: str | None = None     # Override default

class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    tools_used: list[str]
    llm_provider: str
    llm_model: str
    latency_ms: int

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with weather agent.

    Example:
        POST /api/v1/weather/chat
        {
            "message": "What's the weather in Paris?",
            "tenant_id": "acme-corp"
        }
    """
    flow = WeatherAgentFlow(
        llm_provider=request.llm_provider,
        llm_model=request.llm_model
    )

    result = await flow.run_with_retry(
        user_message=request.message,
        conversation_id=UUID(request.conversation_id) if request.conversation_id else None,
        tenant_id=request.tenant_id
    )

    return ChatResponse(**result)

@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: UUID):
    """Get conversation history."""
    repo = ConversationRepository()
    turns = await repo.get_conversation_history(conversation_id)
    return {
        "conversation_id": str(conversation_id),
        "turns": [
            {
                "turn_number": turn.turn_number,
                "user_message": turn.user_message,
                "agent_response": turn.agent_response,
                "tools_used": turn.tools_used,
                "llm_model": turn.llm_model
            }
            for turn in turns
        ]
    }

@router.get("/models")
async def list_available_models():
    """List all available LLM models across providers."""
    return {
        "openai": OpenAIAdapter.supported_models,
        "vertex": VertexAIAdapter.supported_models,
        "mistral": MistralAdapter.supported_models
    }
```

#### 5.2 Register Router
```python
# app/main.py
from app.api.weather_agent import router as weather_router

app.include_router(weather_router)
```

#### Tests (Phase 5)
- [ ] Test POST /chat endpoint
- [ ] Test GET /conversations/{id} endpoint
- [ ] Test GET /models endpoint
- [ ] Test with invalid inputs
- [ ] Test switching LLM providers via API
- [ ] Integration test: curl → API → agent → database

---

### Phase 6: Testing & Documentation (Days 9-10)

**Goal:** 80%+ test coverage and complete documentation

#### 6.1 Test Coverage Requirements
- **LLM Adapters:** 90%+ (critical path)
- **Weather Tool:** 85%+
- **Database Models/Repos:** 90%+
- **Flow Logic:** 85%+
- **API Endpoints:** 80%+

#### 6.2 Test Structure
```
tests/
├── test_adapters/
│   ├── test_llm_openai.py
│   ├── test_llm_vertex.py
│   ├── test_llm_mistral.py
│   └── test_llm_factory.py
├── test_tools/
│   └── test_weather_tool.py
├── test_db/
│   ├── test_conversation_models.py
│   └── test_conversation_repository.py
├── test_flows/
│   └── test_weather_agent_flow.py
├── test_api/
│   └── test_weather_agent_endpoints.py
└── integration/
    └── test_weather_agent_e2e.py
```

#### 6.3 Documentation to Create
- [ ] `docs/wave2 - Core Services/README.md` (this file becomes the Wave 2 summary)
- [ ] `docs/guides/WEATHER_AGENT_GUIDE.md` - How to use the weather agent
- [ ] `docs/guides/MULTI_LLM_GUIDE.md` - How to switch between LLM providers
- [ ] `docs/guides/BUILDING_AGENTS.md` - Template for building new agents
- [ ] Update `docs/DEVELOPER_GUIDE.md` with weather agent examples
- [ ] Update `docs/project/ARCHITECTURE.md` with flow diagrams
- [ ] Update `README.md` with weather agent demo

---

## Configuration Files

### .env.example
```bash
# Database
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/ai_agency

# LLM Provider (choose one)
LLM_PROVIDER=openai  # or "vertex" or "mistral"
LLM_MODEL=gpt-4      # optional, provider default if not set

# OpenAI
OPENAI_API_KEY=sk-...

# Vertex AI (GCP)
GCP_PROJECT_ID=your-project
GCP_REGION=europe-west1
# Vertex AI uses Application Default Credentials or service account

# Mistral
MISTRAL_API_KEY=...

# Weather API
WEATHER_API_KEY=...  # OpenWeatherMap API key (free tier)

# Application
LOG_LEVEL=INFO
ENVIRONMENT=development
```

### GCP Secret Manager (Production)
```bash
# Add to Secret Manager
gcloud secrets create OPENAI_API_KEY --data-file=- <<< "sk-..."
gcloud secrets create MISTRAL_API_KEY --data-file=- <<< "..."
gcloud secrets create WEATHER_API_KEY --data-file=- <<< "..."
```

---

## Deployment Checklist

### Local Development
- [ ] Docker Compose running PostgreSQL
- [ ] Run migrations: `alembic upgrade head`
- [ ] Set environment variables in `.env`
- [ ] Start API: `uvicorn app.main:app --reload`
- [ ] Test: `curl -X POST http://localhost:8080/api/v1/weather/chat -d '{"message": "Weather in Paris?"}'`

### GCP Production
- [ ] Secrets in Secret Manager
- [ ] Cloud SQL migrations applied
- [ ] Cloud Run deployment updated
- [ ] Test: `curl -X POST https://ai-agency-4ebxrg4hdq-ew.a.run.app/api/v1/weather/chat ...`

---

## Success Metrics

At the end of Wave 2, we should have:

### Functional
- ✅ Working weather agent locally
- ✅ Working weather agent on GCP
- ✅ Can switch between OpenAI/Gemini/Mistral models via env var
- ✅ Conversation history persisted in database
- ✅ Retry logic working on failures
- ✅ Multi-turn conversations working

### Technical
- ✅ Test coverage ≥ 80%
- ✅ All tests passing in CI/CD
- ✅ Documentation complete
- ✅ Example usage in README
- ✅ Production deployment working

### Template
- ✅ Other developers can copy weather agent pattern
- ✅ Clear guide for building new agents
- ✅ Reusable components (LLM adapters, base classes, flow patterns)

---

## Timeline

**Total: 10 days**

| Phase | Days | Deliverable |
|-------|------|-------------|
| 1. Multi-LLM Foundation | 2 | OpenAI, Vertex, Mistral adapters working |
| 2. Weather API Tool | 2 | Weather tool with retry logic |
| 3. Database Schema | 1 | Conversation tables and repository |
| 4. Weather Agent Flow | 2 | Complete flow orchestration |
| 5. API Endpoints | 1 | REST API for weather agent |
| 6. Testing & Docs | 2 | 80%+ coverage, complete docs |

---

## Next Steps After Wave 2

With working weather agent, we can:

1. **Wave 3:** Build real use case agents
   - Maturity assessment agent (using similar pattern)
   - Use case grooming agent

2. **Wave 4:** Add RAG capabilities
   - Document ingestion
   - Vector search with pgvector
   - Agents can search knowledge base

3. **Wave 5:** Advanced features
   - Authentication
   - Rate limiting
   - Streaming responses
   - Async webhooks

---

## Questions to Resolve

- [ ] Which weather API to use? (OpenWeatherMap free tier recommended)
- [ ] Should we support streaming responses now or later?
- [ ] Do we need authentication in Wave 2 or Wave 5?
- [ ] Should Mistral be included or focus on OpenAI + Vertex only?

---

**This plan serves as the living document for Wave 2. Update as we progress.**
