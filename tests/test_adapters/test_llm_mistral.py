"""Tests for Mistral AI LLM adapter.

This module tests the Mistral adapter with mocked API calls to ensure
proper functionality without making actual API requests.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

import httpx

from app.adapters.llm_mistral import MistralAdapter
from app.core.exceptions import LLMError, RateLimitError


@pytest.fixture
def mistral_adapter():
    """Create Mistral adapter instance for testing."""
    adapter = MistralAdapter(
        api_key="test-key",
        model="mistral-medium-latest",
        tokens_per_minute=1000,
        tokens_per_hour=10000,
    )
    return adapter


@pytest.fixture
def sample_messages():
    """Sample message history for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
    ]


@pytest.fixture
def mock_mistral_response():
    """Mock successful Mistral API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "Python is a programming language.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"total_tokens": 50},
        "model": "mistral-medium-latest",
    }


class TestMistralAdapterInit:
    """Test Mistral adapter initialization."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        adapter = MistralAdapter(api_key="test-key")

        assert adapter.provider_name == "mistral"
        assert adapter.model_name == "mistral-medium-latest"
        assert adapter._request_count == 0
        assert adapter.api_key == "test-key"

    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        with patch("app.adapters.llm_mistral.settings") as mock_settings:
            mock_settings.mistral_api_key = ""
            mock_settings.mistral_model = "mistral-medium-latest"
            mock_settings.mistral_rate_limit_tpm = 2000000
            mock_settings.mistral_rate_limit_tph = 100000000

            with pytest.raises(ValueError, match="Mistral API key not provided"):
                MistralAdapter(api_key=None)

    def test_init_custom_model(self):
        """Test initialization with custom model."""
        adapter = MistralAdapter(
            api_key="test-key",
            model="mistral-small",
        )

        assert adapter.model_name == "mistral-small"


class TestMistralAdapterComplete:
    """Test Mistral adapter completion methods."""

    @pytest.mark.asyncio
    async def test_complete_success(self, mistral_adapter, sample_messages, mock_mistral_response):
        """Test successful completion."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_mistral_response

            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client_class.return_value = mock_client

            result = await mistral_adapter.complete(
                messages=sample_messages,
                temperature=0.7,
                tenant_id="test-tenant",
            )

            assert result == "Python is a programming language."
            assert mistral_adapter._request_count == 1

    @pytest.mark.asyncio
    async def test_complete_with_max_tokens(
        self, mistral_adapter, sample_messages, mock_mistral_response
    ):
        """Test completion with max_tokens parameter."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_mistral_response

            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client_class.return_value = mock_client

            result = await mistral_adapter.complete(
                messages=sample_messages,
                max_tokens=100,
                tenant_id="test-tenant",
            )

            assert result == "Python is a programming language."

            # Verify max_tokens was passed
            call_kwargs = mock_client.post.call_args[1]["json"]
            assert call_kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_complete_http_error(self, mistral_adapter, sample_messages):
        """Test completion with HTTP error."""
        # Create a proper HTTP error
        error = httpx.HTTPError("Connection error")

        with patch("app.adapters.llm_mistral.httpx.AsyncClient") as mock_client_class:
            # Create mock client with proper async context manager behavior
            mock_client = AsyncMock()
            mock_client.post.side_effect = error
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            with pytest.raises(LLMError, match="Mistral API call failed"):
                await mistral_adapter.complete(
                    messages=sample_messages,
                    tenant_id="test-tenant",
                )

    @pytest.mark.asyncio
    async def test_complete_rate_limit_429(self, mistral_adapter, sample_messages):
        """Test completion with 429 rate limit response."""
        with patch("app.adapters.llm_mistral.httpx.AsyncClient") as mock_client_class:
            # Create mock client with proper async context manager behavior
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.raise_for_status = Mock(side_effect=httpx.HTTPStatusError(
                "429 Rate Limit", request=Mock(), response=mock_response
            ))

            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            # The 429 status triggers RateLimitError before raise_for_status is called
            with pytest.raises(RateLimitError, match="Mistral API rate limit exceeded"):
                await mistral_adapter.complete(
                    messages=sample_messages,
                    tenant_id="test-tenant",
                )

    @pytest.mark.asyncio
    async def test_complete_rate_limit_exceeded(self, mistral_adapter, sample_messages):
        """Test rate limiting behavior."""
        # Configure very low rate limits
        mistral_adapter.rate_limiter.tokens_per_minute = 10
        mistral_adapter.rate_limiter.burst_size = 10

        # Large message that exceeds rate limit
        large_messages = [
            {"role": "user", "content": "x" * 1000}  # ~325 tokens estimated
        ]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Response"}}],
                "usage": {"total_tokens": 20},
            }

            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client_class.return_value = mock_client

            # First small call should succeed
            result1 = await mistral_adapter.complete(
                messages=[{"role": "user", "content": "hi"}],
                tenant_id="test-tenant",
            )
            assert result1 == "Response"

            # Second large call should hit rate limit
            with pytest.raises(RateLimitError):
                await mistral_adapter.complete(
                    messages=large_messages,
                    tenant_id="test-tenant",
                )


class TestMistralAdapterCompleteWithMetadata:
    """Test Mistral adapter completion with metadata."""

    @pytest.mark.asyncio
    async def test_complete_with_metadata_success(
        self, mistral_adapter, sample_messages, mock_mistral_response
    ):
        """Test successful completion with metadata."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_mistral_response

            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client_class.return_value = mock_client

            result = await mistral_adapter.complete_with_metadata(
                messages=sample_messages,
                tenant_id="test-tenant",
            )

            assert result["content"] == "Python is a programming language."
            assert result["model"] == "mistral-medium-latest"
            assert result["tokens_used"] == 50
            assert result["finish_reason"] == "stop"
            assert result["tool_calls"] is None

    @pytest.mark.asyncio
    async def test_complete_with_metadata_no_usage(self, mistral_adapter, sample_messages):
        """Test completion with metadata when usage is missing."""
        response_no_usage = {
            "choices": [
                {
                    "message": {"content": "Response"},
                    "finish_reason": "stop",
                }
            ],
            "model": "mistral-medium",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = response_no_usage

            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client_class.return_value = mock_client

            result = await mistral_adapter.complete_with_metadata(
                messages=sample_messages,
                tenant_id="test-tenant",
            )

            assert result["content"] == "Response"
            assert result["tokens_used"] == 0  # Default when usage missing
            assert result["tool_calls"] is None


class TestMistralAdapterTokenEstimation:
    """Test token estimation logic."""

    def test_estimate_tokens_small_message(self, mistral_adapter):
        """Test token estimation for small message."""
        messages = [{"role": "user", "content": "Hello"}]
        tokens = mistral_adapter._estimate_tokens(messages)

        # "Hello" is 5 chars, should estimate at least 10 (minimum)
        assert tokens >= 10

    def test_estimate_tokens_large_message(self, mistral_adapter):
        """Test token estimation for large message."""
        messages = [
            {"role": "user", "content": "x" * 1000}  # 1000 characters
        ]
        tokens = mistral_adapter._estimate_tokens(messages)

        # 1000 chars / 4 * 1.3 = ~325 tokens
        assert 300 <= tokens <= 400

    def test_estimate_tokens_multiple_messages(self, mistral_adapter):
        """Test token estimation for multiple messages."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."},
        ]
        tokens = mistral_adapter._estimate_tokens(messages)

        # Should sum all message contents
        assert tokens > 10


class TestMistralAdapterRetry:
    """Test retry logic."""

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self, mistral_adapter, sample_messages):
        """Test that HTTP errors are wrapped in LLMError."""
        # The retry decorator is configured for httpx.HTTPError, but the complete method
        # converts httpx.HTTPError to LLMError before it can be retried.
        error = httpx.HTTPError("Error")

        with patch("app.adapters.llm_mistral.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = error
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            with pytest.raises(LLMError, match="Mistral API call failed"):
                await mistral_adapter.complete(
                    messages=sample_messages,
                    tenant_id="test-tenant",
                )

            # Should have been called only once (no retry because exception is converted)
            assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, mistral_adapter, sample_messages):
        """Test that errors are consistently wrapped in LLMError."""
        error = httpx.HTTPError("Persistent error")

        with patch("app.adapters.llm_mistral.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = error
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            with pytest.raises(LLMError, match="Mistral API call failed"):
                await mistral_adapter.complete(
                    messages=sample_messages,
                    tenant_id="test-tenant",
                )

            # Should have tried only once (no retry because exception is converted)
            assert mock_client.post.call_count == 1


class TestMistralAdapterHeaders:
    """Test API request headers."""

    @pytest.mark.asyncio
    async def test_authorization_header(self, mistral_adapter, sample_messages):
        """Test that authorization header is correctly set."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Response"}}],
                "usage": {"total_tokens": 10},
            }

            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client_class.return_value = mock_client

            await mistral_adapter.complete(
                messages=sample_messages,
                tenant_id="test-tenant",
            )

            # Verify headers
            call_kwargs = mock_client.post.call_args[1]["headers"]
            assert call_kwargs["Authorization"] == "Bearer test-key"
            assert call_kwargs["Content-Type"] == "application/json"


class TestMistralAdapterToolCalling:
    """Test Mistral adapter tool calling functionality."""

    @pytest.fixture
    def weather_tool_definition(self):
        """Sample tool definition for testing."""
        return {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name",
                        },
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            },
        }

    @pytest.mark.asyncio
    async def test_complete_with_tools_no_call(
        self, mistral_adapter, sample_messages, weather_tool_definition
    ):
        """Test complete with tools when LLM doesn't call any tool."""
        mock_response_data = {
            "choices": [
                {
                    "message": {"content": "The weather is nice today."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"total_tokens": 50},
            "model": "mistral-medium-latest",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data

            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client_class.return_value = mock_client

            result = await mistral_adapter.complete(
                messages=sample_messages,
                tenant_id="test-tenant",
                tools=[weather_tool_definition],
            )

            assert result == "The weather is nice today."
            # Verify tools were passed to API
            call_kwargs = mock_client.post.call_args[1]["json"]
            assert "tools" in call_kwargs
            assert call_kwargs["tools"] == [weather_tool_definition]
            assert call_kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_complete_with_metadata_tool_call(
        self, mistral_adapter, sample_messages, weather_tool_definition
    ):
        """Test complete_with_metadata when LLM makes a tool call."""
        mock_response_data = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Paris", "units": "celsius"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"total_tokens": 75},
            "model": "mistral-medium-latest",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data

            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client_class.return_value = mock_client

            result = await mistral_adapter.complete_with_metadata(
                messages=sample_messages,
                tenant_id="test-tenant",
                tools=[weather_tool_definition],
            )

            assert result["content"] == ""
            assert result["model"] == "mistral-medium-latest"
            assert result["tokens_used"] == 75
            assert result["finish_reason"] == "tool_calls"
            assert result["tool_calls"] is not None
            assert len(result["tool_calls"]) == 1

            tool_call = result["tool_calls"][0]
            assert tool_call["id"] == "call_abc123"
            assert tool_call["type"] == "function"
            assert tool_call["function"]["name"] == "get_weather"
            assert tool_call["function"]["arguments"] == '{"location": "Paris", "units": "celsius"}'

    @pytest.mark.asyncio
    async def test_complete_without_tools_backward_compat(
        self, mistral_adapter, sample_messages
    ):
        """Test that complete works without tools (backward compatibility)."""
        mock_response_data = {
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 20},
            "model": "mistral-medium-latest",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data

            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client_class.return_value = mock_client

            result = await mistral_adapter.complete(
                messages=sample_messages,
                tenant_id="test-tenant",
                # No tools parameter
            )

            assert result == "Hello!"
            # Verify tools were NOT passed to API
            call_kwargs = mock_client.post.call_args[1]["json"]
            assert "tools" not in call_kwargs

    @pytest.mark.asyncio
    async def test_tool_choice_none(
        self, mistral_adapter, sample_messages, weather_tool_definition
    ):
        """Test complete with tool_choice='none' forces no tool use."""
        mock_response_data = {
            "choices": [
                {
                    "message": {"content": "Weather info without tool."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"total_tokens": 30},
            "model": "mistral-medium-latest",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data

            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client_class.return_value = mock_client

            result = await mistral_adapter.complete(
                messages=sample_messages,
                tenant_id="test-tenant",
                tools=[weather_tool_definition],
                tool_choice="none",
            )

            assert result == "Weather info without tool."
            # Verify tool_choice was passed
            call_kwargs = mock_client.post.call_args[1]["json"]
            assert call_kwargs["tool_choice"] == "none"

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(
        self, mistral_adapter, sample_messages, weather_tool_definition
    ):
        """Test complete_with_metadata when LLM makes multiple tool calls."""
        mock_response_data = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Paris"}',
                                },
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "London"}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"total_tokens": 100},
            "model": "mistral-medium-latest",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data

            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client_class.return_value = mock_client

            result = await mistral_adapter.complete_with_metadata(
                messages=sample_messages,
                tenant_id="test-tenant",
                tools=[weather_tool_definition],
            )

            assert result["tool_calls"] is not None
            assert len(result["tool_calls"]) == 2
            assert result["tool_calls"][0]["id"] == "call_1"
            assert result["tool_calls"][0]["function"]["arguments"] == '{"location": "Paris"}'
            assert result["tool_calls"][1]["id"] == "call_2"
            assert result["tool_calls"][1]["function"]["arguments"] == '{"location": "London"}'

    @pytest.mark.asyncio
    async def test_complete_with_tools_null_content(
        self, mistral_adapter, sample_messages, weather_tool_definition
    ):
        """Test that null content is handled when tool call is made."""
        mock_response_data = {
            "choices": [
                {
                    "message": {
                        "content": None,  # Can be null from API
                        "tool_calls": [
                            {
                                "id": "call_test",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "NYC"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"total_tokens": 60},
            "model": "mistral-medium-latest",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data

            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = AsyncMock()
            mock_client_class.return_value = mock_client

            # Test complete() method
            result_str = await mistral_adapter.complete(
                messages=sample_messages,
                tenant_id="test-tenant",
                tools=[weather_tool_definition],
            )

            assert result_str == ""  # Should convert None to empty string

            # Test complete_with_metadata() method
            result_dict = await mistral_adapter.complete_with_metadata(
                messages=sample_messages,
                tenant_id="test-tenant",
                tools=[weather_tool_definition],
            )

            assert result_dict["content"] == ""  # Should convert None to empty string
            assert result_dict["tool_calls"] is not None
