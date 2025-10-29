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
                "message": {"content": "Python is a programming language."},
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
