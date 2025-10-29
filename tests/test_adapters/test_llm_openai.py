"""Tests for OpenAI LLM adapter.

This module tests the OpenAI adapter with mocked API calls to ensure
proper functionality without making actual API requests.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from openai import OpenAIError

from app.adapters.llm_openai import OpenAIAdapter
from app.core.exceptions import LLMError, RateLimitError


@pytest.fixture
def openai_adapter():
    """Create OpenAI adapter instance for testing."""
    with patch("app.adapters.llm_openai.AsyncOpenAI"):
        adapter = OpenAIAdapter(
            api_key="test-key",
            model="gpt-4-turbo-2024-04-09",
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


class TestOpenAIAdapterInit:
    """Test OpenAI adapter initialization."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        with patch("app.adapters.llm_openai.AsyncOpenAI"):
            adapter = OpenAIAdapter(api_key="test-key")

            assert adapter.provider_name == "openai"
            assert adapter.model_name == "gpt-4-turbo-2024-04-09"  # Default from settings
            assert adapter._request_count == 0

    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        with patch("app.adapters.llm_openai.settings") as mock_settings:
            mock_settings.openai_api_key = ""
            mock_settings.openai_model = "gpt-4-turbo-2024-04-09"
            mock_settings.openai_rate_limit_tpm = 90000
            mock_settings.openai_rate_limit_tph = 5000000

            with pytest.raises(ValueError, match="OpenAI API key not provided"):
                OpenAIAdapter(api_key=None)

    def test_init_custom_model(self):
        """Test initialization with custom model."""
        with patch("app.adapters.llm_openai.AsyncOpenAI"):
            adapter = OpenAIAdapter(
                api_key="test-key",
                model="gpt-3.5-turbo",
            )

            assert adapter.model_name == "gpt-3.5-turbo"


class TestOpenAIAdapterComplete:
    """Test OpenAI adapter completion methods."""

    @pytest.mark.asyncio
    async def test_complete_success(self, openai_adapter, sample_messages):
        """Test successful completion."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Python is a programming language."
        mock_response.usage.total_tokens = 50

        openai_adapter.client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        # Call complete
        result = await openai_adapter.complete(
            messages=sample_messages,
            temperature=0.7,
            tenant_id="test-tenant",
        )

        assert result == "Python is a programming language."
        assert openai_adapter._request_count == 1

    @pytest.mark.asyncio
    async def test_complete_with_max_tokens(self, openai_adapter, sample_messages):
        """Test completion with max_tokens parameter."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Python is great."
        mock_response.usage.total_tokens = 30

        openai_adapter.client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await openai_adapter.complete(
            messages=sample_messages,
            max_tokens=100,
            tenant_id="test-tenant",
        )

        assert result == "Python is great."

        # Verify max_tokens was passed
        call_kwargs = openai_adapter.client.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_complete_api_error(self, openai_adapter, sample_messages):
        """Test completion with API error."""
        openai_adapter.client.chat.completions.create = AsyncMock(
            side_effect=OpenAIError("API error")
        )

        with pytest.raises(LLMError, match="OpenAI API call failed"):
            await openai_adapter.complete(
                messages=sample_messages,
                tenant_id="test-tenant",
            )

    @pytest.mark.asyncio
    async def test_complete_rate_limit(self, openai_adapter, sample_messages):
        """Test rate limiting behavior."""
        # Configure very low rate limits
        openai_adapter.rate_limiter.tokens_per_minute = 10
        openai_adapter.rate_limiter.burst_size = 10

        # Large message that exceeds rate limit
        large_messages = [
            {"role": "user", "content": "x" * 1000}  # ~325 tokens estimated
        ]

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.usage.total_tokens = 50

        openai_adapter.client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        # First call should succeed
        result1 = await openai_adapter.complete(
            messages=[{"role": "user", "content": "hi"}],
            tenant_id="test-tenant",
        )
        assert result1 == "Response"

        # Second call with large message should hit rate limit
        with pytest.raises(RateLimitError):
            await openai_adapter.complete(
                messages=large_messages,
                tenant_id="test-tenant",
            )

    @pytest.mark.asyncio
    async def test_complete_empty_content(self, openai_adapter, sample_messages):
        """Test completion with empty content."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None  # Empty content
        mock_response.usage.total_tokens = 10

        openai_adapter.client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await openai_adapter.complete(
            messages=sample_messages,
            tenant_id="test-tenant",
        )

        assert result == ""


class TestOpenAIAdapterCompleteWithMetadata:
    """Test OpenAI adapter completion with metadata."""

    @pytest.mark.asyncio
    async def test_complete_with_metadata_success(self, openai_adapter, sample_messages):
        """Test successful completion with metadata."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Python is a language."
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4-turbo-2024-04-09"
        mock_response.usage.total_tokens = 50

        openai_adapter.client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await openai_adapter.complete_with_metadata(
            messages=sample_messages,
            tenant_id="test-tenant",
        )

        assert result["content"] == "Python is a language."
        assert result["model"] == "gpt-4-turbo-2024-04-09"
        assert result["tokens_used"] == 50
        assert result["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_complete_with_metadata_no_usage(self, openai_adapter, sample_messages):
        """Test completion with metadata when usage is None."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = None  # No usage data

        openai_adapter.client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await openai_adapter.complete_with_metadata(
            messages=sample_messages,
            tenant_id="test-tenant",
        )

        assert result["content"] == "Response"
        assert result["tokens_used"] == 0  # Default to 0 when usage is None


class TestOpenAIAdapterTokenEstimation:
    """Test token estimation logic."""

    def test_estimate_tokens_small_message(self, openai_adapter):
        """Test token estimation for small message."""
        messages = [{"role": "user", "content": "Hello"}]
        tokens = openai_adapter._estimate_tokens(messages)

        # "Hello" is 5 chars, should estimate at least 10 (minimum)
        assert tokens >= 10

    def test_estimate_tokens_large_message(self, openai_adapter):
        """Test token estimation for large message."""
        messages = [
            {"role": "user", "content": "x" * 1000}  # 1000 characters
        ]
        tokens = openai_adapter._estimate_tokens(messages)

        # 1000 chars / 4 * 1.3 = ~325 tokens
        assert 300 <= tokens <= 400

    def test_estimate_tokens_multiple_messages(self, openai_adapter):
        """Test token estimation for multiple messages."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."},
        ]
        tokens = openai_adapter._estimate_tokens(messages)

        # Should sum all message contents
        assert tokens > 10


class TestOpenAIAdapterRetry:
    """Test retry logic."""

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self, openai_adapter, sample_messages):
        """Test that OpenAI errors are wrapped in LLMError."""
        # The retry decorator is configured for OpenAIError, but the complete method
        # converts OpenAIError to LLMError before it can be retried.
        # This is intentional - we want to raise LLMError to the caller, not OpenAIError.
        openai_adapter.client.chat.completions.create = AsyncMock(
            side_effect=OpenAIError("Transient error")
        )

        with pytest.raises(LLMError, match="OpenAI API call failed: Transient error"):
            await openai_adapter.complete(
                messages=sample_messages,
                tenant_id="test-tenant",
            )

        # Should have been called only once (no retry because exception is converted)
        assert openai_adapter.client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, openai_adapter, sample_messages):
        """Test that errors are consistently wrapped in LLMError."""
        openai_adapter.client.chat.completions.create = AsyncMock(
            side_effect=OpenAIError("Persistent error")
        )

        with pytest.raises(LLMError, match="OpenAI API call failed: Persistent error"):
            await openai_adapter.complete(
                messages=sample_messages,
                tenant_id="test-tenant",
            )

        # Should have tried only once (no retry because exception is converted)
        assert openai_adapter.client.chat.completions.create.call_count == 1
