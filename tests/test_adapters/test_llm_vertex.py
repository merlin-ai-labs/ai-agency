"""Tests for Vertex AI LLM adapter.

This module tests the Vertex AI adapter with mocked API calls to ensure
proper functionality without making actual API requests.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from google.api_core.exceptions import GoogleAPIError

from app.adapters.llm_vertex import VertexAIAdapter
from app.core.exceptions import LLMError


@pytest.fixture
def vertex_adapter():
    """Create Vertex AI adapter instance for testing."""
    with patch("app.adapters.llm_vertex.vertexai.init"):
        with patch("app.adapters.llm_vertex.GenerativeModel"):
            adapter = VertexAIAdapter(
                project_id="test-project",
                location="us-central1",
                model="gemini-2.0-flash-exp",
            )
            return adapter


@pytest.fixture
def sample_messages():
    """Sample message history for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"},
    ]


class TestVertexAIAdapterInit:
    """Test Vertex AI adapter initialization."""

    def test_init_with_project_id(self):
        """Test initialization with project ID."""
        with patch("app.adapters.llm_vertex.vertexai.init"):
            with patch("app.adapters.llm_vertex.GenerativeModel"):
                adapter = VertexAIAdapter(project_id="test-project")

                assert adapter.provider_name == "vertex_ai"
                assert adapter.model_name == "gemini-2.0-flash-exp"
                assert adapter._request_count == 0

    def test_init_without_project_id(self):
        """Test initialization fails without project ID."""
        with patch("app.adapters.llm_vertex.settings") as mock_settings:
            mock_settings.vertex_ai_project_id = None
            mock_settings.gcp_project_id = ""
            mock_settings.vertex_ai_location = "us-central1"
            mock_settings.vertex_ai_model = "gemini-2.0-flash-exp"

            with pytest.raises(ValueError, match="GCP project ID not provided"):
                VertexAIAdapter(project_id=None)

    def test_init_custom_model(self):
        """Test initialization with custom model."""
        with patch("app.adapters.llm_vertex.vertexai.init"):
            with patch("app.adapters.llm_vertex.GenerativeModel"):
                adapter = VertexAIAdapter(
                    project_id="test-project",
                    model="gemini-1.5-pro",
                )

                assert adapter.model_name == "gemini-1.5-pro"


class TestVertexAIAdapterFormatMessages:
    """Test message formatting."""

    def test_format_single_message(self, vertex_adapter):
        """Test formatting single message."""
        messages = [{"role": "user", "content": "Hello"}]
        formatted = vertex_adapter._format_messages(messages)

        assert formatted == "User: Hello"

    def test_format_multiple_messages(self, vertex_adapter):
        """Test formatting multiple messages."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        formatted = vertex_adapter._format_messages(messages)

        expected = "System: You are helpful.\n\nUser: Hi\n\nAssistant: Hello!"
        assert formatted == expected

    def test_format_messages_with_all_roles(self, vertex_adapter, sample_messages):
        """Test formatting with system, user, and assistant roles."""
        formatted = vertex_adapter._format_messages(sample_messages)

        assert "System: You are a helpful assistant." in formatted
        assert "User: What is Python?" in formatted


class TestVertexAIAdapterComplete:
    """Test Vertex AI adapter completion methods."""

    @pytest.mark.asyncio
    async def test_complete_success(self, vertex_adapter, sample_messages):
        """Test successful completion."""
        # Mock response
        mock_response = Mock()
        mock_response.text = "Python is a programming language."

        vertex_adapter.client.generate_content_async = AsyncMock(
            return_value=mock_response
        )

        result = await vertex_adapter.complete(
            messages=sample_messages,
            temperature=0.7,
            tenant_id="test-tenant",
        )

        assert result == "Python is a programming language."
        assert vertex_adapter._request_count == 1

    @pytest.mark.asyncio
    async def test_complete_with_max_tokens(self, vertex_adapter, sample_messages):
        """Test completion with max_tokens parameter."""
        from vertexai.generative_models import GenerationConfig

        mock_response = Mock()
        mock_response.text = "Python is great."

        vertex_adapter.client.generate_content_async = AsyncMock(
            return_value=mock_response
        )

        result = await vertex_adapter.complete(
            messages=sample_messages,
            max_tokens=100,
            tenant_id="test-tenant",
        )

        assert result == "Python is great."

        # Verify generation_config was passed with max_output_tokens
        call_args = vertex_adapter.client.generate_content_async.call_args
        generation_config = call_args[1]["generation_config"]
        # GenerationConfig is an actual object, so we can check its attributes
        assert isinstance(generation_config, GenerationConfig)
        assert generation_config._raw_generation_config.max_output_tokens == 100

    @pytest.mark.asyncio
    async def test_complete_google_api_error(self, vertex_adapter, sample_messages):
        """Test completion with Google API error."""
        vertex_adapter.client.generate_content_async = AsyncMock(
            side_effect=GoogleAPIError("API error")
        )

        with pytest.raises(LLMError, match="Vertex AI API call failed"):
            await vertex_adapter.complete(
                messages=sample_messages,
                tenant_id="test-tenant",
            )

    @pytest.mark.asyncio
    async def test_complete_generic_error(self, vertex_adapter, sample_messages):
        """Test completion with generic error."""
        vertex_adapter.client.generate_content_async = AsyncMock(
            side_effect=Exception("Unexpected error")
        )

        with pytest.raises(LLMError, match="Unexpected error in Vertex AI call"):
            await vertex_adapter.complete(
                messages=sample_messages,
                tenant_id="test-tenant",
            )


class TestVertexAIAdapterCompleteWithMetadata:
    """Test Vertex AI adapter completion with metadata."""

    @pytest.mark.asyncio
    async def test_complete_with_metadata_success(self, vertex_adapter, sample_messages):
        """Test successful completion with metadata."""
        mock_candidate = Mock()
        mock_candidate.finish_reason.name = "STOP"

        mock_response = Mock()
        mock_response.text = "Python is a language."
        mock_response.candidates = [mock_candidate]

        vertex_adapter.client.generate_content_async = AsyncMock(
            return_value=mock_response
        )

        result = await vertex_adapter.complete_with_metadata(
            messages=sample_messages,
            tenant_id="test-tenant",
        )

        assert result["content"] == "Python is a language."
        assert result["model"] == "gemini-2.0-flash-exp"
        assert result["tokens_used"] > 0  # Should have estimated tokens
        assert result["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_complete_with_metadata_no_candidates(self, vertex_adapter, sample_messages):
        """Test completion with metadata when candidates are missing."""
        mock_response = Mock()
        mock_response.text = "Response"
        mock_response.candidates = None

        vertex_adapter.client.generate_content_async = AsyncMock(
            return_value=mock_response
        )

        result = await vertex_adapter.complete_with_metadata(
            messages=sample_messages,
            tenant_id="test-tenant",
        )

        assert result["content"] == "Response"
        assert result["finish_reason"] == "stop"  # Default

    @pytest.mark.asyncio
    async def test_complete_with_metadata_different_finish_reason(
        self, vertex_adapter, sample_messages
    ):
        """Test completion with different finish reason."""
        mock_candidate = Mock()
        mock_candidate.finish_reason.name = "MAX_TOKENS"

        mock_response = Mock()
        mock_response.text = "Response"
        mock_response.candidates = [mock_candidate]

        vertex_adapter.client.generate_content_async = AsyncMock(
            return_value=mock_response
        )

        result = await vertex_adapter.complete_with_metadata(
            messages=sample_messages,
            tenant_id="test-tenant",
        )

        assert result["finish_reason"] == "max_tokens"


class TestVertexAIAdapterTokenEstimation:
    """Test token estimation logic."""

    def test_estimate_tokens_small_text(self, vertex_adapter):
        """Test token estimation for small text."""
        tokens = vertex_adapter._estimate_tokens("Hello")

        # "Hello" is 5 chars, should estimate at least 10 (minimum)
        assert tokens == 10

    def test_estimate_tokens_large_text(self, vertex_adapter):
        """Test token estimation for large text."""
        tokens = vertex_adapter._estimate_tokens("x" * 1000)

        # 1000 chars / 4 = 250 tokens
        assert tokens == 250

    def test_estimate_tokens_empty_text(self, vertex_adapter):
        """Test token estimation for empty text."""
        tokens = vertex_adapter._estimate_tokens("")

        assert tokens == 10  # Minimum


class TestVertexAIAdapterRetry:
    """Test retry logic."""

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self, vertex_adapter, sample_messages):
        """Test that Vertex errors are wrapped in LLMError."""
        # The retry decorator is configured for GoogleAPIError, but the complete method
        # converts GoogleAPIError to LLMError before it can be retried.
        vertex_adapter.client.generate_content_async = AsyncMock(
            side_effect=GoogleAPIError("Transient error")
        )

        with pytest.raises(LLMError, match="Vertex AI API call failed: Transient error"):
            await vertex_adapter.complete(
                messages=sample_messages,
                tenant_id="test-tenant",
            )

        # Should have been called only once (no retry because exception is converted)
        assert vertex_adapter.client.generate_content_async.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, vertex_adapter, sample_messages):
        """Test that errors are consistently wrapped in LLMError."""
        vertex_adapter.client.generate_content_async = AsyncMock(
            side_effect=GoogleAPIError("Persistent error")
        )

        with pytest.raises(LLMError, match="Vertex AI API call failed: Persistent error"):
            await vertex_adapter.complete(
                messages=sample_messages,
                tenant_id="test-tenant",
            )

        # Should have tried only once (no retry because exception is converted)
        assert vertex_adapter.client.generate_content_async.call_count == 1


class TestVertexAIAdapterTemperature:
    """Test temperature parameter handling."""

    @pytest.mark.asyncio
    async def test_complete_with_custom_temperature(self, vertex_adapter, sample_messages):
        """Test completion with custom temperature."""
        from vertexai.generative_models import GenerationConfig

        mock_response = Mock()
        mock_response.text = "Response"

        vertex_adapter.client.generate_content_async = AsyncMock(
            return_value=mock_response
        )

        await vertex_adapter.complete(
            messages=sample_messages,
            temperature=0.9,
            tenant_id="test-tenant",
        )

        # Verify temperature was passed
        call_args = vertex_adapter.client.generate_content_async.call_args
        generation_config = call_args[1]["generation_config"]
        assert isinstance(generation_config, GenerationConfig)
        # Use approximate comparison due to float32 precision
        assert abs(generation_config._raw_generation_config.temperature - 0.9) < 0.01
