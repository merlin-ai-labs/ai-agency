"""Tests for LLM factory.

This module tests the factory function that creates LLM adapter instances
based on provider configuration.
"""

import pytest
from unittest.mock import patch

from app.adapters.llm_factory import (
    get_llm_adapter,
    _create_openai_adapter,
    _create_vertex_adapter,
    _create_mistral_adapter,
)
from app.adapters.llm_openai import OpenAIAdapter
from app.adapters.llm_vertex import VertexAIAdapter
from app.adapters.llm_mistral import MistralAdapter
from app.core.base import BaseAdapter


class TestGetLLMAdapter:
    """Test the main factory function."""

    @patch("app.adapters.llm_factory._create_openai_adapter")
    def test_get_openai_adapter(self, mock_create_openai):
        """Test creating OpenAI adapter."""
        mock_adapter = OpenAIAdapter.__new__(OpenAIAdapter)
        mock_create_openai.return_value = mock_adapter

        result = get_llm_adapter(provider="openai")

        assert result == mock_adapter
        mock_create_openai.assert_called_once_with(None)

    @patch("app.adapters.llm_factory._create_vertex_adapter")
    def test_get_vertex_adapter(self, mock_create_vertex):
        """Test creating Vertex AI adapter."""
        mock_adapter = VertexAIAdapter.__new__(VertexAIAdapter)
        mock_create_vertex.return_value = mock_adapter

        result = get_llm_adapter(provider="vertex")

        assert result == mock_adapter
        mock_create_vertex.assert_called_once_with(None)

    @patch("app.adapters.llm_factory._create_mistral_adapter")
    def test_get_mistral_adapter(self, mock_create_mistral):
        """Test creating Mistral adapter."""
        mock_adapter = MistralAdapter.__new__(MistralAdapter)
        mock_create_mistral.return_value = mock_adapter

        result = get_llm_adapter(provider="mistral")

        assert result == mock_adapter
        mock_create_mistral.assert_called_once_with(None)

    def test_get_adapter_unsupported_provider(self):
        """Test error when provider is not supported."""
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            get_llm_adapter(provider="unsupported")

    @patch("app.adapters.llm_factory._create_openai_adapter")
    @patch("app.config.settings")
    def test_get_adapter_default_provider(self, mock_settings, mock_create_openai):
        """Test using default provider from settings."""
        mock_settings.llm_provider = "openai"
        mock_adapter = OpenAIAdapter.__new__(OpenAIAdapter)
        mock_create_openai.return_value = mock_adapter

        result = get_llm_adapter()  # No provider specified

        assert result == mock_adapter

    @patch("app.adapters.llm_factory._create_openai_adapter")
    def test_get_adapter_with_custom_model(self, mock_create_openai):
        """Test creating adapter with custom model."""
        mock_adapter = OpenAIAdapter.__new__(OpenAIAdapter)
        mock_create_openai.return_value = mock_adapter

        result = get_llm_adapter(provider="openai", model="gpt-3.5-turbo")

        mock_create_openai.assert_called_once_with("gpt-3.5-turbo")

    @patch("app.adapters.llm_factory._create_openai_adapter")
    def test_get_adapter_with_kwargs(self, mock_create_openai):
        """Test creating adapter with additional kwargs."""
        mock_adapter = OpenAIAdapter.__new__(OpenAIAdapter)
        mock_create_openai.return_value = mock_adapter

        result = get_llm_adapter(
            provider="openai",
            model="gpt-4",
            tokens_per_minute=5000,
        )

        mock_create_openai.assert_called_once_with(
            "gpt-4",
            tokens_per_minute=5000,
        )


class TestCreateOpenAIAdapter:
    """Test OpenAI adapter creation."""

    @patch("app.adapters.llm_factory.OpenAIAdapter")
    @patch("app.adapters.llm_factory.settings")
    def test_create_openai_adapter_default(self, mock_settings, mock_adapter_class):
        """Test creating OpenAI adapter with defaults."""
        mock_settings.openai_api_key = "test-key"
        mock_settings.openai_model = "gpt-4-turbo-2024-04-09"
        mock_settings.llm_model = None
        mock_settings.openai_rate_limit_tpm = 90000
        mock_settings.openai_rate_limit_tph = 5000000

        _create_openai_adapter()

        mock_adapter_class.assert_called_once_with(
            api_key="test-key",
            model="gpt-4-turbo-2024-04-09",
            tokens_per_minute=90000,
            tokens_per_hour=5000000,
        )

    @patch("app.adapters.llm_factory.OpenAIAdapter")
    @patch("app.adapters.llm_factory.settings")
    def test_create_openai_adapter_custom_model(self, mock_settings, mock_adapter_class):
        """Test creating OpenAI adapter with custom model."""
        mock_settings.openai_api_key = "test-key"
        mock_settings.openai_model = "gpt-4-turbo-2024-04-09"
        mock_settings.llm_model = None
        mock_settings.openai_rate_limit_tpm = 90000
        mock_settings.openai_rate_limit_tph = 5000000

        _create_openai_adapter(model="gpt-3.5-turbo")

        mock_adapter_class.assert_called_once_with(
            api_key="test-key",
            model="gpt-3.5-turbo",
            tokens_per_minute=90000,
            tokens_per_hour=5000000,
        )

    @patch("app.adapters.llm_factory.OpenAIAdapter")
    @patch("app.adapters.llm_factory.settings")
    def test_create_openai_adapter_model_override(self, mock_settings, mock_adapter_class):
        """Test that llm_model setting overrides openai_model."""
        mock_settings.openai_api_key = "test-key"
        mock_settings.openai_model = "gpt-4-turbo-2024-04-09"
        mock_settings.llm_model = "gpt-4-override"
        mock_settings.openai_rate_limit_tpm = 90000
        mock_settings.openai_rate_limit_tph = 5000000

        _create_openai_adapter()

        # Should use llm_model instead of openai_model
        mock_adapter_class.assert_called_once_with(
            api_key="test-key",
            model="gpt-4-override",
            tokens_per_minute=90000,
            tokens_per_hour=5000000,
        )


class TestCreateVertexAdapter:
    """Test Vertex AI adapter creation."""

    @patch("app.adapters.llm_factory.VertexAIAdapter")
    @patch("app.adapters.llm_factory.settings")
    def test_create_vertex_adapter_default(self, mock_settings, mock_adapter_class):
        """Test creating Vertex AI adapter with defaults."""
        mock_settings.vertex_ai_project_id = "test-project"
        mock_settings.gcp_project_id = "fallback-project"
        mock_settings.vertex_ai_location = "us-central1"
        mock_settings.vertex_ai_model = "gemini-2.0-flash-exp"
        mock_settings.llm_model = None

        _create_vertex_adapter()

        mock_adapter_class.assert_called_once_with(
            project_id="test-project",
            location="us-central1",
            model="gemini-2.0-flash-exp",
        )

    @patch("app.adapters.llm_factory.VertexAIAdapter")
    @patch("app.adapters.llm_factory.settings")
    def test_create_vertex_adapter_fallback_project(self, mock_settings, mock_adapter_class):
        """Test using gcp_project_id as fallback."""
        mock_settings.vertex_ai_project_id = None
        mock_settings.gcp_project_id = "fallback-project"
        mock_settings.vertex_ai_location = "us-central1"
        mock_settings.vertex_ai_model = "gemini-2.0-flash-exp"
        mock_settings.llm_model = None

        _create_vertex_adapter()

        mock_adapter_class.assert_called_once_with(
            project_id="fallback-project",
            location="us-central1",
            model="gemini-2.0-flash-exp",
        )

    @patch("app.adapters.llm_factory.VertexAIAdapter")
    @patch("app.adapters.llm_factory.settings")
    def test_create_vertex_adapter_custom_model(self, mock_settings, mock_adapter_class):
        """Test creating Vertex AI adapter with custom model."""
        mock_settings.vertex_ai_project_id = "test-project"
        mock_settings.gcp_project_id = "fallback-project"
        mock_settings.vertex_ai_location = "us-central1"
        mock_settings.vertex_ai_model = "gemini-2.0-flash-exp"
        mock_settings.llm_model = None

        _create_vertex_adapter(model="gemini-1.5-pro")

        mock_adapter_class.assert_called_once_with(
            project_id="test-project",
            location="us-central1",
            model="gemini-1.5-pro",
        )


class TestCreateMistralAdapter:
    """Test Mistral adapter creation."""

    @patch("app.adapters.llm_factory.MistralAdapter")
    @patch("app.adapters.llm_factory.settings")
    def test_create_mistral_adapter_default(self, mock_settings, mock_adapter_class):
        """Test creating Mistral adapter with defaults."""
        mock_settings.mistral_api_key = "test-key"
        mock_settings.mistral_model = "mistral-medium-latest"
        mock_settings.llm_model = None
        mock_settings.mistral_rate_limit_tpm = 2000000
        mock_settings.mistral_rate_limit_tph = 100000000

        _create_mistral_adapter()

        mock_adapter_class.assert_called_once_with(
            api_key="test-key",
            model="mistral-medium-latest",
            tokens_per_minute=2000000,
            tokens_per_hour=100000000,
        )

    @patch("app.adapters.llm_factory.MistralAdapter")
    @patch("app.adapters.llm_factory.settings")
    def test_create_mistral_adapter_custom_model(self, mock_settings, mock_adapter_class):
        """Test creating Mistral adapter with custom model."""
        mock_settings.mistral_api_key = "test-key"
        mock_settings.mistral_model = "mistral-medium-latest"
        mock_settings.llm_model = None
        mock_settings.mistral_rate_limit_tpm = 2000000
        mock_settings.mistral_rate_limit_tph = 100000000

        _create_mistral_adapter(model="mistral-small")

        mock_adapter_class.assert_called_once_with(
            api_key="test-key",
            model="mistral-small",
            tokens_per_minute=2000000,
            tokens_per_hour=100000000,
        )

    @patch("app.adapters.llm_factory.MistralAdapter")
    @patch("app.adapters.llm_factory.settings")
    def test_create_mistral_adapter_with_kwargs(self, mock_settings, mock_adapter_class):
        """Test creating Mistral adapter with additional kwargs."""
        mock_settings.mistral_api_key = "test-key"
        mock_settings.mistral_model = "mistral-medium-latest"
        mock_settings.llm_model = None
        mock_settings.mistral_rate_limit_tpm = 2000000
        mock_settings.mistral_rate_limit_tph = 100000000

        # Pass custom_param as kwarg instead of tokens_per_minute (which factory sets explicitly)
        _create_mistral_adapter(
            model="mistral-medium",
            timeout=60,  # Pass a different kwarg that won't conflict
        )

        mock_adapter_class.assert_called_once_with(
            api_key="test-key",
            model="mistral-medium",
            tokens_per_minute=2000000,  # From settings
            tokens_per_hour=100000000,
            timeout=60,  # The kwarg we passed
        )


class TestBackwardCompatibility:
    """Test backward compatibility aliases."""

    @patch("app.adapters.llm_factory.get_llm_adapter")
    def test_get_chat_model_alias(self, mock_get_llm_adapter):
        """Test that get_chat_model is an alias for get_llm_adapter."""
        from app.adapters.llm_factory import get_chat_model

        mock_adapter = OpenAIAdapter.__new__(OpenAIAdapter)
        mock_get_llm_adapter.return_value = mock_adapter

        result = get_chat_model(provider="openai")

        # get_chat_model should be the same function
        assert get_chat_model == get_llm_adapter
