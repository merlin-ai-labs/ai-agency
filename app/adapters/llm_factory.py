"""LLM factory for creating adapter instances based on configuration.

This module provides a factory function to instantiate LLM adapters
(OpenAI, Vertex AI, Mistral) based on configuration settings, enabling
easy provider switching.
"""

import logging

from app.adapters.llm_mistral import MistralAdapter
from app.adapters.llm_openai import OpenAIAdapter
from app.adapters.llm_vertex import VertexAIAdapter
from app.config import settings
from app.core.base import BaseAdapter

logger = logging.getLogger(__name__)


def get_llm_adapter(
    provider: str | None = None,
    model: str | None = None,
    **kwargs,
) -> BaseAdapter:
    """Create an LLM adapter instance based on provider configuration.

    Factory function that returns the appropriate LLM adapter (OpenAI,
    Vertex AI, or Mistral) based on the provider parameter or application
    settings.

    Args:
        provider: LLM provider name ("openai", "vertex", "mistral").
                 If None, uses settings.llm_provider
        model: Model name to use. If None, uses provider's default from settings
        **kwargs: Additional provider-specific parameters

    Returns:
        BaseAdapter instance for the specified provider

    Raises:
        ValueError: If provider is not supported or required credentials are missing

    Example:
        >>> # Use default provider from settings
        >>> adapter = get_llm_adapter()
        >>>
        >>> # Explicitly specify provider
        >>> adapter = get_llm_adapter(provider="openai", model="gpt-4-turbo-2024-04-09")
        >>>
        >>> # Use adapter
        >>> response = await adapter.complete([
        ...     {"role": "user", "content": "Hello!"}
        ... ])
    """
    # Use provider from settings if not specified
    provider = provider or settings.llm_provider

    logger.info(
        f"Creating LLM adapter for provider: {provider}",
        extra={
            "provider": provider,
            "model": model,
        },
    )

    # Create appropriate adapter based on provider
    if provider == "openai":
        return _create_openai_adapter(model, **kwargs)
    elif provider == "vertex":
        return _create_vertex_adapter(model, **kwargs)
    elif provider == "mistral":
        return _create_mistral_adapter(model, **kwargs)
    else:
        msg = f"Unsupported LLM provider: {provider}. Supported providers: openai, vertex, mistral"
        raise ValueError(msg)


def _create_openai_adapter(model: str | None = None, **kwargs) -> OpenAIAdapter:
    """Create OpenAI adapter with configuration from settings.

    Args:
        model: Model name. If None, uses settings.openai_model
        **kwargs: Additional OpenAI-specific parameters

    Returns:
        OpenAIAdapter instance

    Raises:
        ValueError: If API key is not configured
    """
    # Use model from llm_model override if available, otherwise use openai_model
    model = model or settings.llm_model or settings.openai_model

    adapter = OpenAIAdapter(
        api_key=settings.openai_api_key,
        model=model,
        tokens_per_minute=settings.openai_rate_limit_tpm,
        tokens_per_hour=settings.openai_rate_limit_tph,
        **kwargs,
    )

    logger.info(
        f"Created OpenAI adapter with model {model}",
        extra={"provider": "openai", "model": model},
    )

    return adapter


def _create_vertex_adapter(model: str | None = None, **kwargs) -> VertexAIAdapter:
    """Create Vertex AI adapter with configuration from settings.

    Args:
        model: Model name. If None, uses settings.vertex_ai_model
        **kwargs: Additional Vertex AI-specific parameters

    Returns:
        VertexAIAdapter instance

    Raises:
        ValueError: If GCP project is not configured
    """
    # Use model from llm_model override if available, otherwise use vertex_ai_model
    model = model or settings.llm_model or settings.vertex_ai_model

    adapter = VertexAIAdapter(
        project_id=settings.vertex_ai_project_id or settings.gcp_project_id,
        location=settings.vertex_ai_location,
        model=model,
        **kwargs,
    )

    logger.info(
        f"Created Vertex AI adapter with model {model}",
        extra={"provider": "vertex_ai", "model": model},
    )

    return adapter


def _create_mistral_adapter(model: str | None = None, **kwargs) -> MistralAdapter:
    """Create Mistral adapter with configuration from settings.

    Args:
        model: Model name. If None, uses settings.mistral_model
        **kwargs: Additional Mistral-specific parameters

    Returns:
        MistralAdapter instance

    Raises:
        ValueError: If API key is not configured
    """
    # Use model from llm_model override if available, otherwise use mistral_model
    model = model or settings.llm_model or settings.mistral_model

    adapter = MistralAdapter(
        api_key=settings.mistral_api_key,
        model=model,
        tokens_per_minute=settings.mistral_rate_limit_tpm,
        tokens_per_hour=settings.mistral_rate_limit_tph,
        **kwargs,
    )

    logger.info(
        f"Created Mistral adapter with model {model}",
        extra={"provider": "mistral", "model": model},
    )

    return adapter


# Alias for backward compatibility
get_chat_model = get_llm_adapter
