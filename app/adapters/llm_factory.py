"""LLM factory for creating chat models based on configuration.

TODO:
- Add embeddings adapters (OpenAI, Vertex)
- Add storage adapters (GCS)
"""

from app.config import settings
from app.adapters.llm_openai import OpenAIChatModel
from app.adapters.llm_vertex import VertexChatModel
import structlog

logger = structlog.get_logger()


def get_chat_model(provider: str = None, **kwargs):
    """
    Create a chat model based on provider configuration.

    Args:
        provider: LLM provider ("openai" or "vertex"). If None, uses settings.llm_provider
        **kwargs: Additional parameters for the model

    Returns:
        ChatModel instance (OpenAI or Vertex)

    TODO:
    - Add caching to avoid recreating models
    - Add model validation
    - Support custom models
    """
    provider = provider or settings.llm_provider

    if provider == "openai":
        logger.info("llm_factory", provider="openai")
        return OpenAIChatModel(**kwargs)
    elif provider == "vertex":
        logger.info("llm_factory", provider="vertex")
        return VertexChatModel(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
