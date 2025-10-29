"""Vertex AI LLM adapter with retry logic.

This module provides a production-ready adapter for Google's Vertex AI
Gemini models. Vertex AI doesn't have strict rate limits, but we still
include retry logic for transient failures.
"""

import logging

import vertexai
from google.api_core.exceptions import GoogleAPIError
from vertexai.generative_models import GenerationConfig, GenerativeModel

from app.config import settings
from app.core.base import BaseAdapter
from app.core.decorators import log_execution, retry, timeout
from app.core.exceptions import LLMError
from app.core.types import LLMResponse, MessageHistory

logger = logging.getLogger(__name__)


class VertexAIAdapter(BaseAdapter):
    """Vertex AI (Gemini) chat completion adapter.

    Provides a unified interface for Google's Gemini models through Vertex AI.
    Vertex AI has no strict rate limits, so rate limiting is optional.

    Attributes:
        provider_name: Always "vertex_ai"
        model_name: Vertex AI model identifier (e.g., "gemini-2.0-flash-exp")
        client: GenerativeModel instance

    Example:
        >>> adapter = VertexAIAdapter(
        ...     project_id="my-project",
        ...     location="us-central1",
        ...     model="gemini-2.0-flash-exp"
        ... )
        >>> response = await adapter.complete([
        ...     {"role": "user", "content": "Hello!"}
        ... ])
        >>> print(response)
        'Hello! How can I help you today?'
    """

    def __init__(
        self,
        project_id: str | None = None,
        location: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize Vertex AI adapter.

        Args:
            project_id: GCP project ID. If None, uses settings.vertex_ai_project_id or gcp_project_id
            location: GCP location. If None, uses settings.vertex_ai_location
            model: Model name. If None, uses settings.vertex_ai_model

        Raises:
            ValueError: If project_id is not provided
        """
        model = model or settings.vertex_ai_model
        super().__init__(provider_name="vertex_ai", model_name=model)

        # Initialize Vertex AI
        project_id = project_id or settings.vertex_ai_project_id or settings.gcp_project_id
        if not project_id:
            msg = "GCP project ID not provided"
            raise ValueError(msg)

        location = location or settings.vertex_ai_location

        vertexai.init(project=project_id, location=location)

        # Initialize generative model
        self.client = GenerativeModel(model)

        logger.info(
            f"Initialized Vertex AI adapter with model {model}",
            extra={
                "provider": "vertex_ai",
                "model": model,
                "project_id": project_id,
                "location": location,
            },
        )

    def _format_messages(self, messages: MessageHistory) -> str:
        """Convert messages to Vertex AI prompt format.

        Vertex AI uses a simple text prompt format, so we need to convert
        the message history into a formatted string.

        Args:
            messages: List of message dictionaries

        Returns:
            Formatted prompt string
        """
        formatted_parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")

        return "\n\n".join(formatted_parts)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Simple estimation: count characters and divide by 4.
        This is approximate but sufficient for tracking.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        return max(len(text) // 4, 10)

    @log_execution
    @timeout(seconds=60.0)
    @retry(
        max_attempts=3,
        backoff_type="exponential",
        min_wait=1.0,
        max_wait=10.0,
        exceptions=(GoogleAPIError,),
    )
    async def complete(
        self,
        messages: MessageHistory,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tenant_id: str = "default",
    ) -> str:
        """Generate a completion from messages.

        Args:
            messages: Conversation history as list of message dicts
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate (None = model default)
            tenant_id: Tenant identifier (for compatibility, not used for rate limiting)

        Returns:
            The generated completion text

        Raises:
            LLMError: If the API call fails

        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "What is Python?"}
            ... ]
            >>> response = await adapter.complete(messages)
        """
        # Format messages for Vertex AI
        prompt = self._format_messages(messages)

        # Configure generation
        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        try:
            logger.debug(
                "Calling Vertex AI API",
                extra={
                    "model": self.model_name,
                    "message_count": len(messages),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )

            response = await self.client.generate_content_async(
                prompt,
                generation_config=generation_config,
            )

            content = response.text

            self._request_count += 1
            logger.info(
                "Vertex AI API call successful",
                extra={
                    "model": self.model_name,
                    "estimated_tokens": self._estimate_tokens(prompt + content),
                    "request_count": self._request_count,
                },
            )

            return content

        except GoogleAPIError as e:
            logger.error(
                "Vertex AI API call failed",
                extra={
                    "model": self.model_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise LLMError(
                f"Vertex AI API call failed: {str(e)}",
                details={
                    "provider": "vertex_ai",
                    "model": self.model_name,
                    "error_type": type(e).__name__,
                },
                original_error=e,
            ) from e
        except Exception as e:
            logger.error(
                "Unexpected error in Vertex AI call",
                extra={
                    "model": self.model_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise LLMError(
                f"Unexpected error in Vertex AI call: {str(e)}",
                details={
                    "provider": "vertex_ai",
                    "model": self.model_name,
                    "error_type": type(e).__name__,
                },
                original_error=e,
            ) from e

    @log_execution
    @timeout(seconds=60.0)
    @retry(
        max_attempts=3,
        backoff_type="exponential",
        min_wait=1.0,
        max_wait=10.0,
        exceptions=(GoogleAPIError,),
    )
    async def complete_with_metadata(
        self,
        messages: MessageHistory,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tenant_id: str = "default",
    ) -> LLMResponse:
        """Generate completion with full response metadata.

        Use this when you need token counts, finish reasons, and other
        metadata beyond just the completion text.

        Args:
            messages: Conversation history
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tenant_id: Tenant identifier (for compatibility)

        Returns:
            Full LLM response with metadata

        Raises:
            LLMError: If the API call fails

        Example:
            >>> response = await adapter.complete_with_metadata(messages)
            >>> print(f"Used ~{response['tokens_used']} tokens")
            >>> print(f"Model: {response['model']}")
        """
        # Format messages for Vertex AI
        prompt = self._format_messages(messages)

        # Configure generation
        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        try:
            logger.debug(
                "Calling Vertex AI API with metadata",
                extra={
                    "model": self.model_name,
                    "message_count": len(messages),
                },
            )

            response = await self.client.generate_content_async(
                prompt,
                generation_config=generation_config,
            )

            content = response.text

            # Vertex AI doesn't provide exact token counts, so we estimate
            estimated_tokens = self._estimate_tokens(prompt + content)

            # Determine finish reason from response
            finish_reason = "stop"
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "finish_reason"):
                    finish_reason = str(candidate.finish_reason.name).lower()

            self._request_count += 1

            result: LLMResponse = {
                "content": content,
                "model": self.model_name,
                "tokens_used": estimated_tokens,
                "finish_reason": finish_reason,
            }

            logger.info(
                "Vertex AI API call with metadata successful",
                extra={
                    "model": self.model_name,
                    "estimated_tokens": estimated_tokens,
                    "finish_reason": finish_reason,
                },
            )

            return result

        except GoogleAPIError as e:
            logger.error(
                "Vertex AI API call failed",
                extra={
                    "model": self.model_name,
                    "error": str(e),
                },
            )
            raise LLMError(
                f"Vertex AI API call failed: {str(e)}",
                details={
                    "provider": "vertex_ai",
                    "model": self.model_name,
                },
                original_error=e,
            ) from e
        except Exception as e:
            logger.error(
                "Unexpected error in Vertex AI call",
                extra={
                    "model": self.model_name,
                    "error": str(e),
                },
            )
            raise LLMError(
                f"Unexpected error in Vertex AI call: {str(e)}",
                details={
                    "provider": "vertex_ai",
                    "model": self.model_name,
                },
                original_error=e,
            ) from e
