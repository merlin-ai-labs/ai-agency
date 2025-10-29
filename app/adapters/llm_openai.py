"""OpenAI LLM adapter with rate limiting and retry logic.

This module provides a production-ready adapter for OpenAI's chat completion API
using GPT-4.1 models. It includes token bucket rate limiting, exponential backoff
retry logic, and comprehensive error handling.
"""

import logging

from openai import AsyncOpenAI, OpenAIError

from app.config import settings
from app.core.base import BaseAdapter
from app.core.decorators import log_execution, retry, timeout
from app.core.exceptions import LLMError, RateLimitError
from app.core.rate_limiter import TokenBucketRateLimiter
from app.core.types import LLMResponse, MessageHistory

logger = logging.getLogger(__name__)


class OpenAIAdapter(BaseAdapter):
    """OpenAI chat completion adapter with rate limiting.

    Provides a unified interface for OpenAI's GPT models with built-in
    rate limiting, retry logic, and token tracking.

    Attributes:
        provider_name: Always "openai"
        model_name: OpenAI model identifier (e.g., "gpt-4-turbo-2024-04-09")
        client: AsyncOpenAI client instance
        rate_limiter: Token bucket rate limiter for API calls

    Example:
        >>> adapter = OpenAIAdapter(
        ...     api_key="sk-...",
        ...     model="gpt-4-turbo-2024-04-09"
        ... )
        >>> response = await adapter.complete([
        ...     {"role": "user", "content": "Hello!"}
        ... ], tenant_id="tenant_123")
        >>> print(response)
        'Hello! How can I help you today?'
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        tokens_per_minute: int | None = None,
        tokens_per_hour: int | None = None,
    ) -> None:
        """Initialize OpenAI adapter.

        Args:
            api_key: OpenAI API key. If None, uses settings.openai_api_key
            model: Model name. If None, uses settings.openai_model
            tokens_per_minute: Rate limit TPM. If None, uses settings.openai_rate_limit_tpm
            tokens_per_hour: Rate limit TPH. If None, uses settings.openai_rate_limit_tph

        Raises:
            ValueError: If API key is not provided
        """
        model = model or settings.openai_model
        super().__init__(provider_name="openai", model_name=model)

        # Initialize API client
        api_key = api_key or settings.openai_api_key
        if not api_key:
            msg = "OpenAI API key not provided"
            raise ValueError(msg)

        self.client = AsyncOpenAI(api_key=api_key)

        # Initialize rate limiter
        tpm = tokens_per_minute or settings.openai_rate_limit_tpm
        tph = tokens_per_hour or settings.openai_rate_limit_tph
        self.rate_limiter = TokenBucketRateLimiter(
            tokens_per_minute=tpm,
            tokens_per_hour=tph,
        )

        logger.info(
            f"Initialized OpenAI adapter with model {model}",
            extra={
                "provider": "openai",
                "model": model,
                "rate_limit_tpm": tpm,
                "rate_limit_tph": tph,
            },
        )

    def _estimate_tokens(self, messages: MessageHistory) -> int:
        """Estimate token count for messages.

        Simple estimation: count words and multiply by 1.3 to account for
        tokenization overhead. This is approximate but sufficient for rate limiting.

        Args:
            messages: List of message dictionaries

        Returns:
            Estimated token count
        """
        total_chars = sum(len(msg["content"]) for msg in messages)
        # Rough estimate: 4 chars per token, plus 20% overhead
        estimated_tokens = int((total_chars / 4) * 1.3)
        return max(estimated_tokens, 10)  # Minimum 10 tokens

    @log_execution
    @timeout(seconds=60.0)
    @retry(
        max_attempts=3,
        backoff_type="exponential",
        min_wait=1.0,
        max_wait=10.0,
        exceptions=(OpenAIError,),
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
            tenant_id: Tenant identifier for rate limiting

        Returns:
            The generated completion text

        Raises:
            LLMError: If the API call fails
            RateLimitError: If rate limit is exceeded

        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "What is Python?"}
            ... ]
            >>> response = await adapter.complete(messages, tenant_id="user_123")
        """
        # Estimate tokens for rate limiting
        estimated_tokens = self._estimate_tokens(messages)
        if max_tokens:
            estimated_tokens += max_tokens

        # Check rate limit
        try:
            await self.rate_limiter.wait_if_needed(tenant_id, estimated_tokens)
        except RateLimitError as e:
            logger.warning(
                "Rate limit exceeded for OpenAI",
                extra={
                    "tenant_id": tenant_id,
                    "estimated_tokens": estimated_tokens,
                },
            )
            raise

        # Make API call
        try:
            logger.debug(
                "Calling OpenAI API",
                extra={
                    "model": self.model_name,
                    "message_count": len(messages),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )

            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content or ""

            self._request_count += 1
            logger.info(
                "OpenAI API call successful",
                extra={
                    "model": self.model_name,
                    "tokens_used": response.usage.total_tokens if response.usage else 0,
                    "request_count": self._request_count,
                },
            )

            return content

        except OpenAIError as e:
            logger.error(
                "OpenAI API call failed",
                extra={
                    "model": self.model_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise LLMError(
                f"OpenAI API call failed: {str(e)}",
                details={
                    "provider": "openai",
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
        exceptions=(OpenAIError,),
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
            tenant_id: Tenant identifier for rate limiting

        Returns:
            Full LLM response with metadata

        Raises:
            LLMError: If the API call fails
            RateLimitError: If rate limit is exceeded

        Example:
            >>> response = await adapter.complete_with_metadata(messages)
            >>> print(f"Used {response['tokens_used']} tokens")
            >>> print(f"Model: {response['model']}")
        """
        # Estimate tokens for rate limiting
        estimated_tokens = self._estimate_tokens(messages)
        if max_tokens:
            estimated_tokens += max_tokens

        # Check rate limit
        try:
            await self.rate_limiter.wait_if_needed(tenant_id, estimated_tokens)
        except RateLimitError as e:
            logger.warning(
                "Rate limit exceeded for OpenAI",
                extra={
                    "tenant_id": tenant_id,
                    "estimated_tokens": estimated_tokens,
                },
            )
            raise

        # Make API call
        try:
            logger.debug(
                "Calling OpenAI API with metadata",
                extra={
                    "model": self.model_name,
                    "message_count": len(messages),
                },
            )

            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else 0
            finish_reason = response.choices[0].finish_reason or "stop"

            self._request_count += 1

            result: LLMResponse = {
                "content": content,
                "model": response.model,
                "tokens_used": tokens_used,
                "finish_reason": finish_reason,
            }

            logger.info(
                "OpenAI API call with metadata successful",
                extra={
                    "model": response.model,
                    "tokens_used": tokens_used,
                    "finish_reason": finish_reason,
                },
            )

            return result

        except OpenAIError as e:
            logger.error(
                "OpenAI API call failed",
                extra={
                    "model": self.model_name,
                    "error": str(e),
                },
            )
            raise LLMError(
                f"OpenAI API call failed: {str(e)}",
                details={
                    "provider": "openai",
                    "model": self.model_name,
                },
                original_error=e,
            ) from e
