"""Mistral AI LLM adapter with rate limiting and retry logic.

This module provides a production-ready adapter for Mistral AI's chat completion API
using their latest models. It includes token bucket rate limiting, exponential backoff
retry logic, and comprehensive error handling.
"""

import logging
from typing import Any

import httpx

from app.config import settings
from app.core.base import BaseAdapter
from app.core.decorators import log_execution, retry, timeout
from app.core.exceptions import LLMError, RateLimitError
from app.core.rate_limiter import TokenBucketRateLimiter
from app.core.types import (
    LLMResponse,
    LLMResponseWithTools,
    MessageHistory,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


class MistralAdapter(BaseAdapter):
    """Mistral AI chat completion adapter with rate limiting.

    Provides a unified interface for Mistral AI models with built-in
    rate limiting, retry logic, and token tracking.

    Attributes:
        provider_name: Always "mistral"
        model_name: Mistral model identifier (e.g., "mistral-medium-latest")
        api_key: Mistral API key
        rate_limiter: Token bucket rate limiter for API calls
        api_base: Mistral API base URL

    Example:
        >>> adapter = MistralAdapter(
        ...     api_key="...",
        ...     model="mistral-medium-latest"
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
        """Initialize Mistral adapter.

        Args:
            api_key: Mistral API key. If None, uses settings.mistral_api_key
            model: Model name. If None, uses settings.mistral_model
            tokens_per_minute: Rate limit TPM. If None, uses settings.mistral_rate_limit_tpm
            tokens_per_hour: Rate limit TPH. If None, uses settings.mistral_rate_limit_tph

        Raises:
            ValueError: If API key is not provided
        """
        model = model or settings.mistral_model
        super().__init__(provider_name="mistral", model_name=model)

        # Initialize API credentials
        self.api_key = api_key or settings.mistral_api_key
        if not self.api_key:
            msg = "Mistral API key not provided"
            raise ValueError(msg)

        self.api_base = "https://api.mistral.ai/v1"

        # Initialize rate limiter
        tpm = tokens_per_minute or settings.mistral_rate_limit_tpm
        tph = tokens_per_hour or settings.mistral_rate_limit_tph
        self.rate_limiter = TokenBucketRateLimiter(
            tokens_per_minute=tpm,
            tokens_per_hour=tph,
        )

        logger.info(
            f"Initialized Mistral adapter with model {model}",
            extra={
                "provider": "mistral",
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
        exceptions=(httpx.HTTPError,),
    )
    async def complete(
        self,
        messages: MessageHistory,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tenant_id: str = "default",
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> str:
        """Generate a completion from messages.

        Args:
            messages: Conversation history as list of message dicts
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate (None = model default)
            tenant_id: Tenant identifier for rate limiting
            tools: Optional list of tool definitions for function calling
            tool_choice: Controls which tool to call ("none", "auto", or specific function)

        Returns:
            The generated completion text (may be empty if LLM calls a tool)

        Raises:
            LLMError: If the API call fails
            RateLimitError: If rate limit is exceeded

        Note:
            When tools are provided and LLM decides to call a tool, content may be empty.
            Use complete_with_metadata() to get tool_calls information.

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
                "Rate limit exceeded for Mistral",
                extra={
                    "tenant_id": tenant_id,
                    "estimated_tokens": estimated_tokens,
                },
            )
            raise

        # Prepare request payload
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        # Add tools if provided (Mistral uses OpenAI-compatible format)
        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Make API call
        try:
            logger.debug(
                "Calling Mistral API",
                extra={
                    "model": self.model_name,
                    "message_count": len(messages),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "tools_provided": tools is not None,
                },
            )

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=60.0,
                )

                # Check for HTTP errors
                if response.status_code == 429:
                    raise RateLimitError(
                        "Mistral API rate limit exceeded",
                        details={"status_code": 429},
                    )

                response.raise_for_status()

                data = response.json()

                content = data["choices"][0]["message"].get("content") or ""
                has_tool_calls = "tool_calls" in data["choices"][0]["message"]

                self._request_count += 1
                logger.info(
                    "Mistral API call successful",
                    extra={
                        "model": self.model_name,
                        "tokens_used": data.get("usage", {}).get("total_tokens", 0),
                        "request_count": self._request_count,
                        "has_tool_calls": has_tool_calls,
                    },
                )

                return content

        except RateLimitError:
            # Re-raise RateLimitError without wrapping
            raise
        except httpx.HTTPError as e:
            logger.error(
                "Mistral API call failed",
                extra={
                    "model": self.model_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise LLMError(
                f"Mistral API call failed: {str(e)}",
                details={
                    "provider": "mistral",
                    "model": self.model_name,
                    "error_type": type(e).__name__,
                },
                original_error=e,
            ) from e
        except Exception as e:
            logger.error(
                "Unexpected error in Mistral call",
                extra={
                    "model": self.model_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise LLMError(
                f"Unexpected error in Mistral call: {str(e)}",
                details={
                    "provider": "mistral",
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
        exceptions=(httpx.HTTPError,),
    )
    async def complete_with_metadata(
        self,
        messages: MessageHistory,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tenant_id: str = "default",
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMResponse | LLMResponseWithTools:
        """Generate completion with full response metadata.

        Use this when you need token counts, finish reasons, and other
        metadata beyond just the completion text. This is the recommended
        method when using tool calling.

        Args:
            messages: Conversation history
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tenant_id: Tenant identifier for rate limiting
            tools: Optional list of tool definitions for function calling
            tool_choice: Controls which tool to call ("none", "auto", or specific function)

        Returns:
            Full LLM response with metadata. If tools are provided and LLM calls
            a tool, the response will include a tool_calls field.

        Raises:
            LLMError: If the API call fails
            RateLimitError: If rate limit is exceeded

        Example:
            >>> response = await adapter.complete_with_metadata(messages)
            >>> print(f"Used {response['tokens_used']} tokens")
            >>> if response.get('tool_calls'):
            ...     print(f"LLM wants to call: {response['tool_calls'][0]['function']['name']}")
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
                "Rate limit exceeded for Mistral",
                extra={
                    "tenant_id": tenant_id,
                    "estimated_tokens": estimated_tokens,
                },
            )
            raise

        # Prepare request payload
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        # Add tools if provided (Mistral uses OpenAI-compatible format)
        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Make API call
        try:
            logger.debug(
                "Calling Mistral API with metadata",
                extra={
                    "model": self.model_name,
                    "message_count": len(messages),
                    "tools_provided": tools is not None,
                },
            )

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=60.0,
                )

                # Check for HTTP errors
                if response.status_code == 429:
                    raise RateLimitError(
                        "Mistral API rate limit exceeded",
                        details={"status_code": 429},
                    )

                response.raise_for_status()

                data = response.json()

                message = data["choices"][0]["message"]
                content = message.get("content") or ""
                tokens_used = data.get("usage", {}).get("total_tokens", 0)
                finish_reason = data["choices"][0].get("finish_reason", "stop")
                model_used = data.get("model", self.model_name)

                self._request_count += 1

                # Build base result
                result_data: dict[str, Any] = {
                    "content": content,
                    "model": model_used,
                    "tokens_used": tokens_used,
                    "finish_reason": finish_reason,
                }

                # Extract tool calls if present (Mistral uses OpenAI-compatible format)
                if "tool_calls" in message and message["tool_calls"]:
                    result_data["tool_calls"] = message["tool_calls"]
                    logger.info(
                        "Mistral API call with tool calls successful",
                        extra={
                            "model": model_used,
                            "tokens_used": tokens_used,
                            "finish_reason": finish_reason,
                            "tool_calls_count": len(message["tool_calls"]),
                        },
                    )
                else:
                    result_data["tool_calls"] = None
                    logger.info(
                        "Mistral API call with metadata successful",
                        extra={
                            "model": model_used,
                            "tokens_used": tokens_used,
                            "finish_reason": finish_reason,
                        },
                    )

                # Return as LLMResponseWithTools (superset of LLMResponse)
                return result_data  # type: ignore

        except RateLimitError:
            # Re-raise RateLimitError without wrapping
            raise
        except httpx.HTTPError as e:
            logger.error(
                "Mistral API call failed",
                extra={
                    "model": self.model_name,
                    "error": str(e),
                },
            )
            raise LLMError(
                f"Mistral API call failed: {str(e)}",
                details={
                    "provider": "mistral",
                    "model": self.model_name,
                },
                original_error=e,
            ) from e
        except Exception as e:
            logger.error(
                "Unexpected error in Mistral call",
                extra={
                    "model": self.model_name,
                    "error": str(e),
                },
            )
            raise LLMError(
                f"Unexpected error in Mistral call: {str(e)}",
                details={
                    "provider": "mistral",
                    "model": self.model_name,
                },
                original_error=e,
            ) from e
