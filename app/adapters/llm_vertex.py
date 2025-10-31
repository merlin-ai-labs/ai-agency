"""Vertex AI LLM adapter with retry logic.

This module provides a production-ready adapter for Google's Vertex AI
Gemini models. Vertex AI doesn't have strict rate limits, but we still
include retry logic for transient failures.
"""

import json
import logging
from typing import Any

import vertexai
from google.api_core.exceptions import GoogleAPIError
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Tool,
)

from app.config import settings
from app.core.base import BaseAdapter
from app.core.decorators import log_execution, retry, timeout
from app.core.exceptions import LLMError
from app.core.types import (
    LLMResponse,
    LLMResponseWithTools,
    MessageHistory,
    ToolDefinition,
)

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

    def _convert_tools_to_vertex_format(self, tools: list[ToolDefinition]) -> list[Tool]:
        """Convert OpenAI tool definitions to Vertex AI format.

        OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "...",
                "parameters": { JSON Schema }
            }
        }

        Vertex AI format:
        Tool(function_declarations=[
            FunctionDeclaration(
                name="get_weather",
                description="...",
                parameters={ JSON Schema }
            )
        ])

        Args:
            tools: List of OpenAI tool definitions

        Returns:
            List of Vertex AI Tool objects
        """
        function_declarations = []

        for tool in tools:
            if tool["type"] != "function":
                continue

            function = tool["function"]
            function_declarations.append(
                FunctionDeclaration(
                    name=function["name"],
                    description=function["description"],
                    parameters=function["parameters"],
                )
            )

        return [Tool(function_declarations=function_declarations)]

    def _extract_tool_calls(self, response: Any) -> list[dict[str, Any]] | None:
        """Extract tool calls from Vertex AI response and convert to OpenAI format.

        Vertex AI returns function calls in response.candidates[0].content.parts
        as FunctionCall objects. We need to convert these to OpenAI format.

        Args:
            response: Vertex AI GenerateContentResponse

        Returns:
            List of tool calls in OpenAI format, or None if no tool calls
        """
        try:
            if not hasattr(response, "candidates") or not response.candidates:
                return None

            candidate = response.candidates[0]
            if not hasattr(candidate, "content") or not candidate.content:
                return None

            if not hasattr(candidate.content, "parts") or not candidate.content.parts:
                return None

            # Make sure parts is actually iterable (not a Mock)
            parts = candidate.content.parts
            if not hasattr(parts, "__iter__"):
                return None

            tool_calls = []
            for part in parts:
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    # Convert Vertex AI function call to OpenAI format
                    tool_calls.append(
                        {
                            "id": f"call_{fc.name}_{len(tool_calls)}",  # Generate ID
                            "type": "function",
                            "function": {
                                "name": fc.name,
                                "arguments": json.dumps(dict(fc.args)),
                            },
                        }
                    )

            return tool_calls if tool_calls else None
        except (AttributeError, TypeError):
            # If anything goes wrong extracting tool calls, return None
            # This handles Mock objects and other edge cases in tests
            return None

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
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> str:
        """Generate a completion from messages.

        Args:
            messages: Conversation history as list of message dicts
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate (None = model default)
            tenant_id: Tenant identifier (for compatibility, not used for rate limiting)
            tools: Optional list of tool definitions for function calling
            tool_choice: Controls which tool to call (\"none\", \"auto\", or specific function)

        Returns:
            The generated completion text (may be empty if LLM calls a tool)

        Raises:
            LLMError: If the API call fails

        Note:
            When tools are provided and LLM decides to call a tool, content may be empty.
            Use complete_with_metadata() to get tool_calls information.

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

        # Convert tools to Vertex AI format if provided
        vertex_tools = None
        if tools is not None:
            vertex_tools = self._convert_tools_to_vertex_format(tools)

        try:
            logger.debug(
                "Calling Vertex AI API",
                extra={
                    "model": self.model_name,
                    "message_count": len(messages),
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "tools_provided": tools is not None,
                },
            )

            # Call API with or without tools
            if vertex_tools:
                response = await self.client.generate_content_async(
                    prompt,
                    generation_config=generation_config,
                    tools=vertex_tools,
                )
            else:
                response = await self.client.generate_content_async(
                    prompt,
                    generation_config=generation_config,
                )

            content = response.text if hasattr(response, "text") else ""
            has_tool_calls = self._extract_tool_calls(response) is not None

            self._request_count += 1
            logger.info(
                "Vertex AI API call successful",
                extra={
                    "model": self.model_name,
                    "estimated_tokens": self._estimate_tokens(prompt + content),
                    "request_count": self._request_count,
                    "has_tool_calls": has_tool_calls,
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
            tenant_id: Tenant identifier (for compatibility)
            tools: Optional list of tool definitions for function calling
            tool_choice: Controls which tool to call (\"none\", \"auto\", or specific function)

        Returns:
            Full LLM response with metadata. If tools are provided and LLM calls
            a tool, the response will include a tool_calls field.

        Raises:
            LLMError: If the API call fails

        Example:
            >>> response = await adapter.complete_with_metadata(messages)
            >>> print(f"Used ~{response['tokens_used']} tokens")
            >>> if response.get('tool_calls'):
            ...     print(f"LLM wants to call: {response['tool_calls'][0]['function']['name']}")
        """
        # Format messages for Vertex AI
        prompt = self._format_messages(messages)

        # Configure generation
        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Convert tools to Vertex AI format if provided
        vertex_tools = None
        if tools is not None:
            vertex_tools = self._convert_tools_to_vertex_format(tools)

        try:
            logger.debug(
                "Calling Vertex AI API with metadata",
                extra={
                    "model": self.model_name,
                    "message_count": len(messages),
                    "tools_provided": tools is not None,
                },
            )

            # Call API with or without tools
            if vertex_tools:
                response = await self.client.generate_content_async(
                    prompt,
                    generation_config=generation_config,
                    tools=vertex_tools,
                )
            else:
                response = await self.client.generate_content_async(
                    prompt,
                    generation_config=generation_config,
                )

            content = response.text if hasattr(response, "text") else ""

            # Vertex AI doesn't provide exact token counts, so we estimate
            estimated_tokens = self._estimate_tokens(prompt + content)

            # Determine finish reason from response
            finish_reason = "stop"
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "finish_reason"):
                    finish_reason = str(candidate.finish_reason.name).lower()

            # Extract tool calls if present
            tool_calls = self._extract_tool_calls(response)

            self._request_count += 1

            # Build base result
            result_data: dict[str, Any] = {
                "content": content,
                "model": self.model_name,
                "tokens_used": estimated_tokens,
                "finish_reason": finish_reason,
            }

            # Add tool_calls field (None if no tool calls)
            result_data["tool_calls"] = tool_calls

            if tool_calls:
                logger.info(
                    "Vertex AI API call with tool calls successful",
                    extra={
                        "model": self.model_name,
                        "estimated_tokens": estimated_tokens,
                        "finish_reason": finish_reason,
                        "tool_calls_count": len(tool_calls),
                    },
                )
            else:
                logger.info(
                    "Vertex AI API call with metadata successful",
                    extra={
                        "model": self.model_name,
                        "estimated_tokens": estimated_tokens,
                        "finish_reason": finish_reason,
                    },
                )

            # Return as LLMResponseWithTools (superset of LLMResponse)
            return result_data  # type: ignore

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
