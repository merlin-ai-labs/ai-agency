"""LangGraph adapter bridge for custom LLM adapters.

This module provides adapters to bridge custom BaseAdapter implementations
to LangGraph's expected interfaces. It allows LangGraph to use our custom
LLM adapters without requiring migration to LangChain's abstractions.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig

from app.core.base import BaseAdapter
from app.core.exceptions import LLMError
from app.core.types import MessageHistory

logger = logging.getLogger(__name__)


class LangGraphLLMAdapter(Runnable):
    """Adapter to bridge custom BaseAdapter to LangGraph's Runnable interface.

    This adapter wraps a custom BaseAdapter instance and makes it compatible
    with LangGraph's expected Runnable protocol. It handles message format
    conversion between LangGraph's BaseMessage types and our custom message format.

    Attributes:
        adapter: The custom BaseAdapter instance to wrap.
        temperature: Default temperature for completions.
        max_tokens: Default max tokens for completions.

    Example:
        >>> from app.adapters.llm_factory import get_llm_adapter
        >>> custom_adapter = get_llm_adapter(provider="openai")
        >>> langgraph_adapter = LangGraphLLMAdapter(custom_adapter)
        >>> # Use in LangGraph graph
        >>> graph = StateGraph(...)
        >>> graph.add_node("llm_node", langgraph_adapter)
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize the LangGraph adapter.

        Args:
            adapter: Custom BaseAdapter instance to wrap.
            temperature: Default temperature for completions.
            max_tokens: Default max tokens for completions.
        """
        self.adapter = adapter
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info(
            f"Initialized LangGraphLLMAdapter for {adapter.provider_name}",
            extra={
                "provider": adapter.provider_name,
                "model": adapter.model_name,
            },
        )

    def invoke(
        self,
        input: list[BaseMessage],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Invoke the adapter with LangGraph messages (synchronous wrapper).

        This method is called by LangGraph synchronously. It wraps the async
        ainvoke method for compatibility.

        Args:
            input: List of LangGraph BaseMessage objects.
            config: Optional RunnableConfig (not used currently).
            **kwargs: Additional parameters (temperature, max_tokens, tools).

        Returns:
            AIMessage with the LLM response.

        Raises:
            LLMError: If adapter call fails.
            RuntimeError: If called in async context (use ainvoke instead).
        """
        import asyncio

        try:
            # Check if we're in an async context
            loop = asyncio.get_running_loop()
            # If we get here, there's a running loop - raise error
            msg = (
                "Cannot run synchronous invoke() in async context. "
                "Use ainvoke() instead or call from synchronous context."
            )
            raise RuntimeError(msg)
        except RuntimeError:
            # No running loop - safe to create new one
            return asyncio.run(self.ainvoke(input, config, **kwargs))

    async def ainvoke(
        self,
        input: list[BaseMessage],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Async invoke the adapter with LangGraph messages.

        Args:
            input: List of LangGraph BaseMessage objects.
            config: Optional RunnableConfig (not used currently).
            **kwargs: Additional parameters (temperature, max_tokens, tools).

        Returns:
            AIMessage with the LLM response.

        Raises:
            ValueError: If conversion fails.
            LLMError: If adapter call fails.
        """
        try:
            # Convert LangGraph messages to our format
            messages = self._convert_langgraph_messages(input)

            # Extract parameters from kwargs or use defaults
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            tools = kwargs.get("tools", None)
            tool_choice = kwargs.get("tool_choice", "auto")

            logger.debug(
                "Async invoking adapter",
                extra={
                    "provider": self.adapter.provider_name,
                    "message_count": len(messages),
                    "has_tools": tools is not None,
                },
            )

            # Call adapter
            response = await self.adapter.complete(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
            )

            # Convert response to AIMessage
            return AIMessage(content=response)

        except Exception as e:
            logger.exception(
                "Async adapter invocation failed",
                extra={
                    "provider": self.adapter.provider_name,
                    "error": str(e),
                },
            )
            # Wrap in LangGraph-compatible error
            raise LLMError(
                f"LLM adapter call failed: {str(e)}",
                details={"provider": self.adapter.provider_name},
                original_error=e,
            ) from e

    def _convert_langgraph_messages(self, messages: list[BaseMessage]) -> MessageHistory:
        """Convert LangGraph BaseMessage objects to our message format.

        Args:
            messages: List of LangGraph BaseMessage objects.

        Returns:
            List of messages in our format (dict with role and content).

        Raises:
            ValueError: If message type is unsupported.
        """
        result: MessageHistory = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                # Convert ToolMessage to tool role format
                result.append(
                    {
                        "role": "tool",
                        "content": msg.content,
                        "tool_call_id": getattr(msg, "tool_call_id", ""),
                    }
                )
            else:
                msg_type = type(msg).__name__
                logger.warning(f"Unsupported message type: {msg_type}, converting to user message")
                # Fallback: try to extract content
                content = getattr(msg, "content", str(msg))
                result.append({"role": "user", "content": content})

        return result

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return (
            f"LangGraphLLMAdapter("
            f"adapter={self.adapter.provider_name!r}, "
            f"model={self.adapter.model_name!r})"
        )
