"""LangGraph tool adapter bridge for custom tools.

This module provides adapters to bridge custom BaseTool instances and tool
functions to LangGraph's expected interfaces. It allows LangGraph to use our
custom tools without requiring migration to LangChain's abstractions.
"""

import json
import logging
from typing import Any, Callable

from langchain_core.tools import BaseTool as LangChainBaseTool

from app.core.base import BaseTool
from app.core.exceptions import ToolError

logger = logging.getLogger(__name__)


class LangGraphToolAdapter(LangChainBaseTool):
    """Adapter to bridge custom BaseTool to LangGraph's BaseTool interface.

    This adapter wraps a custom BaseTool instance and makes it compatible
    with LangGraph's expected tool interface. It handles input/output format
    conversion and preserves existing error handling.

    Attributes:
        tool: The custom BaseTool instance to wrap.
        tenant_id: Optional tenant ID for multi-tenancy context.

    Example:
        >>> from app.tools import get_tool_instance
        >>> custom_tool = get_tool_instance("parse_docs")
        >>> langgraph_tool = LangGraphToolAdapter(custom_tool, tenant_id="tenant_123")
        >>> # Use in LangGraph graph
        >>> graph = StateGraph(...)
        >>> graph.add_node("tool_node", langgraph_tool)
    """

    # Store tool reference in a way compatible with Pydantic
    _tool: BaseTool
    _tenant_id: str | None = None

    def __init__(
        self,
        tool: BaseTool,
        tenant_id: str | None = None,
    ) -> None:
        """Initialize the LangGraph tool adapter.

        Args:
            tool: Custom BaseTool instance to wrap.
            tenant_id: Optional tenant ID for multi-tenancy context.
        """
        # Initialize LangChain BaseTool with tool metadata first
        super().__init__(
            name=tool.name,
            description=tool.description,
        )

        # Store custom attributes after initialization
        object.__setattr__(self, "_tool", tool)
        object.__setattr__(self, "_tenant_id", tenant_id)

        logger.info(
            f"Initialized LangGraphToolAdapter for {tool.name}",
            extra={
                "tool_name": tool.name,
                "tool_version": tool.version,
                "tenant_id": tenant_id,
            },
        )

    @property
    def tool(self) -> BaseTool:
        """Get the wrapped tool instance."""
        return self._tool

    @property
    def tenant_id(self) -> str | None:
        """Get the tenant ID."""
        return self._tenant_id

    def _run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Synchronous run method (not used, but required by LangChain).

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            JSON string of tool output.

        Raises:
            NotImplementedError: This method should not be called directly.
        """
        msg = "LangGraphToolAdapter does not support synchronous execution"
        raise NotImplementedError(msg)

    async def _arun(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Async run method called by LangGraph.

        Args:
            *args: Positional arguments (not used).
            **kwargs: Keyword arguments for tool execution.

        Returns:
            JSON string of tool output.

        Raises:
            ToolError: If tool execution fails.
        """
        try:
            # Add tenant_id to kwargs if provided
            if self._tenant_id and "tenant_id" not in kwargs:
                kwargs["tenant_id"] = self._tenant_id

            logger.debug(
                f"Executing tool via LangGraph adapter: {self._tool.name}",
                extra={
                    "tool_name": self._tool.name,
                    "kwargs": {k: v for k, v in kwargs.items() if k != "tenant_id"},
                },
            )

            # Execute tool
            result = await self._tool.run(**kwargs)

            # Convert result to JSON string (LangGraph expects string output)
            if result["success"]:
                output = json.dumps(result["result"])
            else:
                # Include error in output
                output = json.dumps(
                    {
                        "success": False,
                        "error": result.get("error"),
                        "result": result.get("result"),
                    }
                )

            logger.debug(
                f"Tool execution completed: {self._tool.name}",
                extra={
                    "tool_name": self._tool.name,
                    "success": result["success"],
                },
            )

            return output

        except Exception as e:
            logger.exception(
                f"Tool execution failed: {self._tool.name}",
                extra={
                    "tool_name": self._tool.name,
                    "error": str(e),
                },
            )
            # Wrap in ToolError for consistency
            raise ToolError(
                f"Tool execution failed: {self._tool.name}",
                details={"tool": self._tool.name, "kwargs": kwargs},
                original_error=e,
            ) from e

    @property
    def args_schema(self) -> type | None:
        """Get tool arguments schema for LangGraph.

        Returns:
            Pydantic model class for tool arguments, or None for open schema.

        Note:
            LangGraph uses this to generate tool definitions for LLMs.
            Returning None allows any arguments, which is appropriate for
            our tools that handle validation internally.
        """
        # Return None to allow any arguments
        # Tools handle their own validation via validate_input()
        return None

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return f"LangGraphToolAdapter(tool={self._tool.name!r}, version={self._tool.version!r})"


class LangGraphFunctionToolAdapter(LangChainBaseTool):
    """Adapter for function-based tools (not BaseTool classes).

    This adapter wraps a callable function and makes it compatible with
    LangGraph's tool interface. Useful for legacy tools that are functions
    rather than BaseTool instances.

    Attributes:
        func: The callable function to wrap.
        name: Tool name.
        description: Tool description.
        tenant_id: Optional tenant ID for multi-tenancy context.

    Example:
        >>> from app.tools.weather.v1 import get_weather
        >>> langgraph_tool = LangGraphFunctionToolAdapter(
        ...     func=get_weather,
        ...     name="get_weather",
        ...     description="Get current weather for a location",
        ...     tenant_id="tenant_123"
        ... )
    """

    # Store function reference in a way compatible with Pydantic
    _func: Callable[..., Any]
    _tenant_id: str | None = None

    def __init__(
        self,
        func: Callable[..., Any],
        name: str,
        description: str,
        tenant_id: str | None = None,
    ) -> None:
        """Initialize the function tool adapter.

        Args:
            func: Callable function to wrap.
            name: Tool name.
            description: Tool description.
            tenant_id: Optional tenant ID for multi-tenancy context.
        """
        # Initialize LangChain BaseTool first
        super().__init__(
            name=name,
            description=description,
        )

        # Store custom attributes after initialization
        object.__setattr__(self, "_func", func)
        object.__setattr__(self, "_tenant_id", tenant_id)

        logger.info(
            f"Initialized LangGraphFunctionToolAdapter for {name}",
            extra={
                "tool_name": name,
                "tenant_id": tenant_id,
            },
        )

    def _run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Synchronous run method (not used, but required by LangChain).

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            JSON string of tool output.

        Raises:
            NotImplementedError: This method should not be called directly.
        """
        msg = "LangGraphFunctionToolAdapter does not support synchronous execution"
        raise NotImplementedError(msg)

    async def _arun(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Async run method called by LangGraph.

        Args:
            *args: Positional arguments (not used).
            **kwargs: Keyword arguments for tool execution.

        Returns:
            JSON string of tool output.

        Raises:
            ToolError: If tool execution fails.
        """
        try:
            # Add tenant_id to kwargs if provided
            if self._tenant_id and "tenant_id" not in kwargs:
                kwargs["tenant_id"] = self._tenant_id

            logger.debug(
                f"Executing function tool via LangGraph adapter: {self.name}",
                extra={
                    "tool_name": self.name,
                    "kwargs": {k: v for k, v in kwargs.items() if k != "tenant_id"},
                },
            )

            # Execute function
            if hasattr(self._func, "__call__"):
                # Check if function is async
                import inspect

                if inspect.iscoroutinefunction(self._func):
                    result = await self._func(**kwargs)
                else:
                    # Sync function - wrap in awaitable
                    import asyncio

                    result = await asyncio.to_thread(self._func, **kwargs)
            else:
                msg = f"Function {self.name} is not callable"
                raise ValueError(msg)

            # Convert result to JSON string
            if isinstance(result, dict):
                # Check if result matches ToolOutput format
                if "success" in result:
                    if result["success"]:
                        output = json.dumps(result.get("result", result))
                    else:
                        output = json.dumps(
                            {
                                "success": False,
                                "error": result.get("error"),
                                "result": result.get("result"),
                            }
                        )
                else:
                    # Regular dict result
                    output = json.dumps(result)
            elif isinstance(result, str):
                output = result
            else:
                # Try to serialize
                output = json.dumps(result)

            logger.debug(
                f"Function tool execution completed: {self.name}",
                extra={
                    "tool_name": self.name,
                },
            )

            return output

        except Exception as e:
            logger.exception(
                f"Function tool execution failed: {self.name}",
                extra={
                    "tool_name": self.name,
                    "error": str(e),
                },
            )
            # Wrap in ToolError for consistency
            raise ToolError(
                f"Function tool execution failed: {self.name}",
                details={"tool": self.name, "kwargs": kwargs},
                original_error=e,
            ) from e

    @property
    def args_schema(self) -> type | None:
        """Get tool arguments schema for LangGraph.

        Returns:
            Pydantic model class for tool arguments, or None for open schema.
        """
        # Return None to allow any arguments
        # Function tools handle their own validation
        return None

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return f"LangGraphFunctionToolAdapter(func={self.name!r}, name={self.name!r})"


def create_langgraph_tool(
    tool: BaseTool | Callable[..., Any],
    tenant_id: str | None = None,
    name: str | None = None,
    description: str | None = None,
) -> LangChainBaseTool:
    """Factory function to create LangGraph tool adapter.

    Automatically detects whether tool is a BaseTool instance or a function
    and creates the appropriate adapter.

    Args:
        tool: BaseTool instance or callable function.
        tenant_id: Optional tenant ID for multi-tenancy context.
        name: Tool name (required if tool is a function).
        description: Tool description (required if tool is a function).

    Returns:
        LangGraph-compatible tool adapter.

    Raises:
        ValueError: If tool is a function but name/description are missing.
    """
    if isinstance(tool, BaseTool):
        return LangGraphToolAdapter(tool, tenant_id=tenant_id)
    elif callable(tool):
        if not name or not description:
            msg = "name and description are required for function-based tools"
            raise ValueError(msg)
        return LangGraphFunctionToolAdapter(
            func=tool,
            name=name,
            description=description,
            tenant_id=tenant_id,
        )
    else:
        msg = f"Unsupported tool type: {type(tool)}"
        raise ValueError(msg)
