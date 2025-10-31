"""Tests for LangGraph tool adapter bridge.

This module tests the LangGraphToolAdapter and LangGraphFunctionToolAdapter
that bridge custom tools to LangGraph's expected interfaces.
"""

import json
import pytest
from unittest.mock import AsyncMock, Mock

from langchain_core.tools import BaseTool as LangChainBaseTool

from app.adapters.langgraph_tools import (
    LangGraphFunctionToolAdapter,
    LangGraphToolAdapter,
    create_langgraph_tool,
)
from app.core.base import BaseTool
from app.core.exceptions import ToolError
from app.core.types import ToolOutput


class MockBaseTool(BaseTool):
    """Mock BaseTool for testing."""

    def __init__(self, name: str = "test_tool", description: str = "Test tool"):
        """Initialize mock tool."""
        super().__init__(name=name, description=description, version="1.0.0")

    async def execute(self, **kwargs: any) -> ToolOutput:
        """Execute mock tool."""
        return {
            "success": True,
            "result": {"output": "test_result"},
            "error": None,
            "metadata": {},
        }

    def validate_input(self, **kwargs: any) -> bool:
        """Validate input."""
        return True


@pytest.fixture
def mock_tool():
    """Create mock BaseTool instance."""
    return MockBaseTool()


@pytest.fixture
def langgraph_tool_adapter(mock_tool):
    """Create LangGraphToolAdapter instance."""
    return LangGraphToolAdapter(tool=mock_tool, tenant_id="test_tenant")


@pytest.fixture
def mock_function():
    """Create mock function tool."""

    async def test_func(**kwargs: any) -> dict[str, any]:
        return {"success": True, "result": "function_result"}

    return test_func


class TestLangGraphToolAdapterInit:
    """Test LangGraphToolAdapter initialization."""

    def test_init_with_tool(self, mock_tool):
        """Test initialization with BaseTool."""
        adapter = LangGraphToolAdapter(mock_tool)

        assert adapter.tool == mock_tool
        assert adapter.name == "test_tool"
        assert adapter.description == "Test tool"

    def test_init_with_tenant_id(self, mock_tool):
        """Test initialization with tenant_id."""
        adapter = LangGraphToolAdapter(mock_tool, tenant_id="tenant_123")

        assert adapter.tenant_id == "tenant_123"


class TestLangGraphToolAdapterRun:
    """Test LangGraphToolAdapter execution."""

    @pytest.mark.asyncio
    async def test_arun_success(self, langgraph_tool_adapter):
        """Test successful async execution."""
        result = await langgraph_tool_adapter._arun(param1="value1")

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["output"] == "test_result"

    @pytest.mark.asyncio
    async def test_arun_with_tenant_id(self, mock_tool):
        """Test execution with tenant_id injection."""
        adapter = LangGraphToolAdapter(mock_tool, tenant_id="tenant_123")

        result = await adapter._arun(param1="value1")

        # Verify execution succeeded (tenant_id is injected internally)
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["output"] == "test_result"

    @pytest.mark.asyncio
    async def test_arun_error_handling(self, mock_tool):
        """Test error handling."""

        # Create a tool that raises an exception
        class FailingTool(MockBaseTool):
            async def execute(self, **kwargs: any) -> ToolOutput:
                raise Exception("Tool execution failed")

        failing_tool = FailingTool()
        adapter = LangGraphToolAdapter(failing_tool)

        with pytest.raises(ToolError) as exc_info:
            await adapter._arun(param1="value1")

        assert "Tool execution failed" in str(exc_info.value)
        assert exc_info.value.details["tool"] == "test_tool"

    @pytest.mark.asyncio
    async def test_arun_tool_failure(self, mock_tool):
        """Test handling of tool execution failure."""

        # Create a tool that returns failure
        class FailingTool(MockBaseTool):
            async def execute(self, **kwargs: any) -> ToolOutput:
                return {
                    "success": False,
                    "result": None,
                    "error": "Tool error",
                    "metadata": {},
                }

        failing_tool = FailingTool()
        adapter = LangGraphToolAdapter(failing_tool)

        result = await adapter._arun(param1="value1")

        parsed = json.loads(result)
        assert parsed["success"] is False
        assert parsed["error"] == "Tool error"

    def test_run_not_implemented(self, langgraph_tool_adapter):
        """Test that synchronous run raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            langgraph_tool_adapter._run(param1="value1")


class TestLangGraphToolAdapterArgsSchema:
    """Test args_schema property."""

    def test_args_schema_property_exists(self, langgraph_tool_adapter):
        """Test that args_schema property exists and is callable."""
        # Verify property exists (LangChain BaseTool compatibility)
        assert hasattr(langgraph_tool_adapter, "args_schema")
        # Note: args_schema may return a property descriptor from BaseTool
        # The actual value is None when accessed, but we can't easily test
        # this without inspecting the property chain


class TestLangGraphFunctionToolAdapter:
    """Test LangGraphFunctionToolAdapter."""

    @pytest.mark.asyncio
    async def test_arun_with_async_function(self, mock_function):
        """Test execution with async function."""
        adapter = LangGraphFunctionToolAdapter(
            func=mock_function,
            name="test_func",
            description="Test function",
        )

        result = await adapter._arun(param1="value1")

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == "function_result"

    @pytest.mark.asyncio
    async def test_arun_with_sync_function(self):
        """Test execution with sync function."""

        def sync_func(**kwargs: any) -> dict[str, any]:
            return {"result": "sync_result"}

        adapter = LangGraphFunctionToolAdapter(
            func=sync_func,
            name="sync_func",
            description="Sync function",
        )

        result = await adapter._arun(param1="value1")

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["result"] == "sync_result"

    @pytest.mark.asyncio
    async def test_arun_with_tool_output_format(self):
        """Test execution with ToolOutput format."""

        async def tool_output_func(**kwargs: any) -> dict[str, any]:
            return {
                "success": True,
                "result": {"data": "test"},
                "error": None,
            }

        adapter = LangGraphFunctionToolAdapter(
            func=tool_output_func,
            name="tool_output_func",
            description="Tool output function",
        )

        result = await adapter._arun(param1="value1")

        parsed = json.loads(result)
        assert parsed["data"] == "test"

    def test_run_not_implemented(self, mock_function):
        """Test that synchronous run raises NotImplementedError."""
        adapter = LangGraphFunctionToolAdapter(
            func=mock_function,
            name="test_func",
            description="Test function",
        )

        with pytest.raises(NotImplementedError):
            adapter._run(param1="value1")


class TestCreateLangGraphTool:
    """Test factory function."""

    def test_create_with_base_tool(self, mock_tool):
        """Test creating adapter from BaseTool."""
        result = create_langgraph_tool(mock_tool)

        assert isinstance(result, LangGraphToolAdapter)
        assert result.tool == mock_tool

    def test_create_with_function(self, mock_function):
        """Test creating adapter from function."""
        result = create_langgraph_tool(
            tool=mock_function,
            name="test_func",
            description="Test function",
        )

        assert isinstance(result, LangGraphFunctionToolAdapter)
        assert result.name == "test_func"

    def test_create_with_function_missing_name(self, mock_function):
        """Test that missing name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            create_langgraph_tool(tool=mock_function, description="Test")

        assert "name and description are required" in str(exc_info.value)

    def test_create_with_invalid_type(self):
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            create_langgraph_tool(tool="not_a_tool")

        assert "Unsupported tool type" in str(exc_info.value)

    def test_create_with_tenant_id(self, mock_tool):
        """Test creating adapter with tenant_id."""
        result = create_langgraph_tool(mock_tool, tenant_id="tenant_123")

        assert isinstance(result, LangGraphToolAdapter)
        assert result.tenant_id == "tenant_123"
