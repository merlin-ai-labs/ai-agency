"""Tests for LangGraph LLM adapter bridge.

This module tests the LangGraphLLMAdapter that bridges custom BaseAdapter
instances to LangGraph's expected Runnable interface.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.adapters.langgraph_adapter import LangGraphLLMAdapter
from app.core.exceptions import LLMError


@pytest.fixture
def mock_adapter():
    """Create a mock BaseAdapter instance."""
    adapter = Mock()
    adapter.provider_name = "openai"
    adapter.model_name = "gpt-4-turbo"
    adapter.complete = AsyncMock(return_value="Test response")
    return adapter


@pytest.fixture
def langgraph_adapter(mock_adapter):
    """Create LangGraphLLMAdapter instance."""
    return LangGraphLLMAdapter(adapter=mock_adapter)


class TestLangGraphLLMAdapterInit:
    """Test LangGraphLLMAdapter initialization."""

    def test_init_with_adapter(self, mock_adapter):
        """Test initialization with adapter."""
        adapter = LangGraphLLMAdapter(mock_adapter)

        assert adapter.adapter == mock_adapter
        assert adapter.temperature == 0.7
        assert adapter.max_tokens is None

    def test_init_with_custom_params(self, mock_adapter):
        """Test initialization with custom parameters."""
        adapter = LangGraphLLMAdapter(
            adapter=mock_adapter,
            temperature=0.5,
            max_tokens=1000,
        )

        assert adapter.temperature == 0.5
        assert adapter.max_tokens == 1000


class TestLangGraphLLMAdapterMessageConversion:
    """Test message format conversion."""

    def test_convert_human_message(self, langgraph_adapter):
        """Test conversion of HumanMessage."""
        messages = [HumanMessage(content="Hello")]
        result = langgraph_adapter._convert_langgraph_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_convert_ai_message(self, langgraph_adapter):
        """Test conversion of AIMessage."""
        messages = [AIMessage(content="Hi there")]
        result = langgraph_adapter._convert_langgraph_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Hi there"

    def test_convert_system_message(self, langgraph_adapter):
        """Test conversion of SystemMessage."""
        messages = [SystemMessage(content="You are a helpful assistant")]
        result = langgraph_adapter._convert_langgraph_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a helpful assistant"

    def test_convert_tool_message(self, langgraph_adapter):
        """Test conversion of ToolMessage."""
        messages = [ToolMessage(content='{"result": "success"}', tool_call_id="call_123")]
        result = langgraph_adapter._convert_langgraph_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == '{"result": "success"}'
        assert result[0]["tool_call_id"] == "call_123"

    def test_convert_multiple_messages(self, langgraph_adapter):
        """Test conversion of multiple message types."""
        messages = [
            SystemMessage(content="System"),
            HumanMessage(content="User"),
            AIMessage(content="Assistant"),
        ]
        result = langgraph_adapter._convert_langgraph_messages(messages)

        assert len(result) == 3
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"


class TestLangGraphLLMAdapterAsyncInvoke:
    """Test async invoke method."""

    @pytest.mark.asyncio
    async def test_ainvoke_success(self, langgraph_adapter, mock_adapter):
        """Test successful async invocation."""
        messages = [HumanMessage(content="Hello")]
        result = await langgraph_adapter.ainvoke(messages)

        assert isinstance(result, AIMessage)
        assert result.content == "Test response"
        mock_adapter.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_ainvoke_with_kwargs(self, langgraph_adapter, mock_adapter):
        """Test async invocation with custom parameters."""
        messages = [HumanMessage(content="Hello")]
        result = await langgraph_adapter.ainvoke(
            messages,
            temperature=0.5,
            max_tokens=100,
        )

        assert isinstance(result, AIMessage)
        call_kwargs = mock_adapter.complete.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_ainvoke_with_tools(self, langgraph_adapter, mock_adapter):
        """Test async invocation with tools."""
        messages = [HumanMessage(content="Hello")]
        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        await langgraph_adapter.ainvoke(messages, tools=tools)

        call_kwargs = mock_adapter.complete.call_args[1]
        assert call_kwargs["tools"] == tools

    @pytest.mark.asyncio
    async def test_ainvoke_error_handling(self, langgraph_adapter, mock_adapter):
        """Test error handling in async invocation."""
        mock_adapter.complete.side_effect = Exception("API Error")

        messages = [HumanMessage(content="Hello")]

        with pytest.raises(LLMError) as exc_info:
            await langgraph_adapter.ainvoke(messages)

        assert "LLM adapter call failed" in str(exc_info.value)
        assert exc_info.value.details["provider"] == "openai"


class TestLangGraphLLMAdapterRepr:
    """Test string representation."""

    def test_repr(self, langgraph_adapter):
        """Test string representation."""
        repr_str = repr(langgraph_adapter)

        assert "LangGraphLLMAdapter" in repr_str
        assert "openai" in repr_str
        assert "gpt-4-turbo" in repr_str
