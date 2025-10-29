"""Abstract base classes for AI Agency platform.

This module defines abstract base classes that establish contracts for
core components. All implementations should inherit from these base classes
to ensure consistency and enable polymorphism.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from app.core.types import (
    LLMResponse,
    LLMResponseWithTools,
    Message,
    MessageHistory,
    ToolDefinition,
    ToolOutput,
)

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Abstract base class for all tools.

    Tools are executable components that perform specific tasks within
    agent flows. Each tool must implement the execute method and provide
    validation logic.

    Attributes:
        name: Unique identifier for the tool.
        description: Human-readable description of what the tool does.
        version: Tool version for tracking changes.

    Example:
        >>> class DocumentRetriever(BaseTool):
        ...     def __init__(self):
        ...         super().__init__(
        ...             name="document_retriever",
        ...             description="Retrieve relevant documents",
        ...             version="1.0.0"
        ...         )
        ...
        ...     async def execute(self, **kwargs: Any) -> ToolOutput:
        ...         query = kwargs.get("query", "")
        ...         results = await self._search_documents(query)
        ...         return {
        ...             "success": True,
        ...             "result": results,
        ...             "error": None,
        ...             "metadata": {"count": len(results)}
        ...         }
        ...
        ...     def validate_input(self, **kwargs: Any) -> bool:
        ...         return "query" in kwargs and isinstance(kwargs["query"], str)
    """

    def __init__(self, name: str, description: str, version: str = "1.0.0") -> None:
        """Initialize the tool.

        Args:
            name: Unique tool identifier (lowercase with underscores).
            description: Clear description of tool functionality.
            version: Semantic version string.
        """
        self.name = name
        self.description = description
        self.version = version
        self._execution_count = 0
        logger.debug(f"Initialized tool: {name} v{version}")

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolOutput:
        """Execute the tool with provided arguments.

        This is the main method that performs the tool's function. All
        implementations must be async to support I/O operations.

        Args:
            **kwargs: Tool-specific input parameters.

        Returns:
            Standardized tool output with success status, result, and metadata.

        Raises:
            ToolError: If execution fails.
            ValidationError: If input validation fails.
        """

    @abstractmethod
    def validate_input(self, **kwargs: Any) -> bool:
        """Validate input parameters before execution.

        Implement this to check required parameters and validate
        parameter types and values.

        Args:
            **kwargs: Input parameters to validate.

        Returns:
            True if input is valid, False otherwise.
        """

    async def run(self, **kwargs: Any) -> ToolOutput:
        """Run the tool with validation.

        This is a convenience wrapper that validates input before
        executing the tool. Use this instead of calling execute directly.

        Args:
            **kwargs: Tool input parameters.

        Returns:
            Tool execution result.

        Raises:
            ValidationError: If input validation fails.
            ToolError: If execution fails.
        """
        from app.core.exceptions import ValidationError

        if not self.validate_input(**kwargs):
            msg = f"Invalid input for tool {self.name}"
            raise ValidationError(
                msg,
                details={"tool": self.name, "input": kwargs},
            )

        logger.info(f"Executing tool: {self.name}", extra={"tool": self.name, "input": kwargs})

        result = await self.execute(**kwargs)
        self._execution_count += 1

        logger.info(
            f"Tool execution completed: {self.name}",
            extra={
                "tool": self.name,
                "success": result["success"],
                "execution_count": self._execution_count,
            },
        )

        return result

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return f"{self.__class__.__name__}(name={self.name!r}, version={self.version!r})"


class BaseAdapter(ABC):
    """Abstract base class for LLM adapters.

    Adapters provide a unified interface for different LLM providers
    (OpenAI, Vertex AI, etc.). Each adapter handles provider-specific
    authentication, request formatting, and response parsing.

    Attributes:
        provider_name: Name of the LLM provider (e.g., "openai").
        model_name: Default model to use for completions.

    Example:
        >>> class OpenAIAdapter(BaseAdapter):
        ...     def __init__(self, api_key: str, model: str = "gpt-4"):
        ...         super().__init__(provider_name="openai", model_name=model)
        ...         self.client = OpenAI(api_key=api_key)
        ...
        ...     async def complete(
        ...         self,
        ...         messages: MessageHistory,
        ...         temperature: float = 0.7,
        ...         max_tokens: int | None = None
        ...     ) -> str:
        ...         response = await self.client.chat.completions.create(
        ...             model=self.model_name,
        ...             messages=messages,
        ...             temperature=temperature,
        ...             max_tokens=max_tokens
        ...         )
        ...         return response.choices[0].message.content
    """

    def __init__(self, provider_name: str, model_name: str) -> None:
        """Initialize the adapter.

        Args:
            provider_name: Name of the LLM provider.
            model_name: Default model identifier.
        """
        self.provider_name = provider_name
        self.model_name = model_name
        self._request_count = 0
        logger.debug(f"Initialized adapter: {provider_name} with model {model_name}")

    @abstractmethod
    async def complete(
        self,
        messages: MessageHistory,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> str:
        """Generate a completion from messages.

        Args:
            messages: Conversation history as a list of messages.
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
            max_tokens: Maximum tokens to generate (None = model default).
            tools: Optional list of tool definitions for function calling.
            tool_choice: Controls which tool to call. Can be "none", "auto",
                or {"type": "function", "function": {"name": "tool_name"}}.

        Returns:
            The generated completion text.

        Raises:
            LLMError: If the API call fails.

        Note:
            When tools are provided and the LLM decides to call a tool,
            the content may be empty. Use complete_with_metadata() to get
            tool_calls information.

        Example:
            >>> adapter = get_llm_adapter()
            >>> tools = [{
            ...     "type": "function",
            ...     "function": {
            ...         "name": "get_weather",
            ...         "description": "Get weather for a location",
            ...         "parameters": {
            ...             "type": "object",
            ...             "properties": {
            ...                 "location": {"type": "string"}
            ...             },
            ...             "required": ["location"]
            ...         }
            ...     }
            ... }]
            >>> response = await adapter.complete(
            ...     messages=[{"role": "user", "content": "Weather in Paris?"}],
            ...     tools=tools
            ... )
        """

    @abstractmethod
    async def complete_with_metadata(
        self,
        messages: MessageHistory,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMResponse | LLMResponseWithTools:
        """Generate completion with full response metadata.

        Use this when you need token counts, finish reasons, and other
        metadata beyond just the completion text. This is the recommended
        method when using tool calling.

        Args:
            messages: Conversation history.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            tools: Optional list of tool definitions for function calling.
            tool_choice: Controls which tool to call. Can be "none", "auto",
                or {"type": "function", "function": {"name": "tool_name"}}.

        Returns:
            Full LLM response with metadata. If tools are provided and the LLM
            calls a tool, the response will include a tool_calls field with the
            list of tool calls to execute.

        Raises:
            LLMError: If the API call fails.

        Example with tool calling:
            >>> adapter = get_llm_adapter()
            >>> tools = [{
            ...     "type": "function",
            ...     "function": {
            ...         "name": "get_weather",
            ...         "description": "Get current weather",
            ...         "parameters": {
            ...             "type": "object",
            ...             "properties": {
            ...                 "location": {"type": "string"}
            ...             },
            ...             "required": ["location"]
            ...         }
            ...     }
            ... }]
            >>> response = await adapter.complete_with_metadata(
            ...     messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            ...     tools=tools
            ... )
            >>> if response.get("tool_calls"):
            ...     # LLM wants to call a tool
            ...     for tool_call in response["tool_calls"]:
            ...         tool_name = tool_call["function"]["name"]
            ...         tool_args = json.loads(tool_call["function"]["arguments"])
            ...         # Execute the tool and return results...
        """

    async def create_message(self, role: str, content: str) -> Message:
        """Create a properly formatted message.

        Helper method to create messages with the correct structure.

        Args:
            role: Message role ("user", "assistant", "system").
            content: Message content.

        Returns:
            Formatted message dictionary.

        Raises:
            ValidationError: If role is invalid.
        """
        from app.core.exceptions import ValidationError

        valid_roles = {"user", "assistant", "system"}
        if role not in valid_roles:
            msg = f"Invalid message role: {role}"
            raise ValidationError(
                msg,
                details={"role": role, "valid_roles": list(valid_roles)},
            )

        return Message(role=role, content=content)

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"provider={self.provider_name!r}, "
            f"model={self.model_name!r})"
        )


class BaseFlow(ABC):
    """Abstract base class for agent flows.

    Flows orchestrate tools and LLMs to accomplish complex tasks. Each
    flow defines a series of steps that work together to achieve a goal.

    Attributes:
        name: Unique identifier for the flow.
        description: Human-readable description of what the flow does.
        version: Flow version for tracking changes.

    Example:
        >>> class AssessmentFlow(BaseFlow):
        ...     def __init__(self, llm: BaseAdapter, tools: dict[str, BaseTool]):
        ...         super().__init__(
        ...             name="assessment_flow",
        ...             description="Assess organization AI maturity",
        ...             version="1.0.0"
        ...         )
        ...         self.llm = llm
        ...         self.tools = tools
        ...
        ...     async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        ...         # Step 1: Parse documents
        ...         docs = await self.tools["parse_docs"].run(**input_data)
        ...
        ...         # Step 2: Generate assessment
        ...         assessment = await self.llm.complete([
        ...             {"role": "user", "content": f"Assess: {docs}"}
        ...         ])
        ...
        ...         return {"assessment": assessment, "status": "completed"}
        ...
        ...     async def validate(self, input_data: dict[str, Any]) -> bool:
        ...         return "documents" in input_data
    """

    def __init__(self, name: str, description: str, version: str = "1.0.0") -> None:
        """Initialize the flow.

        Args:
            name: Unique flow identifier (lowercase with underscores).
            description: Clear description of flow functionality.
            version: Semantic version string.
        """
        self.name = name
        self.description = description
        self.version = version
        self.status = "initialized"
        self._execution_count = 0
        logger.debug(f"Initialized flow: {name} v{version}")

    @abstractmethod
    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the flow.

        This is the main method that orchestrates the flow steps. All
        implementations must be async to support I/O operations.

        Args:
            input_data: Input data and parameters for the flow.

        Returns:
            Flow execution results as a dictionary.

        Raises:
            FlowError: If flow execution fails.
            ValidationError: If input validation fails.
        """

    @abstractmethod
    async def validate(self, input_data: dict[str, Any]) -> bool:
        """Validate input data before execution.

        Implement this to check required fields and validate data
        structure and values.

        Args:
            input_data: Input data to validate.

        Returns:
            True if input is valid, False otherwise.
        """

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the flow with validation.

        This is a convenience wrapper that validates input and manages
        flow status. Use this instead of calling run directly.

        Args:
            input_data: Flow input data.

        Returns:
            Flow execution result.

        Raises:
            ValidationError: If input validation fails.
            FlowError: If execution fails.
        """
        from app.core.exceptions import FlowError, ValidationError

        if not await self.validate(input_data):
            msg = f"Invalid input for flow {self.name}"
            raise ValidationError(
                msg,
                details={"flow": self.name, "input": input_data},
            )

        logger.info(f"Starting flow: {self.name}", extra={"flow": self.name})

        self.status = "running"

        try:
            result = await self.run(input_data)
            self.status = "completed"
            self._execution_count += 1

            logger.info(
                f"Flow completed successfully: {self.name}",
                extra={
                    "flow": self.name,
                    "execution_count": self._execution_count,
                },
            )

            return result

        except Exception as e:
            self.status = "failed"
            logger.exception(
                f"Flow failed: {self.name}",
                extra={"flow": self.name, "error": str(e)},
            )
            msg = f"Flow execution failed: {self.name}"
            raise FlowError(
                msg,
                details={"flow": self.name, "error": str(e)},
                original_error=e,
            ) from e

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"version={self.version!r}, "
            f"status={self.status!r})"
        )


class BaseRepository(ABC):
    """Abstract base class for data repositories.

    Repositories encapsulate data access logic and provide a clean
    interface for CRUD operations. Use repositories to abstract database
    operations from business logic.

    Example:
        >>> class DocumentRepository(BaseRepository):
        ...     async def get_by_id(self, id: str) -> Document | None:
        ...         async with self.get_session() as session:
        ...             result = await session.execute(
        ...                 select(Document).where(Document.id == id)
        ...             )
        ...             return result.scalar_one_or_none()
        ...
        ...     async def create(self, data: dict[str, Any]) -> Document:
        ...         async with self.get_session() as session:
        ...             doc = Document(**data)
        ...             session.add(doc)
        ...             await session.commit()
        ...             return doc
    """

    @abstractmethod
    async def get_by_id(self, id: str) -> Any | None:
        """Retrieve a single entity by ID.

        Args:
            id: Entity identifier.

        Returns:
            Entity if found, None otherwise.

        Raises:
            DatabaseError: If database operation fails.
        """

    @abstractmethod
    async def create(self, data: dict[str, Any]) -> Any:
        """Create a new entity.

        Args:
            data: Entity data as dictionary.

        Returns:
            Created entity.

        Raises:
            DatabaseError: If creation fails.
            ValidationError: If data is invalid.
        """

    @abstractmethod
    async def update(self, id: str, data: dict[str, Any]) -> Any | None:
        """Update an existing entity.

        Args:
            id: Entity identifier.
            data: Updated entity data.

        Returns:
            Updated entity if found, None otherwise.

        Raises:
            DatabaseError: If update fails.
            ValidationError: If data is invalid.
        """

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete an entity by ID.

        Args:
            id: Entity identifier.

        Returns:
            True if entity was deleted, False if not found.

        Raises:
            DatabaseError: If deletion fails.
        """

    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
    ) -> list[Any]:
        """List entities with pagination and filtering.

        This is an optional method that repositories can override.

        Args:
            limit: Maximum number of entities to return.
            offset: Number of entities to skip.
            filters: Optional filter criteria.

        Returns:
            List of entities.

        Raises:
            DatabaseError: If query fails.
        """
        msg = "list_all not implemented for this repository"
        raise NotImplementedError(msg)
