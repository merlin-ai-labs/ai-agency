"""Core infrastructure module for AI Agency platform.

This module provides foundational components used throughout the application:
- Exception hierarchy for consistent error handling
- Decorators for retry logic, timeouts, logging, and performance measurement
- Type definitions and protocols for type safety
- Abstract base classes for tools, adapters, flows, and repositories

Import commonly used items from this module rather than from submodules.

Example:
    >>> from app.core import (
    ...     AIAgencyError,
    ...     DatabaseError,
    ...     retry,
    ...     timeout,
    ...     BaseTool,
    ...     BaseAdapter,
    ... )
"""

# Exceptions
# Base classes
from app.core.base import (
    BaseAdapter,
    BaseFlow,
    BaseRepository,
    BaseTool,
)

# Decorators
from app.core.decorators import (
    cache_result,
    log_execution,
    measure_time,
    retry,
    timeout,
    validate_input,
)
from app.core.exceptions import (
    AIAgencyError,
    AuthError,
    ConfigurationError,
    DatabaseError,
    FlowError,
    LLMError,
    RateLimitError,
    StorageError,
    ToolError,
    ValidationError,
)

# Type definitions and protocols
from app.core.types import (
    DocumentChunk,
    DocumentMetadata,
    EmbeddingProtocol,
    FlowProtocol,
    JSONDict,
    JSONValue,
    LLMProtocol,
    LLMResponse,
    Message,
    MessageHistory,
    MetadataDict,
    StorageProtocol,
    ToolInput,
    ToolOutput,
    ToolProtocol,
    VectorStoreProtocol,
)

__all__ = [
    # Exceptions
    "AIAgencyError",
    "AuthError",
    # Base classes
    "BaseAdapter",
    "BaseFlow",
    "BaseRepository",
    "BaseTool",
    "ConfigurationError",
    "DatabaseError",
    # Types
    "DocumentChunk",
    "DocumentMetadata",
    "EmbeddingProtocol",
    "FlowError",
    "FlowProtocol",
    "JSONDict",
    "JSONValue",
    "LLMError",
    "LLMProtocol",
    "LLMResponse",
    "Message",
    "MessageHistory",
    "MetadataDict",
    "RateLimitError",
    "StorageError",
    "StorageProtocol",
    "ToolError",
    "ToolInput",
    "ToolOutput",
    "ToolProtocol",
    "ValidationError",
    "VectorStoreProtocol",
    # Decorators
    "cache_result",
    "log_execution",
    "measure_time",
    "retry",
    "timeout",
    "validate_input",
]
