"""Type definitions and protocols for AI Agency platform.

This module defines common type aliases and protocols (interfaces) used
throughout the application. Protocols provide structural subtyping for
better type safety without requiring inheritance.
"""

from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypedDict

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

# JSON-like data structures
JSONValue: TypeAlias = (
    str | int | float | bool | None | dict[str, "JSONValue"] | list["JSONValue"]
)
"""Type alias for JSON-compatible values."""

JSONDict: TypeAlias = dict[str, JSONValue]
"""Type alias for JSON objects (dictionaries)."""

MetadataDict: TypeAlias = dict[str, str | int | float | bool]
"""Type alias for metadata dictionaries with basic types."""


# Message and chat types
class Message(TypedDict):
    """Type for chat message structure.

    Used for LLM chat completions and conversation history.
    """

    role: str  # "user", "assistant", "system"
    content: str


MessageHistory: TypeAlias = list[Message]
"""Type alias for conversation history."""


# LLM response types
class LLMResponse(TypedDict):
    """Type for LLM API responses."""

    content: str
    model: str
    tokens_used: int
    finish_reason: str  # "stop", "length", "content_filter"


# Tool execution types
class ToolInput(TypedDict, total=False):
    """Type for tool input parameters.

    Using total=False allows optional fields.
    """

    query: str
    top_k: int
    filters: dict[str, Any]
    options: dict[str, Any]


class ToolOutput(TypedDict):
    """Type for tool execution results."""

    success: bool
    result: Any
    error: str | None
    metadata: dict[str, Any]


# Document types
class DocumentMetadata(TypedDict, total=False):
    """Type for document metadata.

    Contains optional metadata fields for documents.
    """

    title: str
    author: str
    created_at: str
    updated_at: str
    category: str
    tags: list[str]
    source: str


class DocumentChunk(TypedDict):
    """Type for document chunks used in RAG."""

    content: str
    chunk_index: int
    metadata: DocumentMetadata


# Protocols (Interfaces)


class LLMProtocol(Protocol):
    """Protocol for LLM provider implementations.

    Any class implementing these methods can be used as an LLM provider,
    regardless of inheritance. This enables duck typing with type safety.

    Example:
        >>> class OpenAIAdapter:
        ...     async def complete(
        ...         self,
        ...         messages: MessageHistory,
        ...         temperature: float = 0.7,
        ...         max_tokens: int | None = None
        ...     ) -> str:
        ...         # Implementation
        ...         pass
        >>>
        >>> # OpenAIAdapter satisfies LLMProtocol without inheriting from it
        >>> adapter: LLMProtocol = OpenAIAdapter()
    """

    async def complete(
        self,
        messages: MessageHistory,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a completion from messages.

        Args:
            messages: Conversation history.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.

        Returns:
            The generated completion text.

        Raises:
            LLMError: If the API call fails.
        """
        ...

    async def complete_with_metadata(
        self,
        messages: MessageHistory,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate completion with full response metadata.

        Args:
            messages: Conversation history.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.

        Returns:
            Full LLM response with metadata.

        Raises:
            LLMError: If the API call fails.
        """
        ...


class ToolProtocol(Protocol):
    """Protocol for tool implementations.

    Tools are callable components that perform specific tasks in agent flows.

    Example:
        >>> class DocumentRetriever:
        ...     async def execute(self, **kwargs: Any) -> ToolOutput:
        ...         query = kwargs.get("query", "")
        ...         # Retrieve documents
        ...         return {
        ...             "success": True,
        ...             "result": documents,
        ...             "error": None,
        ...             "metadata": {}
        ...         }
    """

    name: str
    """Unique name for the tool."""

    description: str
    """Human-readable description of what the tool does."""

    async def execute(self, **kwargs: Any) -> ToolOutput:
        """Execute the tool with provided arguments.

        Args:
            **kwargs: Tool-specific arguments.

        Returns:
            Tool execution result with success status and data.

        Raises:
            ToolError: If execution fails.
        """
        ...

    def validate_input(self, **kwargs: Any) -> bool:
        """Validate input parameters.

        Args:
            **kwargs: Tool input parameters to validate.

        Returns:
            True if input is valid, False otherwise.
        """
        ...


class StorageProtocol(Protocol):
    """Protocol for storage provider implementations.

    Defines interface for storing and retrieving data from various
    storage backends (local, GCS, S3, etc.).

    Example:
        >>> class GCSStorage:
        ...     async def save(self, key: str, data: bytes) -> str:
        ...         # Upload to GCS
        ...         return f"gs://bucket/{key}"
        ...
        ...     async def load(self, key: str) -> bytes:
        ...         # Download from GCS
        ...         return data
    """

    async def save(self, key: str, data: bytes) -> str:
        """Save data to storage.

        Args:
            key: Storage key/path for the data.
            data: Data to store as bytes.

        Returns:
            Storage URL or path where data was saved.

        Raises:
            StorageError: If save operation fails.
        """
        ...

    async def load(self, key: str) -> bytes:
        """Load data from storage.

        Args:
            key: Storage key/path to load.

        Returns:
            Retrieved data as bytes.

        Raises:
            StorageError: If load operation fails.
        """
        ...

    async def delete(self, key: str) -> bool:
        """Delete data from storage.

        Args:
            key: Storage key/path to delete.

        Returns:
            True if deletion was successful.

        Raises:
            StorageError: If delete operation fails.
        """
        ...

    async def exists(self, key: str) -> bool:
        """Check if data exists in storage.

        Args:
            key: Storage key/path to check.

        Returns:
            True if the key exists.
        """
        ...


class EmbeddingProtocol(Protocol):
    """Protocol for embedding provider implementations.

    Defines interface for generating text embeddings for RAG and
    semantic search.

    Example:
        >>> class OpenAIEmbeddings:
        ...     async def embed_text(self, text: str) -> list[float]:
        ...         # Generate embedding
        ...         return [0.1, 0.2, ...]
        ...
        ...     async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...         # Generate embeddings for multiple texts
        ...         return [[0.1, 0.2, ...], ...]
    """

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.

        Raises:
            LLMError: If embedding generation fails.
        """
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            LLMError: If embedding generation fails.
        """
        ...


class VectorStoreProtocol(Protocol):
    """Protocol for vector store implementations.

    Defines interface for storing and querying document embeddings.

    Example:
        >>> class PgVectorStore:
        ...     async def add_documents(
        ...         self,
        ...         documents: list[str],
        ...         embeddings: list[list[float]],
        ...         metadata: list[dict[str, Any]]
        ...     ) -> list[str]:
        ...         # Store in pgvector
        ...         return document_ids
        ...
        ...     async def similarity_search(
        ...         self,
        ...         query_embedding: list[float],
        ...         top_k: int = 5
        ...     ) -> list[dict[str, Any]]:
        ...         # Search similar vectors
        ...         return results
    """

    async def add_documents(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add documents with embeddings to the vector store.

        Args:
            documents: List of document texts.
            embeddings: Corresponding embedding vectors.
            metadata: Optional metadata for each document.

        Returns:
            List of document IDs.

        Raises:
            DatabaseError: If storage fails.
        """
        ...

    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar documents.

        Args:
            query_embedding: Query vector to search for.
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of matching documents with similarity scores.

        Raises:
            DatabaseError: If search fails.
        """
        ...


class FlowProtocol(Protocol):
    """Protocol for agent flow implementations.

    Flows orchestrate tools and LLMs to accomplish tasks.

    Example:
        >>> class AssessmentFlow:
        ...     async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        ...         # Execute flow steps
        ...         return results
        ...
        ...     async def get_status(self) -> str:
        ...         return self.status
    """

    name: str
    """Unique name for the flow."""

    description: str
    """Human-readable description of the flow."""

    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the flow.

        Args:
            input_data: Input data and parameters for the flow.

        Returns:
            Flow execution results.

        Raises:
            FlowError: If flow execution fails.
        """
        ...

    async def validate(self, input_data: dict[str, Any]) -> bool:
        """Validate input data for the flow.

        Args:
            input_data: Input to validate.

        Returns:
            True if input is valid.

        Raises:
            ValidationError: If validation fails.
        """
        ...


# Callback types
ProgressCallback: TypeAlias = (
    "Callable[[int, int, str], None] | Callable[[int, int, str], Awaitable[None]]"
)
"""Type alias for progress callback functions.

Signature: (current: int, total: int, message: str) -> None
"""

ErrorCallback: TypeAlias = "Callable[[Exception], None] | Callable[[Exception], Awaitable[None]]"
"""Type alias for error callback functions.

Signature: (error: Exception) -> None
"""


# Import for forward references
