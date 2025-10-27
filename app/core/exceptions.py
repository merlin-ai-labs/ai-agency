"""Custom exception hierarchy for AI Agency platform.

This module defines all custom exceptions used throughout the application,
providing consistent error handling and meaningful error messages.

All exceptions inherit from AIAgencyError to allow catching all application
errors while still distinguishing from system errors.
"""

from typing import Any


class AIAgencyError(Exception):
    """Base exception for all AI Agency application errors.

    All custom exceptions in the application should inherit from this class.
    This allows catching all application-level errors while distinguishing
    them from system errors (ValueError, TypeError, etc.).

    Attributes:
        message: Human-readable error message.
        details: Additional context about the error as a dictionary.
        original_error: The original exception if this was raised from another exception.
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message describing what went wrong.
            details: Additional context as key-value pairs (e.g., IDs, parameters).
            original_error: The original exception if this was raised from another.
        """
        self.message = message
        self.details = details or {}
        self.original_error = original_error
        super().__init__(self.message)

    def __repr__(self) -> str:
        """Return detailed string representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"details={self.details!r})"
        )


class DatabaseError(AIAgencyError):
    """Raised when database operations fail.

    Use this for any database-related errors including connection failures,
    query execution errors, transaction failures, etc.

    Example:
        >>> try:
        ...     result = await session.execute(query)
        ... except SQLAlchemyError as e:
        ...     raise DatabaseError(
        ...         "Failed to fetch user",
        ...         details={"user_id": user_id, "operation": "fetch"},
        ...         original_error=e
        ...     ) from e
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize DatabaseError.

        Args:
            message: Description of the database error.
            details: Context such as operation name, table, query parameters.
            original_error: The original database exception.
        """
        super().__init__(message, details, original_error)


class LLMError(AIAgencyError):
    """Raised when LLM API calls fail.

    Use this for errors related to LLM provider APIs including rate limits,
    invalid requests, timeouts, and API errors.

    Example:
        >>> try:
        ...     response = await openai_client.complete(prompt)
        ... except OpenAIError as e:
        ...     raise LLMError(
        ...         "OpenAI API call failed",
        ...         details={"provider": "openai", "model": "gpt-4"},
        ...         original_error=e
        ...     ) from e
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize LLMError.

        Args:
            message: Description of the LLM error.
            details: Context such as provider, model, prompt length.
            original_error: The original LLM provider exception.
        """
        super().__init__(message, details, original_error)


class FlowError(AIAgencyError):
    """Raised when flow execution fails.

    Use this for errors during agent flow execution including step failures,
    validation errors, and flow state issues.

    Example:
        >>> if not flow.can_transition_to(next_step):
        ...     raise FlowError(
        ...         "Invalid flow transition",
        ...         details={
        ...             "current_step": flow.current_step,
        ...             "attempted_step": next_step,
        ...             "flow_id": flow.id
        ...         }
        ...     )
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize FlowError.

        Args:
            message: Description of the flow error.
            details: Context such as flow ID, current step, failed step.
            original_error: The original exception that caused the flow failure.
        """
        super().__init__(message, details, original_error)


class ToolError(AIAgencyError):
    """Raised when tool execution fails.

    Use this for errors during tool execution including validation failures,
    execution errors, and tool configuration issues.

    Example:
        >>> try:
        ...     result = await tool.execute(**params)
        ... except Exception as e:
        ...     raise ToolError(
        ...         f"Tool execution failed: {tool.name}",
        ...         details={"tool": tool.name, "params": params},
        ...         original_error=e
        ...     ) from e
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize ToolError.

        Args:
            message: Description of the tool error.
            details: Context such as tool name, parameters, execution context.
            original_error: The original exception from tool execution.
        """
        super().__init__(message, details, original_error)


class ValidationError(AIAgencyError):
    """Raised when input validation fails.

    Use this for any input validation errors including invalid parameters,
    missing required fields, format errors, etc.

    Example:
        >>> if not document_id.startswith("doc_"):
        ...     raise ValidationError(
        ...         "Invalid document ID format",
        ...         details={
        ...             "document_id": document_id,
        ...             "expected_format": "doc_*"
        ...         }
        ...     )
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize ValidationError.

        Args:
            message: Description of the validation error.
            details: Context such as field name, invalid value, expected format.
            original_error: The original validation exception if any.
        """
        super().__init__(message, details, original_error)


class AuthError(AIAgencyError):
    """Raised when authentication fails.

    Use this for authentication-related errors including invalid credentials,
    expired tokens, and missing authentication.

    Example:
        >>> if not api_key:
        ...     raise AuthError(
        ...         "API key required",
        ...         details={"header": "X-API-Key"}
        ...     )
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize AuthError.

        Args:
            message: Description of the authentication error.
            details: Context such as auth method, required credentials.
            original_error: The original authentication exception if any.
        """
        super().__init__(message, details, original_error)


class RateLimitError(AIAgencyError):
    """Raised when rate limits are exceeded.

    Use this when API rate limits or request quotas are exceeded.

    Example:
        >>> if request_count > rate_limit:
        ...     raise RateLimitError(
        ...         "Rate limit exceeded",
        ...         details={
        ...             "limit": rate_limit,
        ...             "current": request_count,
        ...             "window": "1 minute"
        ...         }
        ...     )
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize RateLimitError.

        Args:
            message: Description of the rate limit error.
            details: Context such as limit, current count, time window.
            original_error: The original rate limit exception if any.
        """
        super().__init__(message, details, original_error)


class StorageError(AIAgencyError):
    """Raised when storage operations fail.

    Use this for errors related to file storage, cloud storage (GCS),
    and document storage operations.

    Example:
        >>> try:
        ...     await storage.upload(file_path, data)
        ... except Exception as e:
        ...     raise StorageError(
        ...         "Failed to upload file",
        ...         details={"file_path": file_path, "storage": "gcs"},
        ...         original_error=e
        ...     ) from e
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize StorageError.

        Args:
            message: Description of the storage error.
            details: Context such as file path, storage type, operation.
            original_error: The original storage exception.
        """
        super().__init__(message, details, original_error)


class ConfigurationError(AIAgencyError):
    """Raised when configuration is invalid or missing.

    Use this for errors related to application configuration including
    missing environment variables, invalid settings, etc.

    Example:
        >>> if not settings.DATABASE_URL:
        ...     raise ConfigurationError(
        ...         "DATABASE_URL not configured",
        ...         details={"env_var": "DATABASE_URL"}
        ...     )
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize ConfigurationError.

        Args:
            message: Description of the configuration error.
            details: Context such as config key, expected format.
            original_error: The original configuration exception if any.
        """
        super().__init__(message, details, original_error)
