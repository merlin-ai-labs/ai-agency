"""Production-ready decorators for common patterns.

This module provides reusable decorators for retry logic, timeout handling,
logging, and performance measurement using the tenacity library for
robust retry mechanisms.
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)

# Type variables for generic decorators
P = ParamSpec("P")
T = TypeVar("T")

logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    backoff_type: str = "exponential",
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Retry decorator with configurable backoff strategy.

    Provides robust retry logic with exponential or fixed backoff for
    handling transient failures in async operations.

    Args:
        max_attempts: Maximum number of retry attempts (including initial call).
        backoff_type: Backoff strategy - "exponential" or "fixed".
        min_wait: Minimum wait time between retries in seconds.
        max_wait: Maximum wait time between retries in seconds.
        exceptions: Tuple of exception types to retry on.

    Returns:
        Decorated async function with retry logic.

    Example:
        >>> from app.core.exceptions import LLMError
        >>>
        >>> @retry(max_attempts=3, backoff_type="exponential", exceptions=(LLMError,))
        ... async def call_llm(prompt: str) -> str:
        ...     # May raise LLMError on transient failures
        ...     return await llm_client.complete(prompt)

    Note:
        Uses tenacity library for retry logic. The function will be retried
        on specified exceptions with the configured backoff strategy.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Configure wait strategy based on backoff type
            if backoff_type == "exponential":
                wait_strategy = wait_exponential(
                    min=min_wait,
                    max=max_wait,
                )
            else:
                wait_strategy = wait_fixed(min_wait)

            # Configure retry behavior
            retry_config = AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_strategy,
                retry=retry_if_exception_type(exceptions),
                reraise=True,
            )

            try:
                async for attempt in retry_config:
                    with attempt:
                        logger.debug(
                            f"Executing {func.__name__} (attempt {attempt.retry_state.attempt_number})",
                            extra={
                                "function": func.__name__,
                                "attempt": attempt.retry_state.attempt_number,
                                "max_attempts": max_attempts,
                            },
                        )
                        return await func(*args, **kwargs)

            except RetryError as e:
                logger.exception(
                    f"{func.__name__} failed after {max_attempts} attempts",
                    extra={
                        "function": func.__name__,
                        "max_attempts": max_attempts,
                        "last_exception": str(e.last_attempt.exception()),
                    },
                )
                raise

        return wrapper

    return decorator


def timeout(seconds: float) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Timeout decorator for async functions.

    Raises asyncio.TimeoutError if the function takes longer than the
    specified timeout.

    Args:
        seconds: Maximum execution time in seconds.

    Returns:
        Decorated async function with timeout.

    Example:
        >>> @timeout(seconds=30.0)
        ... async def process_document(doc_id: str) -> ProcessingResult:
        ...     # Will timeout if processing takes > 30 seconds
        ...     return await heavy_processing(doc_id)

    Raises:
        asyncio.TimeoutError: If function execution exceeds timeout.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds,
                )
            except TimeoutError:
                logger.exception(
                    f"{func.__name__} timed out after {seconds} seconds",
                    extra={
                        "function": func.__name__,
                        "timeout": seconds,
                    },
                )
                raise

        return wrapper

    return decorator


def log_execution(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator to log function execution with arguments and results.

    Logs function calls with arguments (sanitized) and execution results
    at DEBUG level. Logs exceptions at ERROR level.

    Args:
        func: The async function to decorate.

    Returns:
        Decorated async function with execution logging.

    Example:
        >>> @log_execution
        ... async def fetch_user(user_id: str) -> User:
        ...     return await db.get_user(user_id)
        # Logs: "Executing fetch_user with user_id=..."
        # Logs: "fetch_user completed in X.XXs"

    Note:
        Sensitive arguments should be filtered in production. Consider
        adding argument filtering for passwords, tokens, etc.
    """

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        # Sanitize arguments for logging (remove sensitive data)
        safe_kwargs = {
            k: "***REDACTED***" if k in {"password", "token", "api_key", "secret"} else v
            for k, v in kwargs.items()
        }

        logger.debug(
            f"Executing {func.__name__}",
            extra={
                "function": func.__name__,
                "args_count": len(args),
                "kwargs": safe_kwargs,
            },
        )

        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time

            logger.debug(
                f"{func.__name__} completed successfully",
                extra={
                    "function": func.__name__,
                    "execution_time": f"{execution_time:.3f}s",
                },
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time

            logger.exception(
                f"{func.__name__} failed with {type(e).__name__}",
                extra={
                    "function": func.__name__,
                    "execution_time": f"{execution_time:.3f}s",
                    "exception": type(e).__name__,
                    "error_message": str(e),
                },
            )
            raise

    return wrapper


def measure_time(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator to measure and log function execution time.

    Measures execution time and logs at INFO level. Useful for
    performance monitoring and optimization.

    Args:
        func: The async function to decorate.

    Returns:
        Decorated async function with time measurement.

    Example:
        >>> @measure_time
        ... async def process_large_dataset(data: list[dict]) -> ProcessingResult:
        ...     # Long-running operation
        ...     return await process(data)
        # Logs: "process_large_dataset took 5.234s"
    """

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start_time = time.time()

        try:
            return await func(*args, **kwargs)

        finally:
            execution_time = time.time() - start_time

            # Log warning if execution is slow (> 5 seconds)
            if execution_time > 5.0:
                logger.warning(
                    f"{func.__name__} took {execution_time:.3f}s (slow execution)",
                    extra={
                        "function": func.__name__,
                        "execution_time": execution_time,
                        "threshold": 5.0,
                    },
                )
            else:
                logger.info(
                    f"{func.__name__} took {execution_time:.3f}s",
                    extra={
                        "function": func.__name__,
                        "execution_time": execution_time,
                    },
                )

    return wrapper


def validate_input(
    validation_func: Callable[[Any], bool],
    error_message: str = "Invalid input",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for input validation.

    Validates function arguments using a custom validation function.

    Args:
        validation_func: Function that takes arguments and returns True if valid.
        error_message: Error message to use in ValidationError if validation fails.

    Returns:
        Decorated async function with input validation.

    Example:
        >>> def validate_doc_id(doc_id: str) -> bool:
        ...     return doc_id.startswith("doc_") and len(doc_id) > 4
        >>>
        >>> @validate_input(
        ...     validation_func=lambda kwargs: validate_doc_id(kwargs.get("doc_id", "")),
        ...     error_message="Invalid document ID format"
        ... )
        ... async def process_document(doc_id: str) -> ProcessingResult:
        ...     return await process(doc_id)

    Raises:
        ValidationError: If validation_func returns False.
    """
    from app.core.exceptions import ValidationError

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Create a dict of all arguments for validation
            all_args = dict(kwargs)

            # Validate input
            if not validation_func(all_args):
                logger.warning(
                    f"Validation failed for {func.__name__}",
                    extra={
                        "function": func.__name__,
                        "error": error_message,
                    },
                )
                raise ValidationError(error_message)

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def cache_result(ttl_seconds: int | None = None) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Simple cache decorator for async functions.

    Caches function results based on arguments. Useful for expensive
    operations with deterministic results.

    Args:
        ttl_seconds: Time-to-live for cache entries. None means no expiration.

    Returns:
        Decorated async function with caching.

    Example:
        >>> @cache_result(ttl_seconds=300)  # 5 minute cache
        ... async def fetch_config(key: str) -> str:
        ...     # Expensive config lookup
        ...     return await expensive_lookup(key)

    Note:
        This is a simple in-memory cache. For production use cases with
        multiple workers, consider using Redis or similar distributed cache.
    """
    cache: dict[str, tuple[T, float]] = {}

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Create cache key from arguments
            cache_key = f"{func.__name__}:{args!s}:{sorted(kwargs.items())!s}"

            # Check if cached result exists and is valid
            if cache_key in cache:
                cached_result, cached_time = cache[cache_key]

                # Check TTL
                if ttl_seconds is None or (time.time() - cached_time) < ttl_seconds:
                    logger.debug(
                        f"Cache hit for {func.__name__}",
                        extra={"function": func.__name__, "cache_key": cache_key},
                    )
                    return cached_result
                # TTL expired, remove from cache
                del cache[cache_key]

            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache[cache_key] = (result, time.time())

            logger.debug(
                f"Cache miss for {func.__name__}, result cached",
                extra={"function": func.__name__, "cache_key": cache_key},
            )

            return result

        return wrapper

    return decorator
