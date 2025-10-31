"""Simple in-memory token bucket rate limiter for LLM API calls.

This is an R&D implementation using in-memory storage. For production with
multiple workers, consider using Redis or similar distributed cache.

The token bucket algorithm allows for:
- Smooth rate limiting over time windows
- Handling burst traffic (up to burst_multiplier)
- Per-tenant/user rate limiting
- Multiple time windows (minute and hour)
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Tuple

from app.core.exceptions import RateLimitError

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """Token bucket rate limiter for controlling LLM API request rates.

    Uses token bucket algorithm to enforce rate limits per tenant with
    support for both per-minute and per-hour limits.

    Attributes:
        tokens_per_minute: Maximum tokens allowed per minute
        tokens_per_hour: Maximum tokens allowed per hour
        burst_size: Allow short bursts (burst_multiplier × tokens_per_minute)

    Example:
        >>> limiter = TokenBucketRateLimiter(
        ...     tokens_per_minute=90000,  # OpenAI GPT-4.1 limit
        ...     tokens_per_hour=5000000
        ... )
        >>>
        >>> # Check if request is allowed
        >>> allowed, wait_time = await limiter.acquire("tenant_123", 1000)
        >>> if not allowed:
        ...     await asyncio.sleep(wait_time)
        >>>
        >>> # Or use wait_if_needed helper
        >>> await limiter.wait_if_needed("tenant_123", 1000)
    """

    def __init__(
        self,
        tokens_per_minute: int = 90000,
        tokens_per_hour: int = 5000000,
        burst_multiplier: float = 1.5,
    ) -> None:
        """Initialize rate limiter.

        Args:
            tokens_per_minute: Maximum tokens per minute
            tokens_per_hour: Maximum tokens per hour
            burst_multiplier: Allow bursts up to this multiple of per-minute limit
        """
        self.tokens_per_minute = tokens_per_minute
        self.tokens_per_hour = tokens_per_hour
        self.burst_size = int(tokens_per_minute * burst_multiplier)

        # In-memory buckets: {tenant_id: {minute_tokens, minute_reset, ...}}
        self._buckets: Dict[str, Dict] = defaultdict(
            lambda: {
                "minute_tokens": self.burst_size,
                "minute_reset": datetime.now() + timedelta(minutes=1),
                "hour_tokens": tokens_per_hour,
                "hour_reset": datetime.now() + timedelta(hours=1),
            }
        )

        logger.debug(
            "Rate limiter initialized",
            extra={
                "tokens_per_minute": tokens_per_minute,
                "tokens_per_hour": tokens_per_hour,
                "burst_size": self.burst_size,
            },
        )

    async def acquire(
        self,
        tenant_id: str,
        estimated_tokens: int,
    ) -> Tuple[bool, int]:
        """Try to acquire tokens for a request.

        Args:
            tenant_id: Unique identifier for the tenant/user
            estimated_tokens: Number of tokens needed for this request

        Returns:
            Tuple of (allowed, wait_seconds):
                - allowed: True if request can proceed
                - wait_seconds: If not allowed, seconds to wait before retry

        Example:
            >>> allowed, wait_time = await limiter.acquire("tenant_1", 500)
            >>> if not allowed:
            ...     print(f"Rate limited. Wait {wait_time}s")
        """
        bucket = self._buckets[tenant_id]
        now = datetime.now()

        # Refill minute bucket if window expired
        if now >= bucket["minute_reset"]:
            bucket["minute_tokens"] = self.burst_size
            bucket["minute_reset"] = now + timedelta(minutes=1)
            logger.debug(
                f"Refilled minute bucket for {tenant_id}",
                extra={"tenant_id": tenant_id, "tokens": self.burst_size},
            )

        # Refill hour bucket if window expired
        if now >= bucket["hour_reset"]:
            bucket["hour_tokens"] = self.tokens_per_hour
            bucket["hour_reset"] = now + timedelta(hours=1)
            logger.debug(
                f"Refilled hour bucket for {tenant_id}",
                extra={"tenant_id": tenant_id, "tokens": self.tokens_per_hour},
            )

        # Check if we have enough tokens in both buckets
        if (
            bucket["minute_tokens"] >= estimated_tokens
            and bucket["hour_tokens"] >= estimated_tokens
        ):
            # Consume tokens from both buckets
            bucket["minute_tokens"] -= estimated_tokens
            bucket["hour_tokens"] -= estimated_tokens

            logger.debug(
                f"Tokens acquired for {tenant_id}",
                extra={
                    "tenant_id": tenant_id,
                    "tokens": estimated_tokens,
                    "minute_remaining": bucket["minute_tokens"],
                    "hour_remaining": bucket["hour_tokens"],
                },
            )
            return True, 0

        # Rate limited - calculate wait time
        wait_seconds = max(
            (bucket["minute_reset"] - now).total_seconds(),
            (bucket["hour_reset"] - now).total_seconds()
            if bucket["hour_tokens"] < estimated_tokens
            else 0,
        )

        logger.warning(
            f"Rate limit reached for {tenant_id}",
            extra={
                "tenant_id": tenant_id,
                "requested_tokens": estimated_tokens,
                "minute_available": bucket["minute_tokens"],
                "hour_available": bucket["hour_tokens"],
                "wait_seconds": wait_seconds,
            },
        )

        return False, max(1, int(wait_seconds))

    async def wait_if_needed(
        self,
        tenant_id: str,
        estimated_tokens: int,
        max_wait: int = 60,
    ) -> None:
        """Wait for rate limit if needed, then proceed.

        This is a convenience method that will automatically wait if
        rate limited, or raise an error if wait time is too long.

        Args:
            tenant_id: Unique identifier for the tenant/user
            estimated_tokens: Number of tokens needed
            max_wait: Maximum seconds willing to wait (raises error if exceeded)

        Raises:
            RateLimitError: If wait time exceeds max_wait or still limited after waiting

        Example:
            >>> try:
            ...     await limiter.wait_if_needed("tenant_1", 500, max_wait=30)
            ...     # Proceed with API call
            ... except RateLimitError:
            ...     # Handle rate limit error
        """
        allowed, wait_time = await self.acquire(tenant_id, estimated_tokens)

        if not allowed:
            if wait_time > max_wait:
                raise RateLimitError(
                    f"Rate limit exceeded. Retry after {wait_time}s (max wait: {max_wait}s)",
                    details={
                        "tenant_id": tenant_id,
                        "wait_seconds": wait_time,
                        "max_wait": max_wait,
                    },
                )

            logger.info(
                f"Waiting {wait_time}s for rate limit",
                extra={"tenant_id": tenant_id, "wait_seconds": wait_time},
            )
            await asyncio.sleep(wait_time)

            # Retry after wait
            allowed, remaining_wait = await self.acquire(tenant_id, estimated_tokens)
            if not allowed:
                raise RateLimitError(
                    f"Rate limit still exceeded after waiting {wait_time}s",
                    details={
                        "tenant_id": tenant_id,
                        "wait_seconds": wait_time,
                        "remaining_wait": remaining_wait,
                    },
                )

    def get_status(self, tenant_id: str) -> Dict:
        """Get current rate limit status for a tenant.

        Args:
            tenant_id: Unique identifier for the tenant/user

        Returns:
            Dict with current token availability and reset times

        Example:
            >>> status = limiter.get_status("tenant_1")
            >>> print(f"Available: {status['minute_tokens']} tokens")
        """
        if tenant_id not in self._buckets:
            return {
                "minute_tokens": self.burst_size,
                "hour_tokens": self.tokens_per_hour,
                "minute_reset": datetime.now() + timedelta(minutes=1),
                "hour_reset": datetime.now() + timedelta(hours=1),
            }

        bucket = self._buckets[tenant_id]
        now = datetime.now()

        return {
            "minute_tokens": bucket["minute_tokens"],
            "hour_tokens": bucket["hour_tokens"],
            "minute_reset": bucket["minute_reset"],
            "hour_reset": bucket["hour_reset"],
            "minute_seconds_until_reset": max(0, (bucket["minute_reset"] - now).total_seconds()),
            "hour_seconds_until_reset": max(0, (bucket["hour_reset"] - now).total_seconds()),
        }

    def reset_tenant(self, tenant_id: str) -> None:
        """Reset rate limits for a specific tenant.

        Useful for testing or administrative actions.

        Args:
            tenant_id: Unique identifier for the tenant/user
        """
        if tenant_id in self._buckets:
            del self._buckets[tenant_id]
            logger.info(f"Reset rate limits for {tenant_id}")
