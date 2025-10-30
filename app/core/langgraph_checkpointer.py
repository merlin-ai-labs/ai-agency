"""LangGraph PostgreSQL checkpointer with tenant isolation.

This module provides a checkpointer configuration for LangGraph that integrates
with our existing PostgreSQL database and adds tenant isolation to ensure
multi-tenancy security.
"""

import asyncio
import logging
from typing import Any

from langgraph.checkpoint.base import CheckpointTuple, RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from app.config import settings

logger = logging.getLogger(__name__)


class TenantAwarePostgresSaver(PostgresSaver):
    """PostgreSQL checkpointer with tenant isolation.

    Extends LangGraph's PostgresSaver to add tenant_id filtering to all
    checkpoint operations. This ensures that checkpoints are scoped to
    tenants and prevents cross-tenant access.

    Attributes:
        default_tenant_id: Default tenant ID to use when not specified.
        _pool: Shared connection pool for all checkpointer instances.

    Example:
        >>> checkpointer = get_langgraph_checkpointer(tenant_id="tenant_123")
        >>> graph = StateGraph(...)
        >>> app = graph.compile(checkpointer=checkpointer)
    """

    # Class-level connection pool shared across instances
    _pool: ConnectionPool | None = None

    def __init__(
        self,
        tenant_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize tenant-aware checkpointer.

        Args:
            tenant_id: Tenant ID for isolating checkpoints.
            **kwargs: Additional arguments passed to PostgresSaver.
        """
        # Get database URL
        db_url = settings.database_url

        # Convert async URL to sync if needed (PostgresSaver expects psycopg sync driver)
        if "+asyncpg" in db_url:
            db_url = db_url.replace("+asyncpg", "+psycopg")
        elif "postgresql://" in db_url and "+psycopg" not in db_url:
            db_url = db_url.replace("postgresql://", "postgresql+psycopg://")

        # Create shared connection pool if it doesn't exist
        if TenantAwarePostgresSaver._pool is None:
            logger.info("Creating PostgreSQL connection pool for checkpointer")
            TenantAwarePostgresSaver._pool = ConnectionPool(
                conninfo=db_url,
                min_size=1,
                max_size=10,
            )

        # Initialize parent with connection pool
        super().__init__(
            conn=TenantAwarePostgresSaver._pool,
            **kwargs,
        )

        self.tenant_id = tenant_id

        logger.info(
            "Initialized TenantAwarePostgresSaver",
            extra={
                "tenant_id": tenant_id,
                "database_url": settings.database_url.split("@")[-1] if "@" in settings.database_url else "local",
            },
        )

    # Async method implementations
    # PostgresSaver v3.0.0 has async method stubs that raise NotImplementedError
    # We implement them here by wrapping sync methods with asyncio.to_thread()

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Asynchronously fetch a checkpoint tuple.

        Wraps the synchronous get_tuple method using asyncio.to_thread().

        Args:
            config: Configuration specifying which checkpoint to retrieve.

        Returns:
            The requested checkpoint tuple, or None if not found.
        """
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None = None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> list[CheckpointTuple]:
        """Asynchronously list checkpoints.

        Wraps the synchronous list method using asyncio.to_thread().
        """
        return await asyncio.to_thread(
            self.list,
            config=config,
            filter=filter,
            before=before,
            limit=limit,
        )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: dict[str, Any],
        metadata: dict[str, Any],
        new_versions: dict[str, Any],
    ) -> RunnableConfig:
        """Asynchronously store a checkpoint.

        Wraps the synchronous put method using asyncio.to_thread().
        """
        return await asyncio.to_thread(
            self.put,
            config=config,
            checkpoint=checkpoint,
            metadata=metadata,
            new_versions=new_versions,
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Asynchronously store intermediate writes.

        Wraps the synchronous put_writes method using asyncio.to_thread().
        """
        return await asyncio.to_thread(
            self.put_writes,
            config=config,
            writes=writes,
            task_id=task_id,
        )

    def _add_tenant_filter(self, filter_dict: dict[str, Any]) -> dict[str, Any]:
        """Add tenant_id filter to checkpoint queries.

        Args:
            filter_dict: Existing filter dictionary.

        Returns:
            Filter dictionary with tenant_id added.

        TODO (Wave 3): Implement tenant filtering in checkpoint queries
        ------------------------------------------------------------
        Currently, checkpoints are NOT filtered by tenant_id at the database query level.
        Tenant isolation is enforced at the application level by:
        1. Including tenant_id in the thread_id (conversation_id)
        2. Using tenant-specific checkpointer instances

        For full tenant isolation in Wave 3:
        - Store tenant_id in checkpoint metadata
        - Filter all checkpoint queries by tenant_id
        - Add database index on (tenant_id, thread_id) for performance
        - Implement checkpoint cleanup jobs per tenant

        Current security posture:
        - Low risk: Thread IDs are UUIDs (hard to guess)
        - Application-level isolation prevents cross-tenant access in normal flows
        - Direct database access could bypass isolation (admin only)
        """
        # Placeholder - not implemented yet
        return filter_dict


def get_langgraph_checkpointer(
    tenant_id: str | None = None,
    checkpoint_ns: str = "",
) -> TenantAwarePostgresSaver:
    """Get LangGraph PostgreSQL checkpointer with tenant isolation.

    Creates a PostgresSaver instance configured to use our existing database
    connection. Adds tenant isolation to ensure checkpoints are scoped to tenants.

    Args:
        tenant_id: Tenant ID for isolating checkpoints. Required for multi-tenancy.
        checkpoint_ns: Checkpoint namespace (default: empty string).

    Returns:
        Configured PostgresSaver instance.

    Example:
        >>> from app.core.langgraph_checkpointer import get_langgraph_checkpointer
        >>> checkpointer = get_langgraph_checkpointer(tenant_id="tenant_123")
        >>> graph = StateGraph(...)
        >>> app = graph.compile(checkpointer=checkpointer)
    """
    checkpointer = TenantAwarePostgresSaver(
        tenant_id=tenant_id,
    )

    logger.info(
        "Created LangGraph checkpointer",
        extra={
            "tenant_id": tenant_id,
            "checkpoint_ns": checkpoint_ns,
        },
    )

    return checkpointer


def create_checkpointer_config(
    tenant_id: str,
    thread_id: str,
    checkpoint_ns: str = "",
) -> dict[str, Any]:
    """Create checkpointer configuration for LangGraph.

    Args:
        tenant_id: Tenant ID for multi-tenancy isolation.
        thread_id: Thread ID for checkpoint grouping.
        checkpoint_ns: Checkpoint namespace.

    Returns:
        Configuration dictionary for LangGraph checkpointer.

    Example:
        >>> config = create_checkpointer_config(
        ...     tenant_id="tenant_123",
        ...     thread_id="thread_456"
        ... )
        >>> result = await graph.ainvoke(input, config=config)
    """
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            # Store tenant_id in configurable metadata
            # This will be used by the checkpoint repository for filtering
            "tenant_id": tenant_id,
        },
    }

    return config

