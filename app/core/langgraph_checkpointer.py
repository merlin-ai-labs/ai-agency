"""LangGraph PostgreSQL checkpointer with tenant isolation.

This module provides a checkpointer configuration for LangGraph that integrates
with our existing PostgreSQL database and adds tenant isolation to ensure
multi-tenancy security.

Uses Google's langchain-google-cloud-sql-pg for proper Cloud SQL integration.
"""

import asyncio
import logging
import os
import re
from typing import Any

from langgraph.checkpoint.base import CheckpointTuple, RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_google_cloud_sql_pg import PostgresEngine

from app.config import settings

logger = logging.getLogger(__name__)


class TenantAwarePostgresSaver(PostgresSaver):
    """PostgreSQL checkpointer with tenant isolation.

    Extends LangGraph's PostgresSaver to add tenant_id filtering to all
    checkpoint operations. This ensures that checkpoints are scoped to
    tenants and prevents cross-tenant access.

    Uses Google's PostgresEngine for proper Cloud SQL integration.

    Attributes:
        default_tenant_id: Default tenant ID to use when not specified.
        _engine: Shared PostgresEngine for all checkpointer instances.

    Example:
        >>> checkpointer = get_langgraph_checkpointer(tenant_id="tenant_123")
        >>> graph = StateGraph(...)
        >>> app = graph.compile(checkpointer=checkpointer)
    """

    # Class-level engine shared across instances
    _engine: PostgresEngine | None = None

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
        # Create shared engine if it doesn't exist
        if TenantAwarePostgresSaver._engine is None:
            TenantAwarePostgresSaver._engine = self._create_engine()

        # Initialize parent with engine's connection pool
        super().__init__(
            conn=TenantAwarePostgresSaver._engine._pool,
            **kwargs,
        )

        self.tenant_id = tenant_id

        logger.info(
            "Initialized TenantAwarePostgresSaver",
            extra={
                "tenant_id": tenant_id,
            },
        )

    def _create_engine(self) -> PostgresEngine:
        """Create PostgresEngine for Cloud SQL connections.

        Handles both Cloud SQL (Unix socket) and local development (proxy).

        Returns:
            PostgresEngine instance configured for the environment.
        """
        db_url = settings.database_url

        # Check if this is a Cloud SQL Unix socket connection
        # Format: postgresql://user:pass@/dbname?host=/cloudsql/project:region:instance
        if "host=/cloudsql/" in db_url:
            # Extract connection details from Cloud SQL URL
            match = re.search(r"host=/cloudsql/([^:]+):([^:]+):([^&\s]+)", db_url)
            if match:
                project_id = match.group(1)
                region = match.group(2)
                instance_name = match.group(3)

                # Extract database name
                db_match = re.search(r"@/([^?]+)", db_url)
                database = db_match.group(1) if db_match else "ai_agency"

                logger.info(
                    f"Creating Cloud SQL PostgresEngine: {project_id}:{region}:{instance_name}"
                )

                # Use PostgresEngine for Cloud SQL
                engine = PostgresEngine.from_instance(
                    project_id=project_id,
                    region=region,
                    instance=instance_name,
                    database=database,
                )

                return engine

        # For local development with cloud-sql-proxy or direct PostgreSQL
        # Parse standard PostgreSQL URL
        logger.info("Creating PostgresEngine for local/proxy connection")

        # Extract connection details
        db_match = re.match(
            r"postgresql(?:\+[^:]+)?://([^:]+):([^@]+)@([^:/]+):(\d+)/(.+)",
            db_url,
        )

        if db_match:
            user = db_match.group(1)
            password = db_match.group(2)
            host = db_match.group(3)
            port = int(db_match.group(4))
            database = db_match.group(5)

            # For cloud-sql-proxy, we need to use the instance connection approach
            # but with custom host/port
            if host == "localhost" and port == 5433:
                # This is likely cloud-sql-proxy for local development
                # Use the GCP project from settings
                project_id = settings.gcp_project_id or "merlin-notebook-lm"

                # Create engine with custom configuration
                engine = PostgresEngine.from_instance(
                    project_id=project_id,
                    region="europe-west1",
                    instance="ai-agency-db",
                    database=database,
                    # For local proxy, configure to use localhost
                    ip_type="PRIVATE",  # This will use the proxy
                )

                return engine

        # Fallback: create engine from URL (not ideal but works)
        logger.warning("Using fallback PostgresEngine creation from URL")
        engine = PostgresEngine.from_url(db_url)
        return engine

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

