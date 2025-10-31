"""add_langgraph_checkpoints_table

Revision ID: 003_add_langgraph_checkpoints
Revises: dc838231235f
Create Date: 2025-01-XX XX:XX:XX.XXXXXX

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision = "003_add_langgraph_checkpoints"
down_revision = "dc838231235f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create langgraph_checkpoints table following LangGraph schema
    op.create_table(
        "langgraph_checkpoints",
        sa.Column("checkpoint_id", sa.String(), nullable=False),
        sa.Column("thread_id", sa.String(), nullable=False),
        sa.Column("checkpoint_ns", sa.String(), nullable=False, server_default=""),
        sa.Column("checkpoint_data", JSONB, nullable=False),
        sa.Column("parent_checkpoint_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("NOW()")),
        # Add tenant_id for multi-tenancy isolation
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("checkpoint_id"),
    )

    # Create indexes for efficient lookups
    op.create_index(
        "idx_checkpoints_thread",
        "langgraph_checkpoints",
        ["thread_id", "checkpoint_ns"],
        unique=False,
    )

    op.create_index(
        "idx_checkpoints_tenant", "langgraph_checkpoints", ["tenant_id", "created_at"], unique=False
    )

    op.create_index(
        "idx_checkpoints_parent", "langgraph_checkpoints", ["parent_checkpoint_id"], unique=False
    )

    # Composite index for tenant-scoped thread queries
    op.create_index(
        "idx_checkpoints_tenant_thread",
        "langgraph_checkpoints",
        ["tenant_id", "thread_id", "checkpoint_ns", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    # Drop indexes first
    op.drop_index("idx_checkpoints_tenant_thread", table_name="langgraph_checkpoints")
    op.drop_index("idx_checkpoints_parent", table_name="langgraph_checkpoints")
    op.drop_index("idx_checkpoints_tenant", table_name="langgraph_checkpoints")
    op.drop_index("idx_checkpoints_thread", table_name="langgraph_checkpoints")

    # Drop table
    op.drop_table("langgraph_checkpoints")
