"""add_flow_type_to_conversations

Revision ID: dc838231235f
Revises: 2180114a7a27
Create Date: 2025-10-29 10:54:58.478636

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON


# revision identifiers, used by Alembic.
revision = "dc838231235f"
down_revision = "2180114a7a27"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add flow_type column to conversations
    op.add_column("conversations", sa.Column("flow_type", sa.String(), nullable=True))

    # Add flow_metadata column to conversations
    op.add_column(
        "conversations", sa.Column("flow_metadata", JSON, nullable=True, server_default="{}")
    )

    # Backfill existing conversations (assume weather)
    op.execute("""
        UPDATE conversations
        SET flow_type = 'weather', flow_metadata = '{}'
        WHERE flow_type IS NULL
    """)

    # Make flow_type NOT NULL
    op.alter_column("conversations", "flow_type", nullable=False)

    # Add flow_type to messages
    op.add_column("messages", sa.Column("flow_type", sa.String(), nullable=True))

    # Add message_metadata to messages
    op.add_column(
        "messages", sa.Column("message_metadata", JSON, nullable=True, server_default="{}")
    )

    # Backfill messages from conversations
    op.execute("""
        UPDATE messages m
        SET flow_type = c.flow_type, message_metadata = '{}'
        FROM conversations c
        WHERE m.conversation_id = c.conversation_id
    """)

    # Make flow_type NOT NULL
    op.alter_column("messages", "flow_type", nullable=False)

    # Create composite indexes
    op.create_index(
        "ix_conversations_tenant_flow", "conversations", ["tenant_id", "flow_type", "updated_at"]
    )

    op.create_index("ix_messages_tenant_flow", "messages", ["tenant_id", "flow_type", "created_at"])

    op.create_index(
        "ix_messages_conversation_flow", "messages", ["conversation_id", "flow_type", "created_at"]
    )


def downgrade() -> None:
    # Drop indexes
    op.drop_index("ix_messages_conversation_flow", table_name="messages")
    op.drop_index("ix_messages_tenant_flow", table_name="messages")
    op.drop_index("ix_conversations_tenant_flow", table_name="conversations")

    # Drop columns from messages
    op.drop_column("messages", "message_metadata")
    op.drop_column("messages", "flow_type")

    # Drop columns from conversations
    op.drop_column("conversations", "flow_metadata")
    op.drop_column("conversations", "flow_type")
