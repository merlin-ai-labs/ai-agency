"""add_conversation_tables

Revision ID: 437dc3bc9d26
Revises: 001
Create Date: 2025-10-28 21:53:09.990823

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON


# revision identifiers, used by Alembic.
revision = '437dc3bc9d26'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create conversations table
    op.create_table(
        'conversations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('conversation_id', sa.String(), nullable=False),
        sa.Column('tenant_id', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for conversations
    op.create_index(
        op.f('ix_conversations_conversation_id'),
        'conversations',
        ['conversation_id'],
        unique=True
    )
    op.create_index(
        op.f('ix_conversations_tenant_id'),
        'conversations',
        ['tenant_id'],
        unique=False
    )

    # Create messages table
    op.create_table(
        'messages',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('conversation_id', sa.String(), nullable=False),
        sa.Column('tenant_id', sa.String(), nullable=False),
        sa.Column('role', sa.String(), nullable=False),
        sa.Column('content', sa.String(), nullable=False),
        sa.Column('tool_calls', JSON, nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for messages
    op.create_index(
        op.f('ix_messages_conversation_id'),
        'messages',
        ['conversation_id'],
        unique=False
    )
    op.create_index(
        op.f('ix_messages_tenant_id'),
        'messages',
        ['tenant_id'],
        unique=False
    )


def downgrade() -> None:
    # Drop indexes first
    op.drop_index(op.f('ix_messages_tenant_id'), table_name='messages')
    op.drop_index(op.f('ix_messages_conversation_id'), table_name='messages')
    op.drop_index(op.f('ix_conversations_tenant_id'), table_name='conversations')
    op.drop_index(op.f('ix_conversations_conversation_id'), table_name='conversations')

    # Drop tables
    op.drop_table('messages')
    op.drop_table('conversations')
