"""Initial migration - create base tables with pgvector support

Revision ID: 001
Revises:
Create Date: 2025-10-27 22:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    # Create tenants table
    op.create_table(
        'tenants',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('tenant_id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('settings', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('tenant_id')
    )
    op.create_index(op.f('ix_tenants_tenant_id'), 'tenants', ['tenant_id'], unique=True)

    # Create runs table
    op.create_table(
        'runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('run_id', sa.String(), nullable=False),
        sa.Column('tenant_id', sa.String(), nullable=False),
        sa.Column('flow_name', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False, server_default='queued'),
        sa.Column('input_data', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('output_data', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('error_message', sa.String(), nullable=True),
        sa.Column('artifact_urls', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('run_id')
    )
    op.create_index(op.f('ix_runs_run_id'), 'runs', ['run_id'], unique=True)
    op.create_index(op.f('ix_runs_tenant_id'), 'runs', ['tenant_id'], unique=False)
    op.create_index(op.f('ix_runs_flow_name'), 'runs', ['flow_name'], unique=False)
    op.create_index(op.f('ix_runs_status'), 'runs', ['status'], unique=False)

    # Composite index for efficient polling of queued runs
    op.create_index('ix_runs_status_created_at', 'runs', ['status', 'created_at'], unique=False)

    # Composite index for tenant-specific filtering
    op.create_index('ix_runs_tenant_flow', 'runs', ['tenant_id', 'flow_name'], unique=False)

    # Create document_chunks table with pgvector
    op.create_table(
        'document_chunks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('tenant_id', sa.String(), nullable=False),
        sa.Column('document_id', sa.String(), nullable=False),
        sa.Column('content', sa.String(), nullable=False),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_document_chunks_tenant_id'), 'document_chunks', ['tenant_id'], unique=False)
    op.create_index(op.f('ix_document_chunks_document_id'), 'document_chunks', ['document_id'], unique=False)

    # Add vector column for embeddings (1536 dimensions for OpenAI)
    # Using raw SQL because SQLAlchemy doesn't have native pgvector type support
    op.execute('ALTER TABLE document_chunks ADD COLUMN embedding vector(1536)')

    # Create vector similarity index using ivfflat (inner product for cosine similarity)
    # Note: This requires some data in the table first. For production, run after data load:
    # op.execute('CREATE INDEX ix_document_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)')

    # For now, create a basic index that works with empty table
    op.execute('CREATE INDEX ix_document_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops)')


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_index(op.f('ix_document_chunks_embedding'), table_name='document_chunks')
    op.drop_index(op.f('ix_document_chunks_document_id'), table_name='document_chunks')
    op.drop_index(op.f('ix_document_chunks_tenant_id'), table_name='document_chunks')
    op.drop_table('document_chunks')

    op.drop_index('ix_runs_tenant_flow', table_name='runs')
    op.drop_index('ix_runs_status_created_at', table_name='runs')
    op.drop_index(op.f('ix_runs_status'), table_name='runs')
    op.drop_index(op.f('ix_runs_flow_name'), table_name='runs')
    op.drop_index(op.f('ix_runs_tenant_id'), table_name='runs')
    op.drop_index(op.f('ix_runs_run_id'), table_name='runs')
    op.drop_table('runs')

    op.drop_index(op.f('ix_tenants_tenant_id'), table_name='tenants')
    op.drop_table('tenants')

    # Note: Not dropping pgvector extension to avoid breaking other databases that might share it
    # op.execute('DROP EXTENSION IF EXISTS vector')
