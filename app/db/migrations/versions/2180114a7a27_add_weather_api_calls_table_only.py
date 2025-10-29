"""add weather_api_calls table only

Revision ID: 2180114a7a27
Revises: 437dc3bc9d26
Create Date: 2025-10-29 09:35:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '2180114a7a27'
down_revision = '437dc3bc9d26'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create weather_api_calls table
    op.create_table(
        'weather_api_calls',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('tenant_id', sa.String(), nullable=False),
        sa.Column('location', sa.String(), nullable=False),
        sa.Column('units', sa.String(), nullable=False),
        sa.Column('temperature', sa.Float(), nullable=True),
        sa.Column('feels_like', sa.Float(), nullable=True),
        sa.Column('weather_condition', sa.String(), nullable=True),
        sa.Column('weather_description', sa.String(), nullable=True),
        sa.Column('humidity', sa.Integer(), nullable=True),
        sa.Column('wind_speed', sa.Float(), nullable=True),
        sa.Column('response_data', sa.JSON(), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=False),
        sa.Column('error_message', sa.String(), nullable=True),
        sa.Column('api_call_ms', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes
    op.create_index('ix_weather_api_calls_tenant_id', 'weather_api_calls', ['tenant_id'])
    op.create_index('ix_weather_api_calls_location', 'weather_api_calls', ['location'])
    op.create_index('ix_weather_api_calls_success', 'weather_api_calls', ['success'])
    op.create_index('ix_weather_api_calls_created_at', 'weather_api_calls', ['created_at'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_weather_api_calls_created_at', table_name='weather_api_calls')
    op.drop_index('ix_weather_api_calls_success', table_name='weather_api_calls')
    op.drop_index('ix_weather_api_calls_location', table_name='weather_api_calls')
    op.drop_index('ix_weather_api_calls_tenant_id', table_name='weather_api_calls')

    # Drop table
    op.drop_table('weather_api_calls')
