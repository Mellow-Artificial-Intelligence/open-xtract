"""Initial database schema

Revision ID: 001_initial
Revises:
Create Date: 2025-12-07

"""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '001_initial'
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'schemas',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('version', sa.Integer(), server_default='1'),
        sa.Column('fields', postgresql.JSONB(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_table(
        'extractions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('schema_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('status', sa.String(50), server_default='pending'),
        sa.Column('model', sa.String(255), nullable=False),
        sa.Column('source_file_path', sa.Text(), nullable=True),
        sa.Column('source_file_name', sa.String(255), nullable=True),
        sa.Column('source_file_type', sa.String(50), nullable=True),
        sa.Column('source_file_size', sa.Integer(), nullable=True),
        sa.Column('result', postgresql.JSONB(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('processing_time_ms', sa.Integer(), nullable=True),
        sa.Column('tokens_used', sa.Integer(), nullable=True),
        sa.Column('cost_usd', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['schema_id'], ['schemas.id'], ondelete='RESTRICT'),
    )
    op.create_index('ix_extractions_schema_id', 'extractions', ['schema_id'])
    op.create_index('ix_extractions_status', 'extractions', ['status'])
    op.create_index('ix_extractions_created_at', 'extractions', ['created_at'])


def downgrade() -> None:
    op.drop_table('extractions')
    op.drop_table('schemas')
