"""Create embeddings table for vector storage

Revision ID: create_embeddings_table
Revises: improve_vector_indexes
Create Date: 2025-01-17 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from app.core.config import settings

# revision identifiers, used by Alembic.
revision = 'create_embeddings_table'
down_revision = 'improve_vector_indexes'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create embeddings table for general vector storage
    op.create_table('embeddings',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('obj_ref', sa.String(length=255), nullable=False),
        sa.Column('namespace', sa.String(length=100), nullable=False, server_default='default'),
        sa.Column('slot', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('embedding', sa.Text(), nullable=False),  # Will be cast to vector later
        sa.Column('meta', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('embedding_dim', sa.Integer(), nullable=False, server_default=str(settings.embedding_dimension)),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('deactivated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for embeddings table
    op.create_index('ix_embeddings_obj_ref', 'embeddings', ['obj_ref'], unique=False)
    op.create_index('ix_embeddings_namespace', 'embeddings', ['namespace'], unique=False)
    op.create_index('ix_embeddings_slot', 'embeddings', ['slot'], unique=False)
    op.create_index('ix_embeddings_is_active', 'embeddings', ['is_active'], unique=False)
    op.create_index('ix_embeddings_created_at', 'embeddings', ['created_at'], unique=False)
    op.create_index('ix_embeddings_deactivated_at', 'embeddings', ['deactivated_at'], unique=False)
    
    # Note: Vector index will be created later when embedding column is cast to vector type
    
    # Create unique constraint for obj_ref + namespace + slot
    op.create_unique_constraint(
        'uq_embeddings_obj_ns_slot',
        'embeddings',
        ['obj_ref', 'namespace', 'slot']
    )
    
    # Create partial unique index for active slots (one active slot per namespace)
    op.create_index(
        'uq_embeddings_ns_active_slot_one',
        'embeddings',
        ['namespace', 'slot'],
        unique=True,
        postgresql_where=sa.text("is_active = true")
    )


def downgrade() -> None:
    # Drop indexes
    op.drop_index('uq_embeddings_ns_active_slot_one', table_name='embeddings')
    op.drop_constraint('uq_embeddings_obj_ns_slot', table_name='embeddings', type_='unique')
    op.drop_index('ix_embeddings_embedding_cosine', table_name='embeddings')
    op.drop_index('ix_embeddings_deactivated_at', table_name='embeddings')
    op.drop_index('ix_embeddings_created_at', table_name='embeddings')
    op.drop_index('ix_embeddings_is_active', table_name='embeddings')
    op.drop_index('ix_embeddings_slot', table_name='embeddings')
    op.drop_index('ix_embeddings_namespace', table_name='embeddings')
    op.drop_index('ix_embeddings_obj_ref', table_name='embeddings')
    
    # Drop table
    op.drop_table('embeddings')
