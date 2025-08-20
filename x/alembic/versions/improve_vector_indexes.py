"""Improve vector indexes with proper operators and namespace strategy

Revision ID: improve_vector_indexes
Revises: add_pgvector_support
Create Date: 2025-01-17 11:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from app.core.config import settings

# revision identifiers, used by Alembic.
revision = 'improve_vector_indexes'
down_revision = 'add_pgvector_support'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Note: Old indexes were not created in previous migration, so we skip dropping them
    
    # Note: Vector indexes will be created later when columns are cast to vector type
    
    # Add namespace and slot strategy columns to questions table
    op.add_column('questions', sa.Column('namespace', sa.String(length=100), nullable=True, server_default='default'))
    op.add_column('questions', sa.Column('slot', sa.Integer(), nullable=True, server_default='1'))
    op.add_column('questions', sa.Column('obj_ref', sa.String(length=255), nullable=True))
    op.add_column('questions', sa.Column('deactivated_at', sa.DateTime(timezone=True), nullable=True))
    
    # Add namespace and slot strategy columns to error_patterns table
    op.add_column('error_patterns', sa.Column('namespace', sa.String(length=100), nullable=True, server_default='default'))
    op.add_column('error_patterns', sa.Column('slot', sa.Integer(), nullable=True, server_default='1'))
    op.add_column('error_patterns', sa.Column('obj_ref', sa.String(length=255), nullable=True))
    op.add_column('error_patterns', sa.Column('deactivated_at', sa.DateTime(timezone=True), nullable=True))
    
    # Create unique constraints for namespace/slot strategy
    op.create_unique_constraint(
        'uq_questions_obj_ns_slot',
        'questions',
        ['obj_ref', 'namespace', 'slot']
    )
    
    op.create_unique_constraint(
        'uq_error_patterns_obj_ns_slot',
        'error_patterns',
        ['obj_ref', 'namespace', 'slot']
    )
    
    # Note: Partial indexes will be created later when is_active column is added to error_patterns
    
    # Create indexes for namespace and slot queries
    op.create_index('ix_questions_namespace', 'questions', ['namespace'], unique=False)
    op.create_index('ix_questions_slot', 'questions', ['slot'], unique=False)
    op.create_index('ix_questions_deactivated_at', 'questions', ['deactivated_at'], unique=False)
    
    op.create_index('ix_error_patterns_namespace', 'error_patterns', ['namespace'], unique=False)
    op.create_index('ix_error_patterns_slot', 'error_patterns', ['slot'], unique=False)
    op.create_index('ix_error_patterns_deactivated_at', 'error_patterns', ['deactivated_at'], unique=False)
    
    # Add embedding dimension configuration column
    op.add_column('questions', sa.Column('embedding_dim', sa.Integer(), nullable=True, server_default=str(settings.embedding_dimension)))
    op.add_column('error_patterns', sa.Column('embedding_dim', sa.Integer(), nullable=True, server_default=str(settings.embedding_dimension)))
    
    # Create HNSW indexes for better quality (optional, for high-performance scenarios)
    # Uncomment if you want HNSW indexes instead of IVFFLAT
    # op.create_index(
    #     'ix_questions_content_embedding_hnsw',
    #     'questions',
    #     ['content_embedding'],
    #     postgresql_using='hnsw',
    #     postgresql_with={'m': 16, 'ef_construction': 200},
    #     postgresql_ops={'content_embedding': 'vector_cosine_ops'}
    # )
    # 
    # op.create_index(
    #     'ix_error_patterns_embedding_hnsw',
    #     'error_patterns',
    #     ['embedding'],
    #     postgresql_using='hnsw',
    #     postgresql_with={'m': 16, 'ef_construction': 200},
    #     postgresql_ops={'embedding': 'vector_cosine_ops'}
    # )


def downgrade() -> None:
    # Drop HNSW indexes if they exist
    # op.drop_index('ix_error_patterns_embedding_hnsw', table_name='error_patterns')
    # op.drop_index('ix_questions_content_embedding_hnsw', table_name='questions')
    
    # Drop namespace/slot indexes
    op.drop_index('ix_error_patterns_deactivated_at', table_name='error_patterns')
    op.drop_index('ix_error_patterns_slot', table_name='error_patterns')
    op.drop_index('ix_error_patterns_namespace', table_name='error_patterns')
    op.drop_index('ix_questions_deactivated_at', table_name='questions')
    op.drop_index('ix_questions_slot', table_name='questions')
    op.drop_index('ix_questions_namespace', table_name='questions')
    
    # Drop partial unique indexes
    op.drop_index('uq_error_patterns_ns_active_slot_one', table_name='error_patterns')
    op.drop_index('uq_questions_ns_active_slot_one', table_name='questions')
    
    # Drop unique constraints
    op.drop_constraint('uq_error_patterns_obj_ns_slot', table_name='error_patterns', type_='unique')
    op.drop_constraint('uq_questions_obj_ns_slot', table_name='questions', type_='unique')
    
    # Drop namespace/slot columns
    op.drop_column('error_patterns', 'embedding_dim')
    op.drop_column('error_patterns', 'deactivated_at')
    op.drop_column('error_patterns', 'obj_ref')
    op.drop_column('error_patterns', 'slot')
    op.drop_column('error_patterns', 'namespace')
    
    op.drop_column('questions', 'embedding_dim')
    op.drop_column('questions', 'deactivated_at')
    op.drop_column('questions', 'obj_ref')
    op.drop_column('questions', 'slot')
    op.drop_column('questions', 'namespace')
    
    # Drop improved indexes
    op.drop_index('ix_error_patterns_embedding_cosine', table_name='error_patterns')
    op.drop_index('ix_questions_content_embedding_cosine', table_name='questions')
    
    # Recreate original indexes without operator classes
    op.create_index('ix_questions_content_embedding', 'questions', ['content_embedding'], 
                   postgresql_using='ivfflat', postgresql_with={'lists': 100})
    op.create_index('ix_error_patterns_embedding', 'error_patterns', ['embedding'], 
                   postgresql_using='ivfflat', postgresql_with={'lists': 50})
