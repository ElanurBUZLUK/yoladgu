"""Add pgvector support

Revision ID: add_pgvector_support
Revises: 485280bb3355
Create Date: 2025-01-17 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from app.core.config import settings

# revision identifiers, used by Alembic.
revision = 'add_pgvector_support'
down_revision = '485280bb3355'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Add embedding columns to questions table
    op.add_column('questions', sa.Column('content_embedding', sa.dialects.postgresql.VECTOR(settings.embedding_dimension), nullable=True))
    op.add_column('questions', sa.Column('estimated_difficulty', sa.Float(), nullable=True))
    op.add_column('questions', sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'))
    
    # Add embedding columns to error_patterns table
    op.add_column('error_patterns', sa.Column('embedding', sa.dialects.postgresql.VECTOR(settings.embedding_dimension), nullable=True))
    op.add_column('error_patterns', sa.Column('pattern_details', sa.Text(), nullable=True))
    
    # Add math profile columns to users table
    op.add_column('users', sa.Column('math_profile_id', sa.UUID(), nullable=True))
    op.add_column('users', sa.Column('global_skill', sa.Float(), nullable=True, server_default='0.5'))
    op.add_column('users', sa.Column('difficulty_factor', sa.Float(), nullable=True, server_default='1.0'))
    op.add_column('users', sa.Column('last_k_outcomes', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('users', sa.Column('thompson_alpha', sa.Float(), nullable=True, server_default='1.0'))
    op.add_column('users', sa.Column('thompson_beta', sa.Float(), nullable=True, server_default='1.0'))
    op.add_column('users', sa.Column('epsilon', sa.Float(), nullable=True, server_default='0.1'))
    op.add_column('users', sa.Column('last_selection_mode', sa.String(length=50), nullable=True))
    op.add_column('users', sa.Column('recovery_mode_active', sa.Boolean(), nullable=True, server_default='false'))
    op.add_column('users', sa.Column('srs_mode_active', sa.Boolean(), nullable=True, server_default='false'))
    
    # Create indexes for vector similarity search
    op.create_index('ix_questions_content_embedding', 'questions', ['content_embedding'], 
                   postgresql_using='ivfflat', postgresql_with={'lists': 100})
    op.create_index('ix_error_patterns_embedding', 'error_patterns', ['embedding'], 
                   postgresql_using='ivfflat', postgresql_with={'lists': 50})
    
    # Create indexes for performance
    op.create_index('ix_questions_is_active', 'questions', ['is_active'], unique=False)
    op.create_index('ix_questions_estimated_difficulty', 'questions', ['estimated_difficulty'], unique=False)
    op.create_index('ix_users_math_profile_id', 'users', ['math_profile_id'], unique=False)


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_users_math_profile_id', table_name='users')
    op.drop_index('ix_questions_estimated_difficulty', table_name='questions')
    op.drop_index('ix_questions_is_active', table_name='questions')
    op.drop_index('ix_error_patterns_embedding', table_name='error_patterns')
    op.drop_index('ix_questions_content_embedding', table_name='questions')
    
    # Drop columns from users table
    op.drop_column('users', 'srs_mode_active')
    op.drop_column('users', 'recovery_mode_active')
    op.drop_column('users', 'last_selection_mode')
    op.drop_column('users', 'epsilon')
    op.drop_column('users', 'thompson_beta')
    op.drop_column('users', 'thompson_alpha')
    op.drop_column('users', 'last_k_outcomes')
    op.drop_column('users', 'difficulty_factor')
    op.drop_column('users', 'global_skill')
    op.drop_column('users', 'math_profile_id')
    
    # Drop columns from error_patterns table
    op.drop_column('error_patterns', 'pattern_details')
    op.drop_column('error_patterns', 'embedding')
    
    # Drop columns from questions table
    op.drop_column('questions', 'is_active')
    op.drop_column('questions', 'estimated_difficulty')
    op.drop_column('questions', 'content_embedding')
    
    # Note: We don't drop the vector extension as it might be used by other parts of the system
