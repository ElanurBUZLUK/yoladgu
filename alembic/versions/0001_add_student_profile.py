"""add student profile table

Revision ID: 0001
Revises: 
Create Date: 2024-01-01 00:00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        'student_profiles',
        sa.Column('student_id', sa.Integer(), sa.ForeignKey('users.id'), primary_key=True),
        sa.Column('level', sa.Float(), nullable=True),
        sa.Column('min_level', sa.Float(), nullable=True),
        sa.Column('max_level', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    )
    op.create_index(op.f('ix_student_profiles_student_id'), 'student_profiles', ['student_id'], unique=False)

def downgrade() -> None:
    op.drop_index(op.f('ix_student_profiles_student_id'), table_name='student_profiles')
    op.drop_table('student_profiles') 