"""add student_response table

Revision ID: 53ceff8c85b1
Revises: f8a10708051d
Create Date: 2025-07-25 15:17:34.343396

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '53ceff8c85b1'
down_revision = 'f8a10708051d'
branch_labels = None
depends_on = None

def upgrade() -> None:
    op.create_table(
        'student_responses',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('student_id', sa.Integer(), sa.ForeignKey('users.id')),
        sa.Column('question_id', sa.Integer(), sa.ForeignKey('questions.id')),
        sa.Column('answer', sa.String(), nullable=False),
        sa.Column('is_correct', sa.Boolean(), nullable=False),
        sa.Column('response_time', sa.Float(), nullable=True),
        sa.Column('confidence_level', sa.Integer(), nullable=True),
        sa.Column('feedback', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )
    op.create_index(op.f('ix_student_responses_id'), 'student_responses', ['id'])

def downgrade() -> None:
    op.drop_index(op.f('ix_student_responses_id'), table_name='student_responses')
    op.drop_table('student_responses') 