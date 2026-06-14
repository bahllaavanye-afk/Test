"""add_agent_activity_logs

Revision ID: g2b3c4d5e6f7
Revises: f1a2b3c4d5e6
Create Date: 2026-06-14 00:00:00.000000

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "g2b3c4d5e6f7"
down_revision = "f1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_activity_logs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("employee_id", sa.String(64), nullable=True),
        sa.Column("agent_type", sa.String(64), nullable=True),
        sa.Column("action", sa.String(256), nullable=False),
        sa.Column("tool_used", sa.String(128), nullable=True),
        sa.Column("input_summary", sa.Text(), nullable=True),
        sa.Column("output_summary", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="ok"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("anomaly_score", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("is_anomaly", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("reviewed_by", sa.String(64), nullable=True),
        sa.Column("reviewed_at", sa.String(64), nullable=True),
        sa.Column("review_note", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("strategy_name", sa.String(128), nullable=True),
        sa.Column("symbol", sa.String(32), nullable=True),
        sa.Column("account_id", sa.String(64), nullable=True),
        sa.Column("extra", sa.JSON(), nullable=True),
    )
    op.create_index(
        "ix_agent_activity_logs_created_at",
        "agent_activity_logs",
        ["created_at"],
    )
    op.create_index(
        "ix_agent_activity_logs_employee_id",
        "agent_activity_logs",
        ["employee_id"],
    )
    op.create_index(
        "ix_agent_activity_logs_is_anomaly",
        "agent_activity_logs",
        ["is_anomaly"],
    )


def downgrade() -> None:
    op.drop_index("ix_agent_activity_logs_is_anomaly", table_name="agent_activity_logs")
    op.drop_index("ix_agent_activity_logs_employee_id", table_name="agent_activity_logs")
    op.drop_index("ix_agent_activity_logs_created_at", table_name="agent_activity_logs")
    op.drop_table("agent_activity_logs")
