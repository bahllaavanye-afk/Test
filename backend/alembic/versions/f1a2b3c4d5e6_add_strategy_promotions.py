"""add_strategy_promotions

Revision ID: f1a2b3c4d5e6
Revises: d4e5f6a7b8c9
Create Date: 2026-06-12

"""
from alembic import op
import sqlalchemy as sa

revision = "f1a2b3c4d5e6"
down_revision = "d4e5f6a7b8c9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "strategy_promotions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("strategy_id", sa.String(), nullable=False),
        sa.Column("strategy_name", sa.String(length=64), nullable=False),
        sa.Column("current_stage", sa.String(length=16), nullable=False, server_default="paper"),
        # Per-stage metrics
        sa.Column("paper_metrics", sa.JSON(), nullable=True),
        sa.Column("shadow_metrics", sa.JSON(), nullable=True),
        sa.Column("staging_metrics", sa.JSON(), nullable=True),
        sa.Column("live_metrics", sa.JSON(), nullable=True),
        # Stage timestamps
        sa.Column("paper_started_at", sa.String(), nullable=True),
        sa.Column("shadow_started_at", sa.String(), nullable=True),
        sa.Column("staging_started_at", sa.String(), nullable=True),
        sa.Column("live_started_at", sa.String(), nullable=True),
        # Review state
        sa.Column("last_review_at", sa.String(), nullable=True),
        sa.Column("promotion_ready", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("promotion_ready_stage", sa.String(), nullable=True),
        sa.Column("awaiting_approval", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        # Approval / rejection
        sa.Column("approved_by", sa.String(), nullable=True),
        sa.Column("approved_at", sa.String(), nullable=True),
        sa.Column("rejection_reason", sa.Text(), nullable=True),
        # Review history
        sa.Column("review_history", sa.JSON(), nullable=True),
        # Notes
        sa.Column("notes", sa.Text(), nullable=True),
        # Timestamps
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["strategy_id"],
            ["strategies.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_strategy_promotions_strategy_id", "strategy_promotions", ["strategy_id"])


def downgrade() -> None:
    op.drop_index("ix_strategy_promotions_strategy_id", table_name="strategy_promotions")
    op.drop_table("strategy_promotions")
