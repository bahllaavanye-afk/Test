"""create_bots_table

Revision ID: d4e5f6a7b8c9
Revises: c1d2e3f4a5b6
Create Date: 2026-06-04

"""
from alembic import op
import sqlalchemy as sa

revision = "d4e5f6a7b8c9"
down_revision = "c1d2e3f4a5b6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "bots",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("account_id", sa.String(), nullable=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True, server_default=""),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("market_type", sa.String(length=20), nullable=False, server_default="equity"),
        sa.Column("trigger", sa.JSON(), nullable=False),
        sa.Column("conditions", sa.JSON(), nullable=True),
        sa.Column("condition_logic", sa.String(length=8), nullable=False, server_default="ALL"),
        sa.Column("action", sa.JSON(), nullable=False),
        sa.Column("exit_rules", sa.JSON(), nullable=True),
        sa.Column("is_enabled", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.Column("run_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_signal", sa.String(length=16), nullable=True),
        sa.Column("last_result", sa.JSON(), nullable=True),
        sa.Column("template_id", sa.String(length=64), nullable=True),
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
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_bots_user_id", "bots", ["user_id"])
    op.create_index("ix_bots_symbol", "bots", ["symbol"])


def downgrade() -> None:
    op.drop_index("ix_bots_symbol", table_name="bots")
    op.drop_index("ix_bots_user_id", table_name="bots")
    op.drop_table("bots")
