"""Catch-up: implementation-shortfall fields on slippage_records.

The model gained arrival_price / is_cost_bps / vwap_shortfall_bps /
period_vwap / execution_duration_seconds without a migration (found by the
schema drift audit, 2026-07-22, when the restored Supabase Postgres came back
on the pre-drift schema). All nullable → purely additive, safe on live data.

Revision ID: k6f7a8b9c0d1
Revises: j5e6f7a8b9c0
"""
from alembic import op
import sqlalchemy as sa

revision = "k6f7a8b9c0d1"
down_revision = "j5e6f7a8b9c0"
branch_labels = None
depends_on = None

_COLS = [
    sa.Column("arrival_price", sa.Numeric(18, 8), nullable=True),
    sa.Column("is_cost_bps", sa.Numeric(8, 4), nullable=True),
    sa.Column("vwap_shortfall_bps", sa.Numeric(8, 4), nullable=True),
    sa.Column("period_vwap", sa.Numeric(18, 8), nullable=True),
    sa.Column("execution_duration_seconds", sa.Float(), nullable=True),
]


def _existing_columns() -> set[str]:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    return {c["name"] for c in insp.get_columns("slippage_records")}


def upgrade() -> None:
    # Idempotent: environments that got these columns via create_all (the
    # SQLite fallback) must not fail the boot-time upgrade.
    have = _existing_columns()
    for col in _COLS:
        if col.name not in have:
            op.add_column("slippage_records", col)


def downgrade() -> None:
    have = _existing_columns()
    for col in reversed(_COLS):
        if col.name in have:
            op.drop_column("slippage_records", col.name)
