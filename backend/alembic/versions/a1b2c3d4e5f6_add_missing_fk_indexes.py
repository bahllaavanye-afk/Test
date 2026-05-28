"""add_missing_fk_indexes

Revision ID: a1b2c3d4e5f6
Revises: ee1a38997dca
Create Date: 2026-05-28
"""
from alembic import op

revision = "a1b2c3d4e5f6"
down_revision = "ee1a38997dca"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_strategies_account_id", "strategies", ["account_id"])
    op.create_index("ix_risk_rules_account_id", "risk_rules", ["account_id"])
    op.create_index("ix_risk_events_account_id", "risk_events", ["account_id"])
    op.create_index("ix_account_snapshots_account_id", "account_snapshots", ["account_id"])


def downgrade() -> None:
    op.drop_index("ix_account_snapshots_account_id", table_name="account_snapshots")
    op.drop_index("ix_risk_events_account_id", table_name="risk_events")
    op.drop_index("ix_risk_rules_account_id", table_name="risk_rules")
    op.drop_index("ix_strategies_account_id", table_name="strategies")
