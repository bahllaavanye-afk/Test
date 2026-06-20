"""cross-desk instrument fields on positions and orders

Revision ID: j5e6f7a8b9c0
Revises: i4d5e6f7a8b9
Create Date: 2026-06-20 00:00:00.000000

Desk consolidation stage 3: give positions and orders one shape across every desk.
``asset_class`` unifies tracking (equity/crypto/option/future/prediction/fx/rate) and the
options instrument fields (underlying/expiry/strike/right/multiplier) let the options desk
be tracked alongside the others. Existing rows default to asset_class='equity', multiplier=1.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "j5e6f7a8b9c0"
down_revision = "i4d5e6f7a8b9"
branch_labels = None
depends_on = None

_TABLES = ("positions", "orders")


def upgrade() -> None:
    for table in _TABLES:
        op.add_column(table, sa.Column("asset_class", sa.String(16), nullable=False,
                                       server_default="equity"))
        op.add_column(table, sa.Column("underlying_symbol", sa.String(32), nullable=True))
        op.add_column(table, sa.Column("expiry", sa.Date(), nullable=True))
        op.add_column(table, sa.Column("strike", sa.Numeric(18, 8), nullable=True))
        op.add_column(table, sa.Column("option_right", sa.String(4), nullable=True))
        op.add_column(table, sa.Column("contract_multiplier", sa.Integer(), nullable=False,
                                       server_default="1"))


def downgrade() -> None:
    for table in _TABLES:
        for col in ("contract_multiplier", "option_right", "strike", "expiry",
                    "underlying_symbol", "asset_class"):
            op.drop_column(table, col)
