"""add bot archive columns (is_archived, archived_at)

Revision ID: i4d5e6f7a8b9
Revises: h3c4d5e6f7a8
Create Date: 2026-06-20 00:00:00.000000

Replaces the old hard-delete of bots with a soft archive. ``is_archived``
hides a bot from active lists / the desk summary / the scheduler while keeping
its row, config, and linked trades; ``archived_at`` records when it happened.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "i4d5e6f7a8b9"
# Was "h3c4d5e6f7a8" — a revision that exists in no file, which broke `alembic upgrade
# head` (phantom parent → spurious second head). The real parent is create_bots_table,
# since this migration ALTERs the bots table that revision creates.
down_revision = "d4e5f6a7b8c9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "bots",
        sa.Column(
            "is_archived",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        "bots",
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_bots_is_archived", "bots", ["is_archived"])


def downgrade() -> None:
    op.drop_index("ix_bots_is_archived", table_name="bots")
    op.drop_column("bots", "archived_at")
    op.drop_column("bots", "is_archived")
