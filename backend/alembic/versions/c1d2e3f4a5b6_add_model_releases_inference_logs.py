"""add_model_releases_and_inference_logs

Revision ID: c1d2e3f4a5b6
Revises: a1b2c3d4e5f6
Create Date: 2026-05-29
"""
from alembic import op
import sqlalchemy as sa

revision = "c1d2e3f4a5b6"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_releases",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("model_name", sa.String(length=64), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column("artifact_path", sa.String(length=512), nullable=False),
        sa.Column("framework", sa.String(length=32), nullable=False, server_default="pytorch"),
        sa.Column("n_features", sa.Integer(), nullable=True),
        sa.Column("seq_len", sa.Integer(), nullable=True),
        sa.Column("model_params", sa.JSON(), nullable=True),
        sa.Column("training_config", sa.JSON(), nullable=True),
        sa.Column("train_metrics", sa.JSON(), nullable=True),
        sa.Column("live_metrics", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="registered"),
        sa.Column("traffic_pct", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.String(length=128), nullable=False, server_default="system"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_releases_model_name", "model_releases", ["model_name"])
    op.create_index("ix_model_releases_status", "model_releases", ["status"])
    op.create_index("ix_mr_model_status", "model_releases", ["model_name", "status"])

    op.create_table(
        "inference_logs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("release_id", sa.String(), nullable=False),
        sa.Column("model_name", sa.String(length=64), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("prediction", sa.Numeric(10, 6), nullable=False),
        sa.Column("signal", sa.String(length=8), nullable=False),
        sa.Column("confidence", sa.Numeric(6, 4), nullable=False),
        sa.Column("latency_ms", sa.Numeric(8, 3), nullable=False),
        sa.Column("ab_group", sa.String(length=16), nullable=False),
        sa.Column("actual_return", sa.Numeric(10, 6), nullable=True),
        sa.Column("is_correct", sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(
            ["release_id"], ["model_releases.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_inference_logs_ts", "inference_logs", ["ts"])
    op.create_index("ix_inference_logs_release_id", "inference_logs", ["release_id"])
    op.create_index("ix_inf_release_ts", "inference_logs", ["release_id", "ts"])
    op.create_index("ix_inf_model_symbol", "inference_logs", ["model_name", "symbol"])


def downgrade() -> None:
    op.drop_index("ix_inf_model_symbol", table_name="inference_logs")
    op.drop_index("ix_inf_release_ts", table_name="inference_logs")
    op.drop_index("ix_inference_logs_release_id", table_name="inference_logs")
    op.drop_index("ix_inference_logs_ts", table_name="inference_logs")
    op.drop_table("inference_logs")

    op.drop_index("ix_mr_model_status", table_name="model_releases")
    op.drop_index("ix_model_releases_status", table_name="model_releases")
    op.drop_index("ix_model_releases_model_name", table_name="model_releases")
    op.drop_table("model_releases")
