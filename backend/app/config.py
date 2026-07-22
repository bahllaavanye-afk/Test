# QuantEdge Platform — Institutional-Grade Quantitative Trading
# Version: 1.0.0  |  Mode: paper (live trading permanently disabled)
# Config loaded from environment variables via Pydantic BaseSettings
# TRADING_MODE defaults to "paper" — live mode is permanently disabled

"""Configuration module for the QuantEdge backend.

This module defines the :class:`Settings` class, a Pydantic ``BaseSettings`` subclass
that loads configuration values from environment variables (or a ``.env`` file) and
provides convenient helpers for runtime checks such as secret‑key validation and
CORS origin parsing.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator
import os as _os
from typing import Dict, List


# Resolve .env to the backend/ dir regardless of where uvicorn is launched from.
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_BACKEND_ENV = _os.path.join(_HERE, "..", ".env")


class Settings(BaseSettings):
    """Pydantic settings container for application configuration.

    The class reads values from environment variables and a ``.env`` file located
    relative to this module. It provides defaults for development and validates
    critical values such as the secret key.
    """

    model_config = SettingsConfigDict(env_file=_BACKEND_ENV, extra="ignore")

    # App
    app_name: str = "QuantEdge"
    environment: str = "development"
    debug: bool = False
    trading_mode: str = "paper"  # 'paper' | 'live'
    # When true, /auth/demo issues a token for a shared demo user so the login-free public
    # app is functional (data loads, buttons work). Set false for real multi-user.
    demo_mode: bool = True
    allowed_origins: str = "http://localhost:5173"

    # Security
    secret_key: str = Field(default="change-me-in-production-32-byte-hex")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7

    # Database — accepts sqlite+aiosqlite:// for local dev, or postgres:// for production.
    # Defaults to SQLite so the app starts without any credentials.
    database_url: str = "sqlite+aiosqlite:///./dev.db"
    # When the primary DATABASE_URL is unreachable at boot (Supabase free tier auto-pauses
    # after 7 idle days and every endpoint 500s), fall back to a local SQLite file so the
    # platform stays functional: bots reseed from templates, desk trades resync from Alpaca.
    # Loudly surfaced in /health/detailed as a failing database_primary check.
    db_fallback_to_sqlite: bool = True
    alembic_database_url: str = "sqlite:///./dev.db"

    @model_validator(mode="before")
    @classmethod
    def _normalise_database_url(cls, values: Dict[str, str]) -> Dict[str, str]:
        """Normalize ``database_url`` to the asyncpg dialect expected by SQLAlchemy.

        Supabase provides ``postgres://`` URLs, but the async driver requires the
        ``postgresql+asyncpg://`` scheme. This validator rewrites the URL accordingly
        before the model is instantiated.
        """
        url = values.get("database_url", "")
        if isinstance(url, str):
            if url.startswith("postgres://"):
                url = "postgresql+asyncpg://" + url[len("postgres://") :]
            elif url.startswith("postgresql://"):
                url = "postgresql+asyncpg://" + url[len("postgresql://") :]
            values["database_url"] = url
        return values

    # Redis (Upstash). Default is UNSET so the app cleanly no-ops the cache when no
    # Redis is provisioned (local dev / Render free tier) instead of hammering a
    # non-existent localhost:6379 and spamming connection-refused. In prod, REDIS_URL
    # is injected via the environment (Upstash rediss:// URL).
    redis_url: str = ""

    # Brokers
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"  # paper by default

    tradestation_client_id: str = ""
    tradestation_secret: str = ""
    tradestation_paper: bool = True

    polymarket_private_key: str = ""
    polymarket_chain_id: int = 137  # Polygon mainnet

    # ML
    models_dir: str = "models_artifacts"

    # Risk defaults
    max_position_pct: float = 0.05       # max 5% of portfolio per trade
    max_drawdown_pct: float = 0.10       # halt all at -10% drawdown
    arb_bucket_pct: float = 0.70         # 70% capital to arbitrage bucket
    ml_bucket_pct: float = 0.30          # 30% capital to ML bucket

    # Anthropic — for CTO agent Slack review and alpha mining
    anthropic_api_key: str = ""        # sk-ant-... from console.anthropic.com

    # Slack — bot token (preferred) or webhooks per channel
    slack_bot_token: str = ""          # xoxb-... (chat:write + chat:write.public scopes)
    slack_signing_secret: str = ""     # Slack App → Basic Information → Signing Secret (verifies /slack/events)
    slack_webhook_default: str = ""
    slack_webhook_orders: str = ""
    slack_webhook_signals: str = ""
    slack_webhook_alerts: str = ""
    slack_webhook_experiments: str = ""
    slack_webhook_system: str = ""

    # Google OAuth (optional — set to enable Google login)
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8000/api/v1/auth/google/callback"

    @model_validator(mode="after")
    def _validate_secret_key(self) -> "Settings":
        """Validate that ``secret_key`` is a secure 32‑byte hex string.

        In non‑development environments the placeholder values are prohibited.
        A short key also raises an error. This guard prevents accidental deployment
        with an insecure secret.
        """
        placeholder = "change-me-in-production-32-byte-hex"
        test_placeholder = "test-secret-key-32-bytes-hex-xxxxx"
        if self.secret_key in (placeholder, test_placeholder):
            if self.trading_mode not in ("development", "dev", "test"):
                raise ValueError(
                    "SECRET_KEY must be set to a secure random 32-byte hex value. "
                    "Run: python -c \"import secrets; print(secrets.token_hex(32))\""
                )
        elif len(self.secret_key) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long.")
        return self

    @property
    def is_paper(self) -> bool:
        """Return ``True`` if the application is running in paper‑trading mode."""
        return self.trading_mode == "paper"

    @property
    def cors_origins(self) -> List[str]:
        """Parse ``allowed_origins`` into a list of origin strings for CORS configuration."""
        return [o.strip() for o in self.allowed_origins.split(",")]


settings = Settings()