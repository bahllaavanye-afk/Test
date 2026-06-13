# QuantEdge Platform — Institutional-Grade Quantitative Trading
# Version: 1.0.0  |  Mode: paper (live trading permanently disabled)
# Config loaded from environment variables via Pydantic BaseSettings
# TRADING_MODE defaults to "paper" — live mode is permanently disabled
import os as _os

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env to the backend/ dir regardless of where uvicorn is launched from.
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_BACKEND_ENV = _os.path.join(_HERE, "..", ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_BACKEND_ENV, extra="ignore")

    # App
    app_name: str = "QuantEdge"
    environment: str = "development"
    debug: bool = False
    trading_mode: str = "paper"  # 'paper' | 'live'
    allowed_origins: str = "http://localhost:5173"

    # Security
    secret_key: str = Field(default="change-me-in-production-32-byte-hex")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7

    # Database — accepts sqlite+aiosqlite:// for local dev, or postgres:// for production.
    # Defaults to SQLite so the app starts without any credentials.
    database_url: str = "sqlite+aiosqlite:///./dev.db"
    alembic_database_url: str = "sqlite:///./dev.db"

    @model_validator(mode="before")
    @classmethod
    def _normalise_database_url(cls, values: dict) -> dict:
        """Render and Supabase provide postgres:// — SQLAlchemy async needs postgresql+asyncpg://."""
        url = values.get("database_url", "")
        if isinstance(url, str):
            if url.startswith("postgres://"):
                url = "postgresql+asyncpg://" + url[len("postgres://"):]
            elif url.startswith("postgresql://"):
                url = "postgresql+asyncpg://" + url[len("postgresql://"):]
            values["database_url"] = url
        return values

    # Redis (Upstash)
    redis_url: str = "redis://localhost:6379"

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

    # Model artifact store — Supabase Storage REST
    supabase_url: str = ""               # https://xxxx.supabase.co
    supabase_service_key: str = ""       # service_role key (never public)
    model_bucket: str = "model-artifacts"
    model_store_enabled: bool = True     # False → local-only mode

    @model_validator(mode="after")
    def _validate_secret_key(self) -> "Settings":
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
        return self.trading_mode == "paper"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]


settings = Settings()
