from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

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

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/quantedge"
    alembic_database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/quantedge"

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

    # Slack webhooks (optional, each channel separately)
    slack_webhook_default: str = ""
    slack_webhook_orders: str = ""
    slack_webhook_signals: str = ""
    slack_webhook_alerts: str = ""
    slack_webhook_experiments: str = ""
    slack_webhook_system: str = ""

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
