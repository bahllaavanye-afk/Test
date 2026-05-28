# Broker Integration Engineer — Module Guide

## Your Role
You own the broker abstraction layer. Every order, quote, and account snapshot goes through this layer. New brokers are plugins — zero changes to any existing file are required to add one.

## Owned Files (safe to modify)
```
backend/app/brokers/
  base.py          # AbstractBroker interface (read-only reference)
  alpaca.py        # Alpaca REST + WebSocket (primary equity broker)
  alpaca_orders.py # Order types: bracket, OCO, trailing stop
  tradestation.py  # TradeStation REST + OAuth2
  binance.py       # CCXT async Binance (crypto)
  polymarket.py    # py-clob-client (prediction markets)
  encryption.py    # AES-256 Fernet encrypt/decrypt for API keys
```

## Do NOT Modify
- `base.py` — the interface is frozen; all brokers must implement it exactly
- `backend/app/risk/manager.py` — risk checks happen above this layer
- Any SQLAlchemy model file

## AbstractBroker Interface (must implement all methods)
```python
class AbstractBroker(ABC):
    async def get_account(self) -> Account: ...
    async def get_positions(self) -> list[Position]: ...
    async def get_orders(self, status: str) -> list[Order]: ...
    async def submit_order(self, req: OrderRequest) -> Order: ...
    async def cancel_order(self, order_id: str) -> bool: ...
    async def get_quote(self, symbol: str) -> Quote: ...
    async def get_bars(self, symbol: str, timeframe: str, start: date, end: date) -> pd.DataFrame: ...
    async def subscribe_trades(self, symbols: list[str], callback: Callable): ...
```

## Adding a New Broker
1. Create `backend/app/brokers/<name>.py` implementing `AbstractBroker`
2. Register in `backend/app/api/v1/accounts.py` in `BROKER_REGISTRY`
3. Add integration test in `backend/tests/integration/test_<name>_broker.py` (using VCR cassettes, not live calls)
4. Document API key format + sandbox URL in a docstring at top of the file

## Security Requirements (NON-NEGOTIABLE)
- **Never** log API keys, secrets, or account IDs
- **Always** encrypt credentials via `encryption.py` before storing in DB
- **Never** store decrypted keys in memory beyond the lifetime of a single request
- Rate limits must be respected; use an `asyncio.Semaphore` to cap concurrent calls
- Paper mode vs live mode is enforced by `Account.mode` column — check it before every order

## Alpaca-Specific Notes
```python
# Paper API base URL
ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"

# Bracket order (entry + SL + TP as linked OCO)
await alpaca.submit_order(OrderRequest(
    symbol="AAPL",
    qty=10,
    side="buy",
    order_class="bracket",
    stop_loss={"stop_price": 148.00},
    take_profit={"limit_price": 160.00},
))

# Supported timeframes for get_bars
TIMEFRAMES = ["1Min", "5Min", "15Min", "1Hour", "1Day"]
```

## Encryption Scheme (AES-256 Fernet)
```python
# encryption.py — key derived from SECRET_KEY via SHA-256
from app.brokers.encryption import encrypt_secret, decrypt_secret

# Store
encrypted = encrypt_secret(raw_api_key)        # store this in DB
# Retrieve
raw = decrypt_secret(encrypted_from_db)        # use for one request, then discard
```

## Running Tests (mock broker, no live calls)
```bash
cd backend && pytest tests/integration/test_alpaca_broker.py -v --record-mode=none
```
