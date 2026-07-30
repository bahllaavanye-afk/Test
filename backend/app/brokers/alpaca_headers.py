from app.config import settings


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests."""
    if not isinstance(settings.alpaca_api_key, str) or not settings.alpaca_api_key.strip():
        raise ValueError("Alpaca API key must be a non-empty string")
    if not isinstance(settings.alpaca_secret_key, str) or not settings.alpaca_secret_key.strip():
        raise ValueError("Alpaca secret key must be a non-empty string")
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }