"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
from app.config import settings


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests.

    Raises:
        ValueError: If either the API key or secret key is missing or empty.
    """
    api_key = getattr(settings, "alpaca_api_key", None)
    secret_key = getattr(settings, "alpaca_secret_key", None)

    if not isinstance(api_key, str) or not api_key:
        raise ValueError("Alpaca API key must be a non‑empty string")
    if not isinstance(secret_key, str) or not secret_key:
        raise ValueError("Alpaca secret key must be a non‑empty string")

    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }


# Unit tests for edge cases
import unittest
from unittest.mock import patch


class TestAlpacaHeaders(unittest.TestCase):
    def test_valid_headers(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.alpaca_api_key = "valid_key"
            mock_settings.alpaca_secret_key = "valid_secret"
            headers = alpaca_headers()
            self.assertEqual(headers["APCA-API-KEY-ID"], "valid_key")
            self.assertEqual(headers["APCA-API-SECRET-KEY"], "valid_secret")

    def test_empty_api_key_raises(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.alpaca_api_key = ""
            mock_settings.alpaca_secret_key = "valid_secret"
            with self.assertRaises(ValueError):
                alpaca_headers()

    def test_none_secret_key_raises(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.alpaca_api_key = "valid_key"
            mock_settings.alpaca_secret_key = None
            with self.assertRaises(ValueError):
                alpaca_headers()


if __name__ == "__main__":
    unittest.main()