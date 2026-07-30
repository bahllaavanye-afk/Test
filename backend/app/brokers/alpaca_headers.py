"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
from app.config import settings


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests."""
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key,
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
    }


# Unit tests for alpaca_headers
import unittest
from unittest.mock import patch


class TestAlpacaHeaders(unittest.TestCase):
    def test_headers_return_correct_keys_and_values(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.alpaca_api_key = "key123"
            mock_settings.alpaca_secret_key = "secret456"
            headers = alpaca_headers()
            self.assertIsInstance(headers, dict)
            self.assertEqual(headers["APCA-API-KEY-ID"], "key123")
            self.assertEqual(headers["APCA-API-SECRET-KEY"], "secret456")
            self.assertEqual(set(headers.keys()), {"APCA-API-KEY-ID", "APCA-API-SECRET-KEY"})

    def test_headers_with_empty_strings(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.alpaca_api_key = ""
            mock_settings.alpaca_secret_key = ""
            headers = alpaca_headers()
            self.assertEqual(headers["APCA-API-KEY-ID"], "")
            self.assertEqual(headers["APCA-API-SECRET-KEY"], "")

    def test_headers_immutable_after_settings_change(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.alpaca_api_key = "initial"
            mock_settings.alpaca_secret_key = "secret"
            headers = alpaca_headers()
            # Change settings after header creation
            mock_settings.alpaca_api_key = "changed"
            mock_settings.alpaca_secret_key = "changed2"
            # Ensure original dict unchanged
            self.assertEqual(headers["APCA-API-KEY-ID"], "initial")
            self.assertEqual(headers["APCA-API-SECRET-KEY"], "secret")


if __name__ == "__main__":
    unittest.main()