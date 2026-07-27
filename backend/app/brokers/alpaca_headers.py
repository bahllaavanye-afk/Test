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


# Unit tests for edge cases
import unittest
from unittest.mock import patch


class TestAlpacaHeaders(unittest.TestCase):
    def test_headers_contain_required_keys(self):
        """Headers must include both authentication keys."""
        hdr = alpaca_headers()
        self.assertIn("APCA-API-KEY-ID", hdr)
        self.assertIn("APCA-API-SECRET-KEY", hdr)

    def test_headers_values_match_settings(self):
        """Returned values should reflect the current settings."""
        with patch.object(settings, "alpaca_api_key", "test_key"), \
             patch.object(settings, "alpaca_secret_key", "test_secret"):
            hdr = alpaca_headers()
            self.assertEqual(hdr["APCA-API-KEY-ID"], "test_key")
            self.assertEqual(hdr["APCA-API-SECRET-KEY"], "test_secret")

    def test_headers_with_empty_strings(self):
        """Empty string credentials should be propagated unchanged."""
        with patch.object(settings, "alpaca_api_key", ""), \
             patch.object(settings, "alpaca_secret_key", ""):
            hdr = alpaca_headers()
            self.assertEqual(hdr["APCA-API-KEY-ID"], "")
            self.assertEqual(hdr["APCA-API-SECRET-KEY"], "")


if __name__ == "__main__":
    unittest.main()