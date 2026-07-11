"""Shared Alpaca authentication header builder.

Every strategy and API handler that calls the Alpaca REST API should import
this function instead of duplicating the header dict inline.
"""
from app.config import settings


def alpaca_headers() -> dict[str, str]:
    """Return Alpaca authentication headers for REST requests."""
    return {
        "APCA-API-KEY-ID": settings.alpga_api_key,
        "APCA-API-SECRET-KEY": settings.alpga_secret_key,
    }


# ---- Unit Tests ----
# These tests focus on boundary conditions such as empty strings and None values.
# They use unittest.mock to temporarily replace the configuration values.

import unittest
from unittest.mock import patch


class TestAlpacaHeaders(unittest.TestCase):
    def test_headers_include_correct_keys(self):
        """Standard case: ensure both required keys are present."""
        with patch.object(settings, "alpga_api_key", "test_key"), \
             patch.object(settings, "alpga_secret_key", "test_secret"):
            hdr = alpaca_headers()
            self.assertIn("APCA-API-KEY-ID", hdr)
            self.assertIn("APCA-API-SECRET-KEY", hdr)
            self.assertEqual(hdr["APCA-API-KEY-ID"], "test_key")
            self.assertEqual(hdr["APCA-API-SECRET-KEY"], "test_secret")

    def test_headers_with_empty_strings(self):
        """Edge case: API credentials are empty strings."""
        with patch.object(settings, "alpga_api_key", ""), \
             patch.object(settings, "alpga_secret_key", ""):
            hdr = alpaca_headers()
            self.assertEqual(hdr["APCA-API-KEY-ID"], "")
            self.assertEqual(hdr["APCA-API-SECRET-KEY"], "")

    def test_headers_with_none_values(self):
        """Edge case: API credentials are None (should still be present)."""
        with patch.object(settings, "alpga_api_key", None), \
             patch.object(settings, "alpga_secret_key", None):
            hdr = alpaca_headers()
            self.assertIsNone(hdr["APCA-API-KEY-ID"])
            self.assertIsNone(hdr["APCA-API-SECRET-KEY"])


if __name__ == "__main__":
    unittest.main()