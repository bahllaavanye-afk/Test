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


# ----------------------------------------------------------------------
# Unit tests for alpaca_headers
# ----------------------------------------------------------------------
import unittest
from unittest.mock import patch


class TestAlpacaHeaders(unittest.TestCase):
    """Edge‑case tests for the alpaca_headers helper."""

    def test_headers_contains_expected_keys(self):
        """Headers should always contain the two Alpaca auth keys."""
        with patch.object(settings, "alpaca_api_key", "dummy_key"), \
             patch.object(settings, "alpaca_secret_key", "dummy_secret"):
            hdr = alpaca_headers()
            self.assertIn("APCA-API-KEY-ID", hdr)
            self.assertIn("APCA-API-SECRET-KEY", hdr)

    def test_headers_reflect_settings_values(self):
        """Returned values must match whatever is present in settings, even if empty."""
        test_cases = [
            ("key123", "secret456"),
            ("", ""),
            ("   spaced_key   ", "\nnewlines\n"),
        ]
        for api_key, secret_key in test_cases:
            with self.subTest(api_key=api_key, secret_key=secret_key):
                with patch.object(settings, "alpaca_api_key", api_key), \
                     patch.object(settings, "alpaca_secret_key", secret_key):
                    hdr = alpaca_headers()
                    self.assertEqual(hdr["APCA-API-KEY-ID"], api_key)
                    self.assertEqual(hdr["APCA-API-SECRET-KEY"], secret_key)

    def test_headers_are_independent_instances(self):
        """Modifying a returned dict must not affect subsequent calls."""
        with patch.object(settings, "alpaca_api_key", "first_key"), \
             patch.object(settings, "alpaca_secret_key", "first_secret"):
            hdr1 = alpaca_headers()
            hdr1["APCA-API-KEY-ID"] = "modified"
            hdr2 = alpaca_headers()
            # hdr2 should reflect the original settings, not the mutated hdr1
            self.assertEqual(hdr2["APCA-API-KEY-ID"], "first_key")
            self.assertEqual(hdr2["APCA-API-SECRET-KEY"], "first_secret")


if __name__ == "__main__":
    unittest.main()