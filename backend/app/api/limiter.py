"""Shared slowapi rate limiter instance.

Provides a singleton `limiter` used across the API for rate limiting.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

__all__ = ["limiter"]