"""Shared slowapi rate limiter instance."""
from slowapi import Limiter
from slowapi.util import get_remote_address


def _validate_key_func(key_func):
    """Validate that the provided key function is callable.

    Args:
        key_func: Function used by SlowAPI to extract a client identifier.

    Raises:
        ValueError: If ``key_func`` is not callable.
    """
    if not callable(key_func):
        raise ValueError("key_func must be a callable that returns a client identifier.")
    return True


# Validate the default key function before creating the limiter instance.
_validate_key_func(get_remote_address)

limiter = Limiter(key_func=get_remote_address)