import base64
import hashlib
import uuid
import logging
import json
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any, Dict

import bcrypt
from cryptography.fernet import Fernet
from jose import jwt

from app.config import settings

logger = logging.getLogger(__name__)

_call_counters = defaultdict(int)


def _monitor(func):
    """Decorator to log execution metrics for security utilities."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration_ms = (time.perf_counter() - start) * 1000
        _call_counters[func.__name__] += 1
        signal_count = _call_counters[func.__name__]
        pnl = result if isinstance(result, (int, float)) else None
        log_payload = {
            "function": func.__name__,
            "signal_count": signal_count,
            "duration_ms": round(duration_ms, 3),
            "pnl": pnl,
        }
        logger.info(json.dumps(log_payload))
        return result
    return wrapper


def _bcrypt_bytes(password: str) -> bytes:
    """Encode a password for bcrypt, respecting its 72‑byte input limit.

    bcrypt only processes the first 72 bytes of the input. Newer versions raise an
    exception if more than 72 bytes are supplied, so we truncate to match bcrypt's
    intrinsic behavior.

    Args:
        password: The plain‑text password to encode.

    Returns:
        The UTF‑8 encoded password truncated to 72 bytes.
    """
    return password.encode("utf-8")[:72]


@_monitor
def hash_password(password: str) -> str:
    """Hash a plain‑text password using bcrypt.

    Args:
        password: The password to hash.

    Returns:
        The bcrypt hash as a UTF‑8 string.
    """
    return bcrypt.hashpw(_bcrypt_bytes(password), bcrypt.gensalt()).decode("utf-8")


@_monitor
def verify_password(plain: str, hashed: str) -> bool:
    """Verify a password against a stored bcrypt hash.

    Args:
        plain: The plain‑text password to verify.
        hashed: The stored bcrypt hash.

    Returns:
        True if the password matches the hash, otherwise False.
    """
    try:
        return bcrypt.checkpw(_bcrypt_bytes(plain), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


@_monitor
def create_access_token(subject: str | Any, expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token.

    Args:
        subject: Identifier for the token's subject (e.g., user ID).
        expires_delta: Optional custom expiration timedelta. If omitted,
            the default from settings is used.

    Returns:
        A signed JWT access token string.
    """
    expire = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    return jwt.encode(
        {"sub": str(subject), "exp": expire, "type": "access"},
        settings.secret_key,
        algorithm=settings.algorithm,
    )


@_monitor
def create_refresh_token(subject: str | Any) -> str:
    """Create a JWT refresh token with a unique identifier.

    Args:
        subject: Identifier for the token's subject.

    Returns:
        A signed JWT refresh token string.
    """
    expire = datetime.now(UTC) + timedelta(days=settings.refresh_token_expire_days)
    payload = {
        "sub": str(subject),
        "exp": expire,
        "type": "refresh",
        "jti": str(uuid.uuid4()),  # unique ID enables per-token revocation
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


@_monitor
def decode_token(token: str) -> Dict[str, Any]:
    """Decode a JWT token without verifying its type.

    Args:
        token: The JWT token string to decode.

    Returns:
        The payload dictionary extracted from the token.
    """
    return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])


def _fernet_key() -> bytes:
    """Derive a stable Fernet key from the application's secret key.

    Returns:
        A 32‑byte URL‑safe base64‑encoded key suitable for Fernet.
    """
    digest = hashlib.sha256(settings.secret_key.encode()).digest()
    return base64.urlsafe_b64encode(digest)


@_monitor
def encrypt_secret(value: str) -> str:
    """Encrypt a broker API secret for secure storage.

    Args:
        value: The plain‑text secret to encrypt.

    Returns:
        The encrypted secret as a UTF‑8 string.
    """
    f = Fernet(_fernet_key())
    return f.encrypt(value.encode()).decode()


@_monitor
def decrypt_secret(encrypted: str) -> str:
    """Decrypt a previously encrypted broker API secret.

    Args:
        encrypted: The encrypted secret string.

    Returns:
        The original plain‑text secret.
    """
    f = Fernet(_fernet_key())
    return f.decrypt(encrypted.encode()).decode()