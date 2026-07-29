import base64
import hashlib
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Dict

import bcrypt
from cryptography.fernet import Fernet
from jose import jwt

from app.config import settings


def _bcrypt_bytes(password: str) -> bytes:
    """
    Encode a password for bcrypt, respecting its 72‑byte input limit.

    bcrypt only processes the first 72 bytes of the input. Newer versions raise an
    exception if more bytes are provided, so we truncate the UTF‑8 encoded password
    to 72 bytes to mirror bcrypt's native behavior.

    Args:
        password: The plaintext password to encode.

    Returns:
        A bytes object containing at most the first 72 UTF‑8 bytes of the password.
    """
    return password.encode("utf-8")[:72]


def hash_password(password: str) -> str:
    """
    Generate a bcrypt hash for a given password.

    The function uses ``bcrypt.gensalt`` to create a new salt and returns the
    resulting hash as a UTF‑8 string.

    Args:
        password: The plaintext password to hash.

    Returns:
        The bcrypt hash string.
    """
    return bcrypt.hashpw(_bcrypt_bytes(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """
    Verify a plaintext password against a stored bcrypt hash.

    Any exception raised by ``bcrypt.checkpw`` (e.g., due to malformed input) is
    caught and treated as a verification failure.

    Args:
        plain: The plaintext password to verify.
        hashed: The stored bcrypt hash.

    Returns:
        ``True`` if the password matches the hash, otherwise ``False``.
    """
    try:
        return bcrypt.checkpw(_bcrypt_bytes(plain), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(subject: Any, expires_delta: timedelta | None = None) -> str:
    """
    Create a JWT access token.

    The token includes the subject identifier, an expiration timestamp, and a
    token type of ``access``. If ``expires_delta`` is not provided, the default
    expiration defined in settings is used.

    Args:
        subject: Identifier for the token's owner (e.g., user ID or email).
        expires_delta: Optional custom expiration delta.

    Returns:
        A signed JWT string.
    """
    expire = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    return jwt.encode(
        {"sub": str(subject), "exp": expire, "type": "access"},
        settings.secret_key,
        algorithm=settings.algorithm,
    )


def create_refresh_token(subject: Any) -> str:
    """
    Create a JWT refresh token.

    The token includes a unique JWT ID (`jti`) to enable per‑token revocation,
    an expiration based on the refresh token lifetime, and a token type of
    ``refresh``.

    Args:
        subject: Identifier for the token's owner.

    Returns:
        A signed JWT string.
    """
    expire = datetime.now(UTC) + timedelta(days=settings.refresh_token_expire_days)
    payload = {
        "sub": str(subject),
        "exp": expire,
        "type": "refresh",
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and verify a JWT token.

    The function validates the signature using the configured secret key and
    returns the token payload as a dictionary.

    Args:
        token: The JWT string to decode.

    Returns:
        The decoded token payload.
    """
    return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])


def _fernet_key() -> bytes:
    """
    Derive a stable Fernet key from the application's secret key.

    The secret key is SHA‑256 hashed and then base64‑url‑safe encoded to produce a
    32‑byte key suitable for Fernet encryption.

    Returns:
        A bytes object representing the Fernet key.
    """
    digest = hashlib.sha256(settings.secret_key.encode()).digest()
    return base64.urlsafe_b64encode(digest)


def encrypt_secret(value: str) -> str:
    """
    Encrypt a broker API secret using AES‑256 (via Fernet).

    Args:
        value: The plaintext secret to encrypt.

    Returns:
        The encrypted secret as a UTF‑8 string.
    """
    f = Fernet(_fernet_key())
    return f.encrypt(value.encode()).decode()


def decrypt_secret(encrypted: str) -> str:
    """
    Decrypt a previously encrypted broker API secret.

    Args:
        encrypted: The encrypted secret string.

    Returns:
        The original plaintext secret.
    """
    f = Fernet(_fernet_key())
    return f.decrypt(encrypted.encode()).decode()