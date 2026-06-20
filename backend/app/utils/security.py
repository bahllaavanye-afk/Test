import base64
import hashlib
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
from cryptography.fernet import Fernet
from jose import jwt

from app.config import settings

# Use the bcrypt library directly rather than passlib's bcrypt backend: passlib 1.7.4
# cannot read the version of bcrypt >= 4.1 (`module 'bcrypt' has no attribute '__about__'`)
# and crashes password hashing. The output is still a standard ``$2b$`` hash, so any
# hashes previously produced by passlib continue to verify unchanged.


def _bcrypt_bytes(password: str) -> bytes:
    """Encode a password for bcrypt, honoring its 72-byte input limit.

    bcrypt only considers the first 72 bytes of the input, and bcrypt >= 5 raises if
    given more, so we truncate to 72 bytes — semantically identical to bcrypt's own
    behavior and the standard way to handle long passwords.
    """
    return password.encode("utf-8")[:72]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_bcrypt_bytes(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_bcrypt_bytes(plain), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(subject: str | Any, expires_delta: timedelta | None = None) -> str:
    expire = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    return jwt.encode({"sub": str(subject), "exp": expire, "type": "access"}, settings.secret_key, algorithm=settings.algorithm)


def create_refresh_token(subject: str | Any) -> str:
    expire = datetime.now(UTC) + timedelta(days=settings.refresh_token_expire_days)
    payload = {
        "sub": str(subject),
        "exp": expire,
        "type": "refresh",
        "jti": str(uuid.uuid4()),  # unique ID enables per-token revocation
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])


def _fernet_key() -> bytes:
    """Derive a stable Fernet key from the secret_key."""
    digest = hashlib.sha256(settings.secret_key.encode()).digest()
    return base64.urlsafe_b64encode(digest)


def encrypt_secret(value: str) -> str:
    """AES-256 encrypt a broker API secret for storage."""
    f = Fernet(_fernet_key())
    return f.encrypt(value.encode()).decode()


def decrypt_secret(encrypted: str) -> str:
    """Decrypt a stored broker API secret."""
    f = Fernet(_fernet_key())
    return f.decrypt(encrypted.encode()).decode()
