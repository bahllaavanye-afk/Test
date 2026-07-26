import base64
import hashlib
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, Set

import bcrypt
from cryptography.fernet import Fernet
from jose import JWTError, jwt

from app.config import settings

# NOTE: The bcrypt library is used directly to avoid compatibility issues with
# passlib. The output format remains the standard ``$2b$`` hash, so hashes
# generated previously continue to verify correctly.

_revoked_tokens: Set[str] = set()


def _bcrypt_bytes(password: str) -> bytes:
    """Encode a password for bcrypt, respecting its 72‑byte input limit.

    bcrypt only considers the first 72 bytes of the input; newer versions raise
    an exception if the input exceeds this limit. Truncating mirrors bcrypt's
    native behaviour.
    """
    return password.encode("utf-8")[:72]


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    return bcrypt.hashpw(_bcrypt_bytes(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(_bcrypt_bytes(plain), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def _fernet_key() -> bytes:
    """Derive a stable Fernet key from the application secret."""
    # The secret_key is expected to be a sufficiently random string.
    # Derive a 32‑byte key via SHA‑256 and encode it for Fernet.
    digest = hashlib.sha256(settings.secret_key.encode()).digest()
    return base64.urlsafe_b64encode(digest)


def encrypt_secret(value: str) -> str:
    """Encrypt a broker API secret for safe storage."""
    f = Fernet(_fernet_key())
    return f.encrypt(value.encode()).decode()


def decrypt_secret(encrypted: str) -> str:
    """Decrypt a previously encrypted broker API secret."""
    f = Fernet(_fernet_key())
    return f.decrypt(encrypted.encode()).decode()


def _build_common_claims(subject: str | Any, token_type: str, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Construct the JWT payload with common claims and optional extensions."""
    now = datetime.now(UTC)
    payload: Dict[str, Any] = {
        "sub": str(subject),
        "iat": now,
        "jti": str(uuid.uuid4()),
        "type": token_type,
    }
    if extra:
        payload.update(extra)
    return payload


def create_access_token(subject: str | Any, expires_delta: timedelta | None = None) -> str:
    """Create a short‑lived access token.

    The token includes `iat` (issued‑at) and `jti` (unique identifier) claims.
    """
    expire = datetime.now(UTC) + (expires_delta or timedelta(minutes=settings.access_token_expire_minutes))
    payload = _build_common_claims(subject, "access", {"exp": expire})
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def create_refresh_token(subject: str | Any) -> str:
    """Create a long‑lived refresh token.

    Refresh tokens are distinguished by the `type` claim and contain a unique `jti`
    to enable per‑token revocation.
    """
    expire = datetime.now(UTC) + timedelta(days=settings.refresh_token_expire_days)
    payload = _build_common_claims(subject, "refresh", {"exp": expire})
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_token(token: str, expected_type: str | None = None) -> Dict[str, Any]:
    """Decode and validate a JWT.

    Raises:
        JWTError: If the token is malformed, expired, or fails verification.
        ValueError: If the token type does not match `expected_type`.
    """
    try:
        claims = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    except JWTError as exc:
        raise JWTError(f"Token decode failed: {exc}") from exc

    # Verify token type if an expectation is provided.
    token_type = claims.get("type")
    if expected_type and token_type != expected_type:
        raise ValueError(f"Unexpected token type: {token_type!r}, expected {expected_type!r}")

    # Check revocation status.
    jti = claims.get("jti")
    if jti and jti in _revoked_tokens:
        raise ValueError("Token has been revoked")

    return claims


def revoke_token(token: str) -> None:
    """Mark a token as revoked using its `jti` claim."""
    try:
        claims = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm], options={"verify_exp": False})
        jti = claims.get("jti")
        if jti:
            _revoked_tokens.add(jti)
    except JWTError:
        # If the token cannot be decoded, we cannot revoke it.
        pass


def is_token_revoked(token: str) -> bool:
    """Return ``True`` if the token's `jti` is present in the revocation set."""
    try:
        claims = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm], options={"verify_exp": False})
        jti = claims.get("jti")
        return jti in _revoked_tokens if jti else False
    except JWTError:
        return False