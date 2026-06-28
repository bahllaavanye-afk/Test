"""Auth security helpers tests."""
import time
from datetime import datetime, timezone, timedelta

import pytest

from app.utils.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    encrypt_secret,
    decrypt_secret,
)


def test_hash_password_different_each_time():
    h1 = hash_password("test123")
    h2 = hash_password("test123")
    assert h1 != h2  # bcrypt uses salt


def test_verify_password_correct():
    h = hash_password("test123")
    assert verify_password("test123", h)


def test_verify_password_wrong():
    h = hash_password("test123")
    assert not verify_password("wrong", h)


def test_access_token_roundtrip():
    token = create_access_token("user-id-123")
    payload = decode_token(token)
    assert payload["sub"] == "user-id-123"
    assert payload["type"] == "access"


def test_refresh_token_roundtrip():
    token = create_refresh_token("user-id-456")
    payload = decode_token(token)
    assert payload["sub"] == "user-id-456"
    assert payload["type"] == "refresh"


def test_encrypt_decrypt_roundtrip():
    plain = "binance-secret-key-abc123"
    enc = encrypt_secret(plain)
    assert enc != plain
    dec = decrypt_secret(enc)
    assert dec == plain


def test_encrypt_different_each_time():
    plain = "same-secret"
    e1 = encrypt_secret(plain)
    e2 = encrypt_secret(plain)
    # Fernet adds random IV, so ciphertext differs
    assert e1 != e2
    assert decrypt_secret(e1) == decrypt_secret(e2) == plain


def test_access_token_has_exp_claim():
    token = create_access_token("user-id-789")
    payload = decode_token(token)
    assert "exp" in payload
    exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
    now = datetime.now(tz=timezone.utc)
    # Token expiry should be in the future (allow a small drift)
    assert exp > now - timedelta(seconds=5)


def test_refresh_token_has_exp_claim():
    token = create_refresh_token("user-id-101")
    payload = decode_token(token)
    assert "exp" in payload
    exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
    now = datetime.now(tz=timezone.utc)
    assert exp > now - timedelta(seconds=5)


def test_decode_invalid_token_raises():
    # Completely malformed token
    with pytest.raises(Exception):
        decode_token("not-a-valid-token")

    # Correctly signed token but with tampered payload
    token = create_access_token("user-id-202")
    parts = token.split(".")
    # Corrupt the payload part (second segment)
    tampered = f"{parts[0]}.{'A' * len(parts[1])}.{parts[2]}"
    with pytest.raises(Exception):
        decode_token(tampered)


def test_decrypt_invalid_ciphertext_raises():
    # Random string that is not valid Fernet ciphertext
    with pytest.raises(Exception):
        decrypt_secret("invalid-ciphertext")