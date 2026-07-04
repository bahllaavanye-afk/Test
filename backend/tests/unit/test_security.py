"""Tests for authentication and security utilities."""
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


def test_hash_password_is_salted():
    """Hashing the same password twice should yield different results due to salting."""
    h1 = hash_password("test123")
    h2 = hash_password("test123")
    assert h1 != h2, "Expected different hashes for the same password because of salt"


def test_verify_password_success():
    """Correct password should verify against its hash."""
    pwd = "test123"
    hashed = hash_password(pwd)
    assert verify_password(pwd, hashed), "Password verification failed for a correct password"


@pytest.mark.parametrize(
    "password,expected",
    [
        ("wrong", False),
        ("", False),
        ("test1234", False),
    ],
)
def test_verify_password_failure(password: str, expected: bool):
    """Incorrect passwords must not validate."""
    hashed = hash_password("test123")
    assert verify_password(password, hashed) is expected


def test_access_token_roundtrip():
    """Access token should encode and decode the user identifier and type."""
    user_id = "user-id-123"
    token = create_access_token(user_id)
    payload = decode_token(token)
    assert payload["sub"] == user_id
    assert payload["type"] == "access"


def test_refresh_token_roundtrip():
    """Refresh token should encode and decode the user identifier and type."""
    user_id = "user-id-456"
    token = create_refresh_token(user_id)
    payload = decode_token(token)
    assert payload["sub"] == user_id
    assert payload["type"] == "refresh"


def test_encrypt_decrypt_roundtrip():
    """Encryption should be reversible and produce different ciphertext each call."""
    plain = "binance-secret-key-abc123"
    encrypted = encrypt_secret(plain)
    assert encrypted != plain, "Encrypted value should differ from plaintext"
    decrypted = decrypt_secret(encrypted)
    assert decrypted == plain, "Decrypted value does not match original plaintext"


def test_encrypt_is_nondeterministic():
    """Repeated encryption of the same plaintext must yield different ciphertexts."""
    plain = "same-secret"
    enc1 = encrypt_secret(plain)
    enc2 = encrypt_secret(plain)
    assert enc1 != enc2, "Ciphertexts should differ due to random IV"
    assert decrypt_secret(enc1) == decrypt_secret(enc2) == plain


def test_encrypt_decrypt_empty_string():
    """Encryption and decryption should correctly handle empty strings."""
    plain = ""
    encrypted = encrypt_secret(plain)
    assert encrypted != plain, "Encryption of empty string should still produce ciphertext"
    decrypted = decrypt_secret(encrypted)
    assert decrypted == plain, "Decrypted empty string does not match original"