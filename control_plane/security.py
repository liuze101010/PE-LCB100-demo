from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519


PBKDF2_ITERATIONS = 600_000


def hash_password(password: str) -> str:
    if len(password) < 12:
        raise ValueError("password must contain at least 12 characters")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, rounds, salt_hex, digest_hex = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(rounds)
        )
        return hmac.compare_digest(actual.hex(), digest_hex)
    except (TypeError, ValueError):
        return False


def new_session(hours: int) -> tuple[str, str, datetime]:
    token = secrets.token_urlsafe(40)
    csrf = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(hours=hours)
    return token, csrf, expires


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _fernet(secret: str) -> Fernet:
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    return Fernet(key)


def encrypt_secret(value: str, key: str) -> str:
    return _fernet(key).encrypt(value.encode()).decode()


def decrypt_secret(value: str, key: str) -> str:
    return _fernet(key).decrypt(value.encode()).decode()


def generate_ssh_keypair() -> tuple[str, str]:
    private_key = ed25519.Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_text = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH,
    ).decode()
    return private_pem, public_text


def sha256_json(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()
