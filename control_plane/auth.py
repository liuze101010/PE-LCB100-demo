from __future__ import annotations

from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import User
from .security import verify_password


class AuthBackend(Protocol):
    name: str

    def authenticate(self, db: Session, username: str, password: str) -> User | None: ...


class LocalAuthBackend:
    name = "local"

    def authenticate(self, db: Session, username: str, password: str) -> User | None:
        user = db.scalar(select(User).where(User.username == username))
        if user is None or not user.active or not verify_password(password, user.password_hash):
            return None
        return user


def get_auth_backend() -> AuthBackend:
    # OIDC/LDAP adapters plug in here once the institution confirms its protocol.
    return LocalAuthBackend()
