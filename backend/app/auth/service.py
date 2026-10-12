from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.auth.security import (
    generate_session_token,
    hash_password,
    hash_session_token,
    verify_password,
)
from backend.app.db.models import Session as DbSession
from backend.app.db.models import User


SESSION_DURATION_HOURS = 8


class AuthenticationError(Exception):
    """Raised when authentication credentials are invalid."""


class DuplicateEmailError(Exception):
    """Raised when an email is already registered."""


class AuthenticationService:
    """Business logic for registration, login, sessions, and logout."""

    def __init__(self, db: Session) -> None:
        self.db = db

    @staticmethod
    def normalize_email(email: str) -> str:
        """Normalize email addresses before storage and lookup."""
        return email.strip().lower()

    @staticmethod
    def validate_password(password: str) -> None:
        """Validate minimum password requirements."""
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long.")

        if not any(character.isupper() for character in password):
            raise ValueError("Password must contain at least one uppercase letter.")

        if not any(character.islower() for character in password):
            raise ValueError("Password must contain at least one lowercase letter.")

        if not any(character.isdigit() for character in password):
            raise ValueError("Password must contain at least one digit.")

    def register(self, name: str, email: str, password: str) -> User:
        """Create a new user account."""
        normalized_email = self.normalize_email(email)

        if not name.strip():
            raise ValueError("Name cannot be empty.")

        if "@" not in normalized_email:
            raise ValueError("Invalid email address.")

        self.validate_password(password)

        existing_user = self.db.scalar(
            select(User).where(User.email == normalized_email)
        )

        if existing_user is not None:
            raise DuplicateEmailError("Email is already registered.")

        user = User(
            name=name.strip(),
            email=normalized_email,
            password_hash=hash_password(password),
            role="user",
        )

        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)

        return user

    def authenticate(self, email: str, password: str) -> User:
        """Authenticate a user without revealing whether the email exists."""
        normalized_email = self.normalize_email(email)

        user = self.db.scalar(
            select(User).where(User.email == normalized_email)
        )

        if user is None or not verify_password(password, user.password_hash):
            raise AuthenticationError("Invalid email or password.")

        user.last_login_at = datetime.now(timezone.utc)
        self.db.commit()
        self.db.refresh(user)

        return user

    def create_session(self, user: User) -> str:
        """Create a server-side session and return the raw token once."""
        raw_token = generate_session_token()
        token_hash = hash_session_token(raw_token)

        now = datetime.now(timezone.utc)

        session = DbSession(
            user_id=user.id,
            token_hash=token_hash,
            created_at=now,
            expires_at=now + timedelta(hours=SESSION_DURATION_HOURS),
        )

        self.db.add(session)
        self.db.commit()

        return raw_token

    def get_user_from_token(self, raw_token: str) -> User | None:
        """Resolve a valid, unexpired session token to its user."""
        if not raw_token:
            return None

        token_hash = hash_session_token(raw_token)

        session = self.db.scalar(
            select(DbSession).where(DbSession.token_hash == token_hash)
        )

        if session is None:
            return None
        now = datetime.now(timezone.utc)
        expires_at = session.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        if expires_at <= now:
            self.db.delete(session)
            self.db.commit()
            return None

        return session.user

    def logout(self, raw_token: str) -> None:
        """Invalidate a session server-side."""
        if not raw_token:
            return

        token_hash = hash_session_token(raw_token)

        session = self.db.scalar(
            select(DbSession).where(DbSession.token_hash == token_hash)
        )

        if session is not None:
            self.db.delete(session)
            self.db.commit()
