from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from backend.app.auth.service import (
    AuthenticationError,
    AuthenticationService,
    DuplicateEmailError,
)
from backend.app.db.database import Base
from backend.app.db.models import Session as DbSession
from backend.app.db.models import User


@pytest.fixture()
def db():
    """Create an isolated in-memory database for each test."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    Base.metadata.create_all(bind=engine)

    with Session(engine) as session:
        yield session


def test_register_creates_user(db):
    service = AuthenticationService(db)

    user = service.register(
        name="Test User",
        email="Test@Example.com",
        password="StrongPassword1",
    )

    assert user.id is not None
    assert user.email == "test@example.com"
    assert user.password_hash != "StrongPassword1"

    stored_user = db.scalar(
        select(User).where(User.email == "test@example.com")
    )

    assert stored_user is not None


def test_duplicate_registration_is_rejected(db):
    service = AuthenticationService(db)

    service.register(
        name="First User",
        email="same@example.com",
        password="StrongPassword1",
    )

    with pytest.raises(DuplicateEmailError):
        service.register(
            name="Second User",
            email="SAME@example.com",
            password="StrongPassword1",
        )


def test_weak_password_is_rejected(db):
    service = AuthenticationService(db)

    with pytest.raises(ValueError):
        service.register(
            name="Test User",
            email="weak@example.com",
            password="weak",
        )


def test_wrong_login_is_rejected(db):
    service = AuthenticationService(db)

    service.register(
        name="Test User",
        email="login@example.com",
        password="StrongPassword1",
    )

    with pytest.raises(AuthenticationError):
        service.authenticate(
            email="login@example.com",
            password="WrongPassword1",
        )


def test_correct_login_creates_session(db):
    service = AuthenticationService(db)

    user = service.register(
        name="Test User",
        email="login@example.com",
        password="StrongPassword1",
    )

    authenticated_user = service.authenticate(
        email="LOGIN@example.com",
        password="StrongPassword1",
    )

    assert authenticated_user.id == user.id
    assert authenticated_user.last_login_at is not None

    token = service.create_session(authenticated_user)

    assert token
    assert len(token) > 20

    stored_session = db.scalar(select(DbSession))

    assert stored_session is not None
    assert stored_session.token_hash != token

    resolved_user = service.get_user_from_token(token)

    assert resolved_user is not None
    assert resolved_user.id == user.id


def test_logout_invalidates_session(db):
    service = AuthenticationService(db)

    user = service.register(
        name="Test User",
        email="logout@example.com",
        password="StrongPassword1",
    )

    token = service.create_session(user)

    assert service.get_user_from_token(token) is not None

    service.logout(token)

    assert service.get_user_from_token(token) is None


def test_expired_session_is_rejected(db):
    service = AuthenticationService(db)

    user = service.register(
        name="Test User",
        email="expired@example.com",
        password="StrongPassword1",
    )

    token = service.create_session(user)

    stored_session = db.scalar(select(DbSession))
    assert stored_session is not None

    stored_session.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    db.commit()

    assert service.get_user_from_token(token) is None


def test_timezone_aware_expired_session_is_rejected():
    expired_record = type(
        "SessionRecord",
        (),
        {
            "expires_at": datetime.now(timezone.utc) - timedelta(minutes=1),
            "user": None,
        },
    )()

    class SessionStore:
        deleted = None
        committed = False

        def scalar(self, _statement):
            return expired_record

        def delete(self, record):
            self.deleted = record

        def commit(self):
            self.committed = True

    db = SessionStore()
    service = AuthenticationService(db)

    assert service.get_user_from_token("test-session-token") is None
    assert db.deleted is expired_record
    assert db.committed
