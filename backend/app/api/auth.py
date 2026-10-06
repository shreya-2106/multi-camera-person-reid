from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from backend.app.auth.service import (
    AuthenticationError,
    AuthenticationService,
    DuplicateEmailError,
)
from backend.app.db.database import get_db


router = APIRouter(
    prefix="/api/auth",
    tags=["authentication"],
)


SESSION_COOKIE_NAME = "session_token"


class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    confirm_password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
):
    """Return the authenticated user or raise HTTP 401."""

    token = request.cookies.get(SESSION_COOKIE_NAME)

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )

    service = AuthenticationService(db)
    user = service.get_user_from_token(token)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )

    return user


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(
    payload: RegisterRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    """Register a new user and create an authenticated session."""

    if payload.password != payload.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords do not match.",
        )

    service = AuthenticationService(db)

    try:
        user = service.register(
            name=payload.name,
            email=str(payload.email),
            password=payload.password,
        )

        from backend.app.db.models import AuditLog
        token = service.create_session(user)
        db.add(
    AuditLog(
        user_id=user.id,
        action="login",
        detail="User logged in successfully.",))
        db.commit()
        

    except DuplicateEmailError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Registration could not be completed.",
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=8 * 60 * 60,
    )

    return {
        "message": "Registration successful.",
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "role": user.role,
        },
    }


@router.post("/login")
def login(
    payload: LoginRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    """Authenticate a user and create a server-side session."""

    service = AuthenticationService(db)

    try:
        user = service.authenticate(
            email=str(payload.email),
            password=payload.password,
        )

        token = service.create_session(user)

    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=8 * 60 * 60,
    )

    return {
        "message": "Login successful.",
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "role": user.role,
        },
    }


@router.post("/logout")
def logout(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    """Invalidate the current server-side session."""

    token = request.cookies.get(SESSION_COOKIE_NAME)

    service = AuthenticationService(db)
    service.logout(token)

    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
    )

    return {
        "message": "Logout successful.",
    }


@router.get("/me")
def me(
    user=Depends(get_current_user),
):
    """Return the currently authenticated user."""

    return {
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "role": user.role,
        }
    }