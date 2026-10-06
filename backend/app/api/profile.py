from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.auth import get_current_user
from backend.app.auth.security import hash_password, verify_password
from backend.app.auth.service import AuthenticationService
from backend.app.db.database import get_db
from backend.app.db.models import AuditLog, User


router = APIRouter(
    prefix="/api",
    tags=["profile"],
)


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str
    confirm_password: str


@router.get("/profile")
def get_profile(
    user: User = Depends(get_current_user),
):
    """Return the authenticated user's account information."""

    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role,
        "created_at": user.created_at,
        "last_login_at": user.last_login_at,
    }


@router.patch("/profile/password")
def change_password(
    payload: ChangePasswordRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Change the authenticated user's password."""

    if payload.new_password != payload.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords do not match.",
        )

    if not verify_password(
        payload.current_password,
        user.password_hash,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect.",
        )

    try:
        AuthenticationService.validate_password(
            payload.new_password,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    user.password_hash = hash_password(payload.new_password)

    audit_entry = AuditLog(
        user_id=user.id,
        action="password_changed",
        detail="User changed account password.",
    )

    db.add(audit_entry)
    db.commit()

    return {
        "message": "Password changed successfully.",
    }


@router.get("/audit-log")
def get_audit_log(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the authenticated user's append-only audit history."""

    entries = db.scalars(
        select(AuditLog)
        .where(AuditLog.user_id == user.id)
        .order_by(AuditLog.created_at.desc())
    ).all()

    return {
        "entries": [
            {
                "id": entry.id,
                "action": entry.action,
                "detail": entry.detail,
                "created_at": entry.created_at,
            }
            for entry in entries
        ]
    }