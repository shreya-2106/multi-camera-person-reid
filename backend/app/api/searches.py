from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.auth import get_current_user
from backend.app.db.database import get_db
from backend.app.db.models import AuditLog, Search, SearchResult, User
from backend.app.services.mock_analysis import run_mock_analysis


router = APIRouter(
    prefix="/api/searches",
    tags=["searches"],
)


BACKEND_DIR = Path(__file__).resolve().parents[2]
STORAGE_DIR = BACKEND_DIR / "storage"
SEARCH_STORAGE_DIR = STORAGE_DIR / "searches"

ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
}


def _get_owned_search(
    search_id: int,
    user_id: int,
    db: Session,
) -> Search:
    search = db.scalar(
        select(Search).where(
            Search.id == search_id,
            Search.user_id == user_id,
        )
    )

    if search is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Search not found.",
        )

    return search


def _validate_video(filename: str | None) -> str:
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Video filename is required.",
        )

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Unsupported video format. "
                "Use MP4, AVI, MOV, MKV, or M4V."
            ),
        )

    return extension


async def _save_upload(
    upload: UploadFile,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with destination.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)

            if not chunk:
                break

            handle.write(chunk)

    await upload.close()


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_search(
    case_name: str = Form("Untitled search"),
    videos: list[UploadFile] = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a search, persist uploaded videos, and run mock analysis."""

    if not videos:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one video is required.",
        )

    if len(videos) > 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A maximum of 8 cameras is supported.",
        )

    cleaned_case_name = case_name.strip()

    if not cleaned_case_name:
        cleaned_case_name = "Untitled search"

    search = Search(
        user_id=user.id,
        case_name=cleaned_case_name,
        camera_count=len(videos),
        status="processing",
    )

    db.add(search)
    db.commit()
    db.refresh(search)

    search_directory = SEARCH_STORAGE_DIR / str(search.id)

    try:
        camera_names: list[str] = []

        for index, upload in enumerate(videos, start=1):
            extension = _validate_video(upload.filename)

            camera_name = f"CAM{index:02d}"
            camera_names.append(camera_name)

            destination = (
                search_directory
                / "videos"
                / f"{camera_name}{extension}"
            )

            await _save_upload(
                upload,
                destination,
            )

        result = run_mock_analysis(
            camera_names=camera_names,
        )

        search.status = (
            "found"
            if result["match_found"]
            else "not_found"
        )

        search_result = SearchResult(
            search_id=search.id,
            identity_label=result.get("identity_label"),
            overall_confidence=result.get("overall_confidence"),
            result_json=json.dumps(
                {
                    "videos": [
                        {
                            "camera": camera_names[index],
                            "filename": videos[index].filename,
                        }
                        for index in range(len(videos))
                    ],
                    "result": result,
                }
            ),
        )

        db.add(search_result)

        db.add(
            AuditLog(
                user_id=user.id,
                action="search_created",
                detail=f"Search {search.id} created.",
            )
        )

        db.commit()
        db.refresh(search)
        db.refresh(search_result)

        return {
            "id": search.id,
            "case_name": search.case_name,
            "camera_count": search.camera_count,
            "status": search.status,
            "created_at": search.created_at,
            "result": result,
        }

    except HTTPException:
        search.status = "error"
        db.commit()

        if search_directory.exists():
            shutil.rmtree(search_directory)

        raise

    except Exception:
        search.status = "error"
        db.commit()

        if search_directory.exists():
            shutil.rmtree(search_directory)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search processing failed.",
        )


@router.get("")
def list_searches(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return only searches belonging to the authenticated user."""

    searches = db.scalars(
        select(Search)
        .where(Search.user_id == user.id)
        .order_by(Search.created_at.desc())
    ).all()

    return {
        "searches": [
            {
                "id": search.id,
                "case_name": search.case_name,
                "created_at": search.created_at,
                "camera_count": search.camera_count,
                "status": search.status,
            }
            for search in searches
        ]
    }


@router.get("/{search_id}")
def get_search(
    search_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return a saved search and its stored result."""

    search = _get_owned_search(
        search_id,
        user.id,
        db,
    )

    result = db.scalar(
        select(SearchResult).where(
            SearchResult.search_id == search.id,
        )
    )

    if result is None:
        result_payload = None
    else:
        result_payload = json.loads(result.result_json)

    return {
        "id": search.id,
        "case_name": search.case_name,
        "created_at": search.created_at,
        "camera_count": search.camera_count,
        "status": search.status,
        "result": result_payload,
    }


@router.post("/{search_id}/flag")
def flag_search(
    search_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Flag a saved result as potentially incorrect."""

    search = _get_owned_search(
        search_id,
        user.id,
        db,
    )

    result = db.scalar(
        select(SearchResult).where(
            SearchResult.search_id == search.id,
        )
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Search result not found.",
        )

    payload = json.loads(result.result_json)
    payload["flagged"] = True

    result.result_json = json.dumps(payload)

    db.add(
        AuditLog(
            user_id=user.id,
            action="search_flagged",
            detail=f"Search {search.id} flagged as incorrect.",
        )
    )

    db.commit()

    return {
        "message": "Search result flagged.",
        "search_id": search.id,
        "flagged": True,
    }


@router.delete("/{search_id}")
def delete_search(
    search_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete an owned search, result, and associated video files."""

    search = _get_owned_search(
        search_id,
        user.id,
        db,
    )

    search_directory = SEARCH_STORAGE_DIR / str(search.id)

    db.delete(search)

    db.add(
        AuditLog(
            user_id=user.id,
            action="search_deleted",
            detail=f"Search {search.id} deleted.",
        )
    )

    db.commit()

    if search_directory.exists():
        shutil.rmtree(search_directory)

    return {
        "message": "Search deleted.",
        "search_id": search_id,
    }