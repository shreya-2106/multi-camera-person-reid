from __future__ import annotations

import json
import logging
import re
import shutil
import time
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.auth import get_current_user
from backend.app.db.database import get_db
from backend.app.db.models import AuditLog, Search, SearchResult, User
from backend.app.services.real_analysis import run_real_analysis
from src.pipeline.types import build_matched_person_summary
from src.temporal.track_summary import TrackSummaryBuilder

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/searches",
    tags=["searches"],
)


BACKEND_DIR = Path(__file__).resolve().parents[2]
STORAGE_DIR = BACKEND_DIR / "storage"
SEARCH_STORAGE_DIR = STORAGE_DIR / "searches"
ANALYSIS_STORAGE_DIR = STORAGE_DIR / "analyses"

_CAMERA_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_CROP_FILENAME_PATTERN = re.compile(r"^frame_\d{6,}\.jpg$", re.IGNORECASE)

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


def _stored_search_result(search_id: int, db: Session) -> dict:
    record = db.scalar(
        select(SearchResult).where(SearchResult.search_id == search_id)
    )
    if record is None:
        return {}
    try:
        payload = json.loads(record.result_json)
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _pipeline_result(payload: dict) -> dict:
    result = payload.get("result", payload)
    return result if isinstance(result, dict) else {}


def _add_legacy_summary(search_id: int, payload: dict) -> dict:
    """Add summary/timing fields to older persisted results without changing DB data."""
    result = _pipeline_result(payload)
    if isinstance(result.get("summary"), dict):
        return payload

    videos = result.get("videos", [])
    identities = result.get("global_identities", [])
    if not isinstance(videos, list) or not isinstance(identities, list):
        return payload

    for video in videos:
        if not isinstance(video, dict) or video.get("track_observations"):
            continue
        camera_id = video.get("camera_id")
        if not isinstance(camera_id, str) or not _CAMERA_ID_PATTERN.fullmatch(camera_id):
            continue
        tracks_csv = (
            ANALYSIS_STORAGE_DIR
            / f"search_{search_id}"
            / "outputs"
            / "tracking"
            / camera_id
            / "tracks.csv"
        )
        try:
            summaries = TrackSummaryBuilder(tracks_csv).build()
        except Exception:
            continue
        video["track_observations"] = [
            {
                "camera_id": summary.camera_id,
                "track_id": summary.track_id,
                "first_frame": summary.start_frame,
                "last_frame": summary.end_frame,
                "first_time_seconds": summary.start_time,
                "last_time_seconds": summary.end_time,
                "duration_seconds": summary.duration_seconds,
                "timestamp_basis": "video_relative_seconds_from_frame_index_and_fps",
            }
            for summary in summaries
        ]

    result["summary"] = build_matched_person_summary(videos, identities)
    return payload


def _image_is_within_crop_root(path: Path, crop_root: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(crop_root.resolve(strict=True))
        return path.is_file()
    except (OSError, ValueError):
        return False


def _is_valid_crop_image(path: Path) -> bool:
    try:
        from PIL import Image

        with Image.open(path) as image:
            if image.format != "JPEG":
                return False
            image.verify()
        return True
    except Exception:
        return False


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


def _first_accepted_identity(analysis_result):
    """Return only an identity accepted by the existing cross-camera gate."""
    return next(
        (
            identity
            for identity in analysis_result.global_identities
            if identity.decision == "accept"
        ),
        None,
    )


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
    """Create a search, persist uploaded videos, and run real AI analysis."""

    if not videos:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 videos are required.",
        )

    if len(videos) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Real multi-camera analysis requires at least 2 videos.",
        )

    if len(videos) > 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Real multi-camera analysis supports a maximum of 3 cameras.",
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

    database_started = time.perf_counter()
    db.add(search)
    db.commit()
    db.refresh(search)
    logger.info(
        "search_database_timing stage=create seconds=%.3f cameras=%d",
        time.perf_counter() - database_started,
        len(videos),
    )

    search_directory = SEARCH_STORAGE_DIR / str(search.id)

    try:
        camera_names: list[str] = []
        video_paths: list[Path] = []
        upload_started = time.perf_counter()

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

            video_paths.append(destination)

        logger.info(
            "search_upload_timing seconds=%.3f cameras=%d total_bytes=%d",
            time.perf_counter() - upload_started,
            len(video_paths),
            sum(path.stat().st_size for path in video_paths),
        )

        analysis_result = await run_in_threadpool(
            run_real_analysis,
            video_paths=video_paths,
            camera_ids=camera_names,
            analysis_id=search.id,
        )

        result = analysis_result.to_dict()
        accepted_identity = _first_accepted_identity(analysis_result)

        search.status = "found" if accepted_identity is not None else "not_found"

        search_result = SearchResult(
            search_id=search.id,
            identity_label=(
                f"Global Person {accepted_identity.global_person_id}"
                if accepted_identity is not None
                else None
            ),
            overall_confidence=None,
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

        database_started = time.perf_counter()
        db.commit()
        db.refresh(search)
        db.refresh(search_result)
        logger.info(
            "search_database_timing stage=save_result seconds=%.3f",
            time.perf_counter() - database_started,
        )

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
        if isinstance(result_payload, dict):
            result_payload = _add_legacy_summary(search.id, result_payload)

    return {
        "id": search.id,
        "case_name": search.case_name,
        "created_at": search.created_at,
        "camera_count": search.camera_count,
        "status": search.status,
        "result": result_payload,
    }


@router.get("/{search_id}/evidence")
def list_search_evidence(
    search_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List a small set of real, owned crop images for a saved search."""
    _get_owned_search(search_id, user.id, db)
    result = _pipeline_result(_stored_search_result(search_id, db))
    videos = result.get("videos", [])
    identities = result.get("global_identities", [])
    if not isinstance(videos, list) or not isinstance(identities, list):
        return {"images": []}

    accepted_global_ids = {
        identity.get("global_person_id")
        for identity in identities
        if isinstance(identity, dict) and identity.get("decision") == "accept"
    }
    crops_root = ANALYSIS_STORAGE_DIR / f"search_{search_id}" / "outputs" / "crops"
    images = []

    for video in videos:
        if not isinstance(video, dict):
            continue
        camera_id = video.get("camera_id")
        if not isinstance(camera_id, str) or not _CAMERA_ID_PATTERN.fullmatch(camera_id):
            continue
        camera_root = crops_root / camera_id
        if not camera_root.is_dir():
            continue

        camera_identities = [
            identity for identity in identities
            if isinstance(identity, dict) and identity.get("camera_id") == camera_id
        ]
        if accepted_global_ids:
            camera_identities = [
                identity for identity in camera_identities
                if identity.get("global_person_id") in accepted_global_ids
            ]

        track_ids = []
        for identity in camera_identities:
            track_id = identity.get("track_id")
            if isinstance(track_id, int) and track_id not in track_ids:
                track_ids.append(track_id)

        if not track_ids:
            track_ids = sorted(
                int(match.group(1))
                for directory in camera_root.iterdir()
                if directory.is_dir()
                and (match := re.fullmatch(r"track_(\d+)", directory.name))
            )

        for track_id in track_ids[:4]:
            track_directory = camera_root / f"track_{track_id:03d}"
            if not track_directory.is_dir():
                continue
            crop_files = sorted(
                path for path in track_directory.iterdir()
                if path.is_file()
                and _CROP_FILENAME_PATTERN.fullmatch(path.name)
                and _image_is_within_crop_root(path, crops_root)
                and _is_valid_crop_image(path)
            )
            for crop in crop_files[:2]:
                images.append({
                    "camera_id": camera_id,
                    "track_id": track_id,
                    "filename": crop.name,
                    "image_url": (
                        f"/api/searches/{search_id}/evidence/"
                        f"{camera_id}/{track_id}/{crop.name}"
                    ),
                })

    return {"images": images}


@router.get("/{search_id}/evidence/{camera_id}/{track_id}/{filename}")
def get_search_evidence_image(
    search_id: int,
    camera_id: str,
    track_id: int,
    filename: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Serve only a generated crop belonging to an owned search."""
    _get_owned_search(search_id, user.id, db)
    if (
        not _CAMERA_ID_PATTERN.fullmatch(camera_id)
        or not _CROP_FILENAME_PATTERN.fullmatch(filename)
        or track_id < 0
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence image not found.")

    result = _pipeline_result(_stored_search_result(search_id, db))
    videos = result.get("videos", [])
    identities = result.get("global_identities", [])
    known_cameras = {
        video.get("camera_id") for video in videos
        if isinstance(video, dict)
    } if isinstance(videos, list) else set()
    known_track = any(
        isinstance(identity, dict)
        and identity.get("camera_id") == camera_id
        and identity.get("track_id") == track_id
        for identity in identities
    ) if isinstance(identities, list) else False
    if camera_id not in known_cameras or not known_track:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence image not found.")

    crop_root = ANALYSIS_STORAGE_DIR / f"search_{search_id}" / "outputs" / "crops"
    image_path = crop_root / camera_id / f"track_{track_id:03d}" / filename
    if not _image_is_within_crop_root(image_path, crop_root):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence image not found.")
    if not _is_valid_crop_image(image_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence image not found.")

    return FileResponse(image_path, media_type="image/jpeg")


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
