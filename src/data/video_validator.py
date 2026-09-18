"""Lightweight metadata validation for locally supplied CCTV videos."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".m4v"})
CAMERA_PATTERN = re.compile(r"(?:^|[_-])(cam(?:era)?[_-]?\d+)(?:[_-]|$)", re.IGNORECASE)


def _camera_id(filename: str) -> str | None:
    match = CAMERA_PATTERN.search(Path(filename).stem)
    return match.group(1).replace("_", "").replace("-", "").upper() if match else None

def infer_camera_id(filename: str) -> str | None:
    """Infer a camera ID from a video filename."""
    return _camera_id(filename)


def validate_video(path: Path) -> dict[str, Any]:
    """Read metadata only; no full frame-by-frame processing is performed."""
    result: dict[str, Any] = {"filename": path.name, "camera_id": _camera_id(path.name), "file_size_bytes": path.stat().st_size}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        result.update({"status": "UNREADABLE", "width": None, "height": None, "fps": None, "frame_count": None, "duration_seconds": None, "codec": None})
        return result
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or None
        frame_value = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_count = int(frame_value) if frame_value and frame_value > 0 else None
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4)).rstrip("\x00") or None
        result.update({"status": "READABLE", "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or None,
                       "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None, "fps": fps,
                       "frame_count": frame_count, "duration_seconds": frame_count / fps if frame_count is not None and fps else None,
                       "codec": codec})
        return result
    finally:
        capture.release()


def validate_videos(videos_root: Path) -> dict[str, Any]:
    """Create a JSON-serializable report for available video files."""
    root = Path(videos_root)
    report = {"validation_timestamp": datetime.now(timezone.utc).isoformat(), "videos_root": str(root),
              "supported_video_extensions": sorted(VIDEO_EXTENSIONS)}
    if not root.is_dir():
        return report | {"status": "NOT_FOUND", "videos": [], "warnings": ["Video directory is missing."]}
    videos = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    return report | {"status": "FOUND", "videos": [validate_video(path) for path in videos],
                     "warnings": [] if videos else ["No supported video files were found."]}
