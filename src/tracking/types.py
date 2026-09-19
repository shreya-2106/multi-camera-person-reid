"""Structured, CSV-friendly results from local person tracking."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

TRACK_CSV_FIELDNAMES = [
    "camera_id", "frame_number", "timestamp_seconds", "track_id", "x1", "y1",
    "x2", "y2", "confidence", "class_id", "class_name",
]


@dataclass
class TrackedPerson:
    """One ByteTrack-confirmed person location for a video frame."""

    camera_id: Optional[str]
    frame_number: int
    timestamp_seconds: Optional[float]
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.track_id, int):
            raise ValueError("track_id must be an integer")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("bounding box coordinates must be ordered")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrackingVideoResult:
    video_path: str
    camera_id: Optional[str]
    status: str
    frames_processed: int = 0
    track_count: int = 0
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    tracks: list[TrackedPerson] = field(default_factory=list)
    output_csv_path: Optional[str] = None
    annotated_video_path: Optional[str] = None
    error: Optional[str] = None


def write_tracks_csv(tracks: list[TrackedPerson], output_path: Path) -> Path:
    """Write tracks with a stable header, including for an empty result."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACK_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(track.to_dict() for track in tracks)
    return output_path
