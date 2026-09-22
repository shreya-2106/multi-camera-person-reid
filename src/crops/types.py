"""Typed, CSV-friendly output for Phase 5 person crop extraction."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

CROP_CSV_FIELDNAMES = [
    "camera_id", "track_id", "frame_number", "source_video", "crop_path",
    "original_x1", "original_y1", "original_x2", "original_y2",
    "final_x1", "final_y1", "final_x2", "final_y2",
    "crop_width", "crop_height", "crop_area", "detection_confidence",
]


@dataclass(frozen=True)
class CropMetadata:
    """Metadata for one saved crop; track IDs remain local to the source video."""
    camera_id: str
    track_id: int
    frame_number: int
    source_video: str
    crop_path: str
    original_x1: float
    original_y1: float
    original_x2: float
    original_y2: float
    final_x1: int
    final_y1: int
    final_x2: int
    final_y2: int
    crop_width: int
    crop_height: int
    crop_area: int
    detection_confidence: float

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id must be non-empty")
        if not isinstance(self.track_id, int):
            raise ValueError("track_id must be an integer")
        if self.crop_width <= 0 or self.crop_height <= 0 or self.crop_area <= 0:
            raise ValueError("crop dimensions and area must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CropExtractionResult:
    """Summary of incremental extraction for one source video."""
    video_path: str
    camera_id: Optional[str]
    status: str
    frames_processed: int = 0
    tracks_seen: int = 0
    crops_saved: int = 0
    crops_rejected: int = 0
    rejection_counts: dict[str, int] = field(default_factory=dict)
    output_directory: Optional[str] = None
    metadata_csv_path: Optional[str] = None
    error: Optional[str] = None


def write_crops_csv(metadata: list[CropMetadata], output_path: Path) -> Path:
    """Write the metadata header even when no crop qualified."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CROP_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(item.to_dict() for item in metadata)
    return output_path
