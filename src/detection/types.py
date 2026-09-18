"""
Structured representations for YOLO person-detection results.

Coordinate convention: all bounding boxes use (x1, y1, x2, y2) in pixel
coordinates of the *original* video frame, where:
    (x1, y1) = top-left corner
    (x2, y2) = bottom-right corner
Origin (0, 0) is the top-left of the frame, consistent with OpenCV.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

CSV_FIELDNAMES = [
    "camera_id",
    "frame_number",
    "timestamp_seconds",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
]


@dataclass
class Detection:
    """A single person detection in a single video frame."""

    camera_id: Optional[str]
    frame_number: int
    timestamp_seconds: Optional[float]
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(
                f"Invalid box: (x1={self.x1}, y1={self.y1}) must be top-left of "
                f"(x2={self.x2}, y2={self.y2})"
            )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DetectionVideoResult:
    """Outcome of running detection over an entire video."""

    video_path: str
    camera_id: Optional[str]
    status: str  # "OK" | "NOT_FOUND" | "UNREADABLE" | "ERROR"
    frames_processed: int = 0
    detection_count: int = 0
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    detections: List[Detection] = field(default_factory=list)
    output_csv_path: Optional[str] = None
    annotated_video_path: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def write_detections_csv(detections: List[Detection], output_path: Path) -> Path:
    """Write detections to a CSV file, creating parent directories as needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for det in detections:
            writer.writerow(det.to_dict())
    return output_path
