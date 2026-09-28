"""Data types for the multi-video analysis pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class VideoInput:
    """One video supplied to the analysis pipeline."""

    video_path: str
    camera_id: str

    def __post_init__(self) -> None:
        if not self.video_path:
            raise ValueError("video_path must be non-empty")

        if not self.camera_id:
            raise ValueError("camera_id must be non-empty")

    def to_dict(self) -> dict:
        """Convert the video input to a dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class VideoAnalysisResult:
    """Result of processing one uploaded video."""

    camera_id: str
    video_path: str
    status: str

    frames_processed: int = 0
    tracks_detected: int = 0
    crops_generated: int = 0
    embeddings_generated: int = 0

    output_directory: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id must be non-empty")

        if not self.video_path:
            raise ValueError("video_path must be non-empty")

        if not self.status:
            raise ValueError("status must be non-empty")

        for field_name in (
            "frames_processed",
            "tracks_detected",
            "crops_generated",
            "embeddings_generated",
        ):
            value = getattr(self, field_name)

            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"{field_name} must be a non-negative integer"
                )

    def to_dict(self) -> dict:
        """Convert the analysis result to a dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class MultiVideoAnalysisResult:
    """Overall result for a multi-video analysis request."""

    status: str
    videos: tuple[VideoAnalysisResult, ...]

    unique_global_people: int = 0

    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.status:
            raise ValueError("status must be non-empty")

        if self.unique_global_people < 0:
            raise ValueError(
                "unique_global_people must be non-negative"
            )

    def to_dict(self) -> dict:
        """Convert the complete result to a dictionary."""
        return {
            "status": self.status,
            "videos": [
                video.to_dict()
                for video in self.videos
            ],
            "unique_global_people": self.unique_global_people,
            "error": self.error,
        }