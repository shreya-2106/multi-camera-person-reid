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
        """Convert the video analysis result to a dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class GlobalIdentityResult:
    """Serializable result for one global person identity."""

    global_person_id: int
    camera_id: str
    track_id: int
    decision: str
    score: Optional[dict]
    evidence: Optional[dict]
    assignment_reason: str

    def __post_init__(self) -> None:
        if self.global_person_id < 1:
            raise ValueError(
                "global_person_id must be a positive integer"
            )

        if not self.camera_id:
            raise ValueError("camera_id must be non-empty")

        if self.track_id < 0:
            raise ValueError(
                "track_id must be a non-negative integer"
            )

        if not self.decision:
            raise ValueError("decision must be non-empty")

        if self.score is not None and not isinstance(self.score, dict):
            raise ValueError("score must be a dictionary or None")
        if self.evidence is not None and not isinstance(self.evidence, dict):
            raise ValueError("evidence must be a dictionary or None")

        if not self.assignment_reason:
            raise ValueError(
                "assignment_reason must be non-empty"
            )

    def to_dict(self) -> dict:
        """Convert the identity result to a dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class MultiVideoAnalysisResult:
    """Overall result for a multi-video analysis request."""

    status: str
    videos: tuple[VideoAnalysisResult, ...]

    unique_global_people: int = 0
    global_identities: tuple[GlobalIdentityResult, ...] = ()
    candidate_reports: tuple[dict, ...] = ()

    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.status:
            raise ValueError("status must be non-empty")

        if self.unique_global_people < 0:
            raise ValueError(
                "unique_global_people must be non-negative"
            )

        if not isinstance(self.candidate_reports, tuple):
            raise ValueError(
                "candidate_reports must be a tuple"
            )

        for report in self.candidate_reports:
            if not isinstance(report, dict):
                raise ValueError(
                    "Each candidate report must be a dictionary"
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
            "global_identities": [
                identity.to_dict()
                for identity in self.global_identities
            ],
            "candidate_reports": list(
                self.candidate_reports
            ),
            "error": self.error,
        }