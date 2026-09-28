"""Typed records for Phase 10 global identity assignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class TrackReference:
    """Reference to one local camera track."""

    camera_id: str
    track_id: int

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id must be non-empty")

        if not isinstance(self.track_id, int) or self.track_id < 0:
            raise ValueError(
                "track_id must be a non-negative integer"
            )

    def to_dict(self) -> dict:
        """Convert the track reference to a dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class IdentityEvidence:
    """Evidence supporting a global identity assignment."""

    mean_similarity: float
    max_similarity: float
    comparison_count: int

    travel_time_seconds: Optional[float] = None
    topology_valid: Optional[bool] = None
    temporal_valid: Optional[bool] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.mean_similarity <= 1.0:
            raise ValueError(
                "mean_similarity must be between 0.0 and 1.0"
            )

        if not 0.0 <= self.max_similarity <= 1.0:
            raise ValueError(
                "max_similarity must be between 0.0 and 1.0"
            )

        if self.comparison_count <= 0:
            raise ValueError(
                "comparison_count must be greater than zero"
            )

        if (
            self.travel_time_seconds is not None
            and self.travel_time_seconds < 0.0
        ):
            raise ValueError(
                "travel_time_seconds must be non-negative"
            )

    def to_dict(self) -> dict:
        """Convert the evidence to a dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class IdentityAssignment:
    """Assignment of a local track to a global person identity."""

    global_person_id: int
    track: TrackReference

    evidence: Optional[IdentityEvidence] = None

    assignment_reason: str = "new_identity"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.global_person_id, int)
            or self.global_person_id <= 0
        ):
            raise ValueError(
                "global_person_id must be a positive integer"
            )

        if not self.assignment_reason:
            raise ValueError(
                "assignment_reason must be non-empty"
            )

    def to_dict(self) -> dict:
        """Convert the identity assignment to a dictionary."""
        data = {
            "global_person_id": self.global_person_id,
            "track": self.track.to_dict(),
            "evidence": (
                self.evidence.to_dict()
                if self.evidence is not None
                else None
            ),
            "assignment_reason": self.assignment_reason,
        }

        return data 