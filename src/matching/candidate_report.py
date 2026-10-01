"""Candidate reporting types for Phase 14."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.temporal.scoring import CandidateScore
from src.temporal.transition import TransitionCandidate
from src.temporal.evidence import TransitionEvidence


@dataclass(frozen=True)
class CandidateReport:
    """Report the outcome and evidence for one cross-camera track pair."""

    from_camera: str
    from_track_id: int

    to_camera: str
    to_track_id: int

    status: str
    reason: Optional[str] = None

    travel_time_seconds: Optional[float] = None

    transition: Optional[TransitionCandidate] = None
    evidence: Optional[TransitionEvidence] = None
    score: Optional[CandidateScore] = None

    def __post_init__(self) -> None:
        if not self.from_camera:
            raise ValueError("from_camera must be non-empty")

        if not self.to_camera:
            raise ValueError("to_camera must be non-empty")

        if self.from_camera == self.to_camera:
            raise ValueError("Candidate report must be cross-camera")

        if self.from_track_id < 0:
            raise ValueError("from_track_id must be non-negative")

        if self.to_track_id < 0:
            raise ValueError("to_track_id must be non-negative")

        if self.status not in {"valid", "rejected"}:
            raise ValueError(
                "status must be either 'valid' or 'rejected'"
            )

        if self.status == "rejected" and not self.reason:
            raise ValueError(
                "Rejected candidates must include a reason"
            )

        if self.travel_time_seconds is not None:
            if self.travel_time_seconds < 0.0:
                raise ValueError(
                    "travel_time_seconds must be non-negative"
                )

    def to_dict(self) -> dict:
        """Return the report as a serializable dictionary."""
        return {
            "from_camera": self.from_camera,
            "from_track_id": self.from_track_id,
            "to_camera": self.to_camera,
            "to_track_id": self.to_track_id,
            "status": self.status,
            "reason": self.reason,
            "travel_time_seconds": self.travel_time_seconds,
            "transition": (
                self.transition.to_dict()
                if self.transition is not None
                else None
            ),
            "evidence": (
                self.evidence.to_dict()
                if self.evidence is not None
                else None
            ),
            "score": (
                self.score.to_dict()
                if self.score is not None
                else None
            ),
        }