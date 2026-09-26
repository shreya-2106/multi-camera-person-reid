"""Combined appearance and temporal evidence for Phase 9."""

from __future__ import annotations

from dataclasses import dataclass

from src.reid.track_matcher import TrackSimilarityResult

from .transition import TransitionCandidate


class EvidenceError(RuntimeError):
    """Raised when cross-camera evidence cannot be calculated."""


@dataclass(frozen=True)
class TransitionEvidence:
    """Evidence describing how plausible a cross-camera match is."""

    from_camera: str
    from_track_id: int

    to_camera: str
    to_track_id: int

    mean_similarity: float
    max_similarity: float
    comparison_count: int

    travel_time_seconds: float

    topology_valid: bool
    temporal_valid: bool

    def __post_init__(self) -> None:
        if not self.from_camera:
            raise ValueError(
                "from_camera must be non-empty"
            )

        if not self.to_camera:
            raise ValueError(
                "to_camera must be non-empty"
            )

        if self.from_camera == self.to_camera:
            raise ValueError(
                "Evidence must be cross-camera"
            )

        if self.from_track_id < 0:
            raise ValueError(
                "from_track_id must be non-negative"
            )

        if self.to_track_id < 0:
            raise ValueError(
                "to_track_id must be non-negative"
            )

        if not 0.0 <= self.mean_similarity <= 1.0:
            raise ValueError(
                "mean_similarity must be between 0 and 1"
            )

        if not 0.0 <= self.max_similarity <= 1.0:
            raise ValueError(
                "max_similarity must be between 0 and 1"
            )

        if self.comparison_count <= 0:
            raise ValueError(
                "comparison_count must be greater than zero"
            )

        if self.travel_time_seconds < 0.0:
            raise ValueError(
                "travel_time_seconds must be non-negative"
            )

    def to_dict(self) -> dict:
        """Return the evidence as a serializable dictionary."""
        return {
            "from_camera": self.from_camera,
            "from_track_id": self.from_track_id,
            "to_camera": self.to_camera,
            "to_track_id": self.to_track_id,
            "mean_similarity": self.mean_similarity,
            "max_similarity": self.max_similarity,
            "comparison_count": self.comparison_count,
            "travel_time_seconds": self.travel_time_seconds,
            "topology_valid": self.topology_valid,
            "temporal_valid": self.temporal_valid,
        }


class TransitionEvidenceBuilder:
    """Combine appearance similarity with transition evidence."""

    def build(
        self,
        candidate: TransitionCandidate,
        similarity: TrackSimilarityResult,
    ) -> TransitionEvidence:
        """Build evidence for one valid transition candidate."""

        if candidate is None:
            raise EvidenceError(
                "Transition candidate cannot be None"
            )

        if similarity is None:
            raise EvidenceError(
                "Track similarity cannot be None"
            )

        return TransitionEvidence(
            from_camera=candidate.from_camera,
            from_track_id=candidate.from_track_id,
            to_camera=candidate.to_camera,
            to_track_id=candidate.to_track_id,
            mean_similarity=similarity.mean_similarity,
            max_similarity=similarity.max_similarity,
            comparison_count=similarity.comparison_count,
            travel_time_seconds=candidate.travel_time_seconds,
            topology_valid=True,
            temporal_valid=True,
        )