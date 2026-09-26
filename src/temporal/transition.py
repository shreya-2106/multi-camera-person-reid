"""Cross-camera transition candidate generation for Phase 9."""

from __future__ import annotations

from dataclasses import dataclass

from .topology import CameraTopology
from .types import TrackSummary


class TransitionError(RuntimeError):
    """Raised when transition analysis cannot be performed."""


@dataclass(frozen=True)
class TransitionCandidate:
    """A temporally and topologically valid cross-camera candidate."""

    from_camera: str
    from_track_id: int
    to_camera: str
    to_track_id: int

    exit_time: float
    entry_time: float
    travel_time_seconds: float

    def __post_init__(self) -> None:
        if not self.from_camera:
            raise ValueError("from_camera must be non-empty")

        if not self.to_camera:
            raise ValueError("to_camera must be non-empty")

        if self.from_camera == self.to_camera:
            raise ValueError(
                "Transition must be cross-camera"
            )

        if self.from_track_id < 0:
            raise ValueError(
                "from_track_id must be non-negative"
            )

        if self.to_track_id < 0:
            raise ValueError(
                "to_track_id must be non-negative"
            )

        if self.exit_time < 0.0:
            raise ValueError(
                "exit_time must be non-negative"
            )

        if self.entry_time < 0.0:
            raise ValueError(
                "entry_time must be non-negative"
            )

        if self.travel_time_seconds < 0.0:
            raise ValueError(
                "travel_time_seconds must be non-negative"
            )

    def to_dict(self) -> dict:
        """Return the candidate as a serializable dictionary."""
        return {
            "from_camera": self.from_camera,
            "from_track_id": self.from_track_id,
            "to_camera": self.to_camera,
            "to_track_id": self.to_track_id,
            "exit_time": self.exit_time,
            "entry_time": self.entry_time,
            "travel_time_seconds": self.travel_time_seconds,
        }


class TransitionCandidateBuilder:
    """Generate valid cross-camera transition candidates."""

    def __init__(
        self,
        topology: CameraTopology,
    ) -> None:
        self.topology = topology

    def build_pair(
        self,
        from_track: TrackSummary,
        to_track: TrackSummary,
    ) -> TransitionCandidate | None:
        """Check whether two tracks form a valid transition."""

        if from_track.camera_id == to_track.camera_id:
            return None

        if not self.topology.can_transition(
            from_track.camera_id,
            to_track.camera_id,
        ):
            return None

        travel_time = (
            to_track.start_time
            - from_track.end_time
        )

        if not self.topology.is_travel_time_valid(
            from_track.camera_id,
            to_track.camera_id,
            travel_time,
        ):
            return None

        return TransitionCandidate(
            from_camera=from_track.camera_id,
            from_track_id=from_track.track_id,
            to_camera=to_track.camera_id,
            to_track_id=to_track.track_id,
            exit_time=from_track.end_time,
            entry_time=to_track.start_time,
            travel_time_seconds=travel_time,
        )

    def build(
        self,
        tracks: list[TrackSummary],
    ) -> list[TransitionCandidate]:
        """Build all valid cross-camera transition candidates."""

        candidates: list[TransitionCandidate] = []

        for from_track in tracks:
            for to_track in tracks:
                if from_track is to_track:
                    continue

                candidate = self.build_pair(
                    from_track,
                    to_track,
                )

                if candidate is not None:
                    candidates.append(candidate)

        return candidates