"""Cross-camera transition candidate generation for Phase 9."""

from __future__ import annotations

from dataclasses import dataclass
import math

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

        if not math.isfinite(self.exit_time) or not math.isfinite(self.entry_time):
            raise ValueError("transition times must be finite")

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
        from_offset_seconds: float = 0.0,
        to_offset_seconds: float = 0.0,
    ) -> TransitionCandidate | None:
        """Check a transition after adding each clip's timeline offset.

        The offset is added to video-relative seconds to place observations on
        a shared timeline. Callers must supply a known offset, not an estimate
        inferred from appearance similarity.
        """

        if not math.isfinite(from_offset_seconds) or not math.isfinite(to_offset_seconds):
            raise ValueError("timeline offsets must be finite")

        if from_track.camera_id == to_track.camera_id:
            return None

        if not self.topology.can_transition(
            from_track.camera_id,
            to_track.camera_id,
        ):
            return None

        travel_time = (
            (to_track.start_time + to_offset_seconds)
            - (from_track.end_time + from_offset_seconds)
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
            exit_time=from_track.end_time + from_offset_seconds,
            entry_time=to_track.start_time + to_offset_seconds,
            travel_time_seconds=travel_time,
        )

    def explain_pair(
        self,
        from_track: TrackSummary,
        to_track: TrackSummary,
        from_offset_seconds: float = 0.0,
        to_offset_seconds: float = 0.0,
    ) -> dict:
        """Explain the transition using offsets added to video-relative time."""

        if not math.isfinite(from_offset_seconds) or not math.isfinite(to_offset_seconds):
            raise ValueError("timeline offsets must be finite")

        if from_track.camera_id == to_track.camera_id:
            return {
                "status": "rejected",
                "reason": "same_camera",
                "travel_time_seconds": None,
            }

        if not self.topology.can_transition(
            from_track.camera_id,
            to_track.camera_id,
        ):
            return {
                "status": "rejected",
                "reason": "invalid_topology",
                "travel_time_seconds": None,
            }

        travel_time = (
            (to_track.start_time + to_offset_seconds)
            - (from_track.end_time + from_offset_seconds)
        )

        if not self.topology.is_travel_time_valid(
            from_track.camera_id,
            to_track.camera_id,
            travel_time,
        ):
            return {
                "status": "rejected",
                "reason": "invalid_travel_time",
                "travel_time_seconds": travel_time,
            }

        return {
            "status": "valid",
            "reason": None,
            "travel_time_seconds": travel_time,
        }

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
