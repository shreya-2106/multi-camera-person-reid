"""Typed records for Phase 9 temporal track analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrackSummary:
    """Summary of one local track observed by one camera."""

    camera_id: str
    track_id: int

    start_frame: int
    end_frame: int

    start_time: float
    end_time: float

    start_center_x: float
    start_center_y: float

    end_center_x: float
    end_center_y: float

    frames_seen: int

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id must be non-empty")

        if not isinstance(self.track_id, int) or self.track_id < 0:
            raise ValueError(
                "track_id must be a non-negative integer"
            )

        if self.start_frame < 0:
            raise ValueError(
                "start_frame must be non-negative"
            )

        if self.end_frame < self.start_frame:
            raise ValueError(
                "end_frame must be greater than or equal to start_frame"
            )

        if self.start_time < 0.0:
            raise ValueError(
                "start_time must be non-negative"
            )

        if self.end_time < self.start_time:
            raise ValueError(
                "end_time must be greater than or equal to start_time"
            )

        if self.frames_seen <= 0:
            raise ValueError(
                "frames_seen must be greater than zero"
            )

    @property
    def duration_seconds(self) -> float:
        """Return the observed duration of the track in seconds."""
        return self.end_time - self.start_time

    @property
    def displacement_pixels(self) -> float:
        """Return the straight-line displacement of the track center."""
        dx = self.end_center_x - self.start_center_x
        dy = self.end_center_y - self.start_center_y

        return (dx ** 2 + dy ** 2) ** 0.5

    def to_dict(self) -> dict:
        """Convert the summary to a serializable dictionary."""
        data = asdict(self)
        data["duration_seconds"] = self.duration_seconds
        data["displacement_pixels"] = self.displacement_pixels
        return data