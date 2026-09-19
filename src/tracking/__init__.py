"""Phase 4 local ByteTrack integration (no cross-camera identity matching)."""

from src.tracking.types import TrackedPerson, TrackingVideoResult

__all__ = ["Tracker", "TrackingError", "TrackedPerson", "TrackingVideoResult"]


def __getattr__(name):
    """Load the CLI module lazily so ``python -m src.tracking.tracker`` is clean."""
    if name in {"Tracker", "TrackingError"}:
        from src.tracking.tracker import Tracker, TrackingError
        return {"Tracker": Tracker, "TrackingError": TrackingError}[name]
    raise AttributeError(name)
