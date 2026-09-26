"""Phase 9 temporal and camera-topology analysis."""

from .evidence import (
    EvidenceError,
    TransitionEvidence,
    TransitionEvidenceBuilder,
)
from .topology import (
    CameraTopology,
    CameraTransition,
    TopologyError,
)
from .transition import (
    TransitionCandidate,
    TransitionCandidateBuilder,
    TransitionError,
)
from .types import TrackSummary
from .track_summary import (
    TrackSummaryBuilder,
    TrackSummaryError,
)

__all__ = [
    "CameraTopology",
    "CameraTransition",
    "TopologyError",
    "TransitionCandidate",
    "TransitionCandidateBuilder",
    "TransitionError",
    "TrackSummary",
    "TrackSummaryBuilder",
    "TrackSummaryError",
    "EvidenceError",
    "TransitionEvidence",
    "TransitionEvidenceBuilder",
]
from .scoring import (
    CandidateScore,
    CandidateScorer,
    ScoringError,
)