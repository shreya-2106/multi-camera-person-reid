"""Global identity matching and assignment."""

from .identity_assigner import (
    GlobalIdentityAssigner,
    IdentityAssignmentError,
    IdentityCandidate,
)
from .identity_decision import (
    IdentityDecision,
    IdentityDecisionError,
    IdentityDecisionMaker,
    IdentityDecisionResult,
)
from .identity_pipeline import (
    GlobalIdentityPipeline,
    IdentityPipelineError,
    IdentityPipelineResult,
)
from .types import (
    IdentityAssignment,
    IdentityEvidence,
    TrackReference,
)

__all__ = [
    "GlobalIdentityAssigner",
    "IdentityAssignmentError",
    "IdentityCandidate",
    "IdentityDecision",
    "IdentityDecisionError",
    "IdentityDecisionMaker",
    "IdentityDecisionResult",
    "GlobalIdentityPipeline",
    "IdentityPipelineError",
    "IdentityPipelineResult",
    "IdentityAssignment",
    "IdentityEvidence",
    "TrackReference",
]