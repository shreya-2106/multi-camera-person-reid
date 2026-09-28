"""Identity decision logic for Phase 10 global person assignment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.temporal.scoring import CandidateScore


class IdentityDecisionError(RuntimeError):
    """Raised when an identity decision cannot be calculated."""


class IdentityDecision(str, Enum):
    """Decision states for a scored cross-camera candidate."""

    ACCEPT = "accept"
    DEFER = "defer"
    REJECT = "reject"


@dataclass(frozen=True)
class IdentityDecisionResult:
    """Result of evaluating one candidate score."""

    decision: IdentityDecision
    score: CandidateScore
    reason: str

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("reason must be non-empty")

    def to_dict(self) -> dict:
        """Convert the decision result to a serializable dictionary."""
        return {
            "decision": self.decision.value,
            "score": self.score.to_dict(),
            "reason": self.reason,
        }


class IdentityDecisionMaker:
    """Convert candidate scores into explicit identity decisions.

    The thresholds are engineering parameters for the initial Phase 10
    implementation. They are not scientifically calibrated Re-ID thresholds.
    """

    def __init__(
        self,
        accept_threshold: float = 0.85,
        defer_threshold: float = 0.65,
    ) -> None:
        if not 0.0 <= defer_threshold <= 1.0:
            raise IdentityDecisionError(
                "defer_threshold must be between 0 and 1"
            )

        if not 0.0 <= accept_threshold <= 1.0:
            raise IdentityDecisionError(
                "accept_threshold must be between 0 and 1"
            )

        if defer_threshold >= accept_threshold:
            raise IdentityDecisionError(
                "defer_threshold must be lower than accept_threshold"
            )

        self.accept_threshold = accept_threshold
        self.defer_threshold = defer_threshold

    def decide(
        self,
        score: CandidateScore,
    ) -> IdentityDecisionResult:
        """Evaluate a candidate score without assigning a global identity."""

        if score is None:
            raise IdentityDecisionError(
                "Candidate score cannot be None"
            )

        if score.total_score >= self.accept_threshold:
            return IdentityDecisionResult(
                decision=IdentityDecision.ACCEPT,
                score=score,
                reason="score_meets_accept_threshold",
            )

        if score.total_score >= self.defer_threshold:
            return IdentityDecisionResult(
                decision=IdentityDecision.DEFER,
                score=score,
                reason="score_is_ambiguous",
            )

        return IdentityDecisionResult(
            decision=IdentityDecision.REJECT,
            score=score,
            reason="score_below_defer_threshold",
        )