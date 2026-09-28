"""Phase 10 global identity assignment pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.matching.identity_assigner import (
    GlobalIdentityAssigner,
    IdentityCandidate,
)
from src.matching.identity_decision import (
    IdentityDecision,
    IdentityDecisionMaker,
    IdentityDecisionResult,
)
from src.matching.types import IdentityAssignment, TrackReference
from src.temporal.scoring import CandidateScore


class IdentityPipelineError(RuntimeError):
    """Raised when the identity pipeline cannot complete."""


@dataclass(frozen=True)
class IdentityPipelineResult:
    """Result of processing one identity candidate."""

    candidate: IdentityCandidate
    decision: IdentityDecisionResult
    assignment: IdentityAssignment | None

    def __post_init__(self) -> None:
        if self.candidate is None:
            raise ValueError("candidate must not be None")

        if self.decision is None:
            raise ValueError("decision must not be None")

    @property
    def accepted(self) -> bool:
        """Return whether the candidate was accepted."""
        return self.decision.decision == IdentityDecision.ACCEPT

    @property
    def deferred(self) -> bool:
        """Return whether the candidate was deferred."""
        return self.decision.decision == IdentityDecision.DEFER

    @property
    def rejected(self) -> bool:
        """Return whether the candidate was rejected."""
        return self.decision.decision == IdentityDecision.REJECT

    def to_dict(self) -> dict:
        """Convert the result to a serializable dictionary."""
        return {
            "candidate": {
                "camera_id": self.candidate.track.camera_id,
                "track_id": self.candidate.track.track_id,
                "target_global_person_id": (
                    self.candidate.target_global_person_id
                ),
                "evidence": self.candidate.evidence.to_dict(),
            },
            "decision": self.decision.to_dict(),
            "assignment": (
                self.assignment.to_dict()
                if self.assignment is not None
                else None
            ),
        }


class GlobalIdentityPipeline:
    """Process candidate scores into global identity assignments."""

    def __init__(
        self,
        assigner: GlobalIdentityAssigner | None = None,
        decision_maker: IdentityDecisionMaker | None = None,
    ) -> None:
        self.assigner = (
            assigner
            if assigner is not None
            else GlobalIdentityAssigner()
        )

        self.decision_maker = (
            decision_maker
            if decision_maker is not None
            else IdentityDecisionMaker()
        )

    def process_candidate(
        self,
        candidate: IdentityCandidate,
        score: CandidateScore,
    ) -> IdentityPipelineResult:
        """Process one candidate through decision and assignment."""
        if candidate is None:
            raise IdentityPipelineError(
                "Identity candidate cannot be None"
            )

        if score is None:
            raise IdentityPipelineError(
                "Candidate score cannot be None"
            )

        try:
            decision = self.decision_maker.decide(score)

            assignment = self.assigner.assign_decision(
                candidate,
                decision,
            )

        except Exception as exc:
            raise IdentityPipelineError(
                f"Identity pipeline failed: {exc}"
            ) from exc

        return IdentityPipelineResult(
            candidate=candidate,
            decision=decision,
            assignment=assignment,
        )

    def process_candidates(
        self,
        candidates: Iterable[
            tuple[IdentityCandidate, CandidateScore]
        ],
    ) -> tuple[IdentityPipelineResult, ...]:
        """Process multiple candidate-score pairs in sequence."""
        if candidates is None:
            raise IdentityPipelineError(
                "Candidates cannot be None"
            )

        results: list[IdentityPipelineResult] = []

        for candidate, score in candidates:
            results.append(
                self.process_candidate(
                    candidate,
                    score,
                )
            )

        return tuple(results)

    def get_assignment(
        self,
        camera_id: str,
        track_id: int,
    ) -> IdentityAssignment | None:
        """Return the global assignment for a camera track."""
        track = TrackReference(
            camera_id=camera_id,
            track_id=track_id,
        )

        return self.assigner.get_assignment(track)

    def get_identity_tracks(
        self,
        global_person_id: int,
    ) -> tuple[TrackReference, ...]:
        """Return all local tracks belonging to one identity."""
        return self.assigner.get_identity_tracks(
            global_person_id
        )

    def reset(self) -> None:
        """Clear all global identity assignments."""
        self.assigner.reset()