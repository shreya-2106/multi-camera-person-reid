"""Global identity assignment for Phase 10."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .identity_decision import (
    IdentityDecision,
    IdentityDecisionResult,
)
from .types import IdentityAssignment, IdentityEvidence, TrackReference


class IdentityAssignmentError(RuntimeError):
    """Raised when global identity assignment cannot be completed."""


@dataclass(frozen=True)
class IdentityCandidate:
    """A candidate relationship between a local track and an identity."""

    track: TrackReference
    evidence: IdentityEvidence
    target_global_person_id: int

    def __post_init__(self) -> None:
        if self.target_global_person_id <= 0:
            raise ValueError(
                "target_global_person_id must be positive"
            )


class GlobalIdentityAssigner:
    """Maintain consistent global person identities."""

    def __init__(self) -> None:
        self._next_global_person_id = 1
        self._assignments: dict[TrackReference, IdentityAssignment] = {}

    @property
    def assignments(self) -> tuple[IdentityAssignment, ...]:
        """Return all current assignments."""
        return tuple(self._assignments.values())

    def _create_identity(
        self,
        track: TrackReference,
        evidence: IdentityEvidence | None = None,
        reason: str = "new_identity",
    ) -> IdentityAssignment:
        """Create a new global identity for an unseen track."""
        global_person_id = self._next_global_person_id

        assignment = IdentityAssignment(
            global_person_id=global_person_id,
            track=track,
            evidence=evidence,
            assignment_reason=reason,
        )

        self._assignments[track] = assignment
        self._next_global_person_id += 1

        return assignment

    def assign_new_track(
        self,
        track: TrackReference,
    ) -> IdentityAssignment:
        """
        Assign an unseen local track to a new global identity.

        An already assigned track is returned unchanged.
        """
        existing = self._assignments.get(track)

        if existing is not None:
            return existing

        return self._create_identity(track)

    def assign_candidate(
        self,
        candidate: IdentityCandidate,
    ) -> IdentityAssignment:
        """
        Assign a candidate track to an existing global identity.

        The target identity must already exist.

        A local track cannot be assigned to a different identity
        after it has already been assigned.
        """
        existing = self._assignments.get(candidate.track)

        if existing is not None:
            if (
                existing.global_person_id
                != candidate.target_global_person_id
            ):
                raise IdentityAssignmentError(
                    "Track is already assigned to a different "
                    "global identity: "
                    f"{candidate.track.camera_id}/"
                    f"{candidate.track.track_id}"
                )

            return existing

        target_exists = any(
            assignment.global_person_id
            == candidate.target_global_person_id
            for assignment in self._assignments.values()
        )

        if not target_exists:
            raise IdentityAssignmentError(
                "Target global identity does not exist: "
                f"{candidate.target_global_person_id}"
            )

        assignment = IdentityAssignment(
            global_person_id=candidate.target_global_person_id,
            track=candidate.track,
            evidence=candidate.evidence,
            assignment_reason="matched_existing_identity",
        )

        self._assignments[candidate.track] = assignment

        return assignment

    def assign_decision(
        self,
        candidate: IdentityCandidate,
        decision: IdentityDecisionResult,
    ) -> IdentityAssignment | None:
        """
        Apply a candidate only when the identity decision is ACCEPT.

        DEFER and REJECT decisions leave the candidate unassigned.
        """
        if decision is None:
            raise IdentityAssignmentError(
                "Identity decision cannot be None"
            )

        if decision.decision == IdentityDecision.ACCEPT:
            return self.assign_candidate(candidate)

        if decision.decision in (
            IdentityDecision.DEFER,
            IdentityDecision.REJECT,
        ):
            return None

        raise IdentityAssignmentError(
            f"Unsupported identity decision: {decision.decision}"
        )

    def get_assignment(
        self,
        track: TrackReference,
    ) -> IdentityAssignment | None:
        """Return the assignment for a track, if one exists."""
        return self._assignments.get(track)

    def get_identity_tracks(
        self,
        global_person_id: int,
    ) -> tuple[TrackReference, ...]:
        """Return all local tracks assigned to one global identity."""
        if global_person_id <= 0:
            raise ValueError(
                "global_person_id must be positive"
            )

        return tuple(
            assignment.track
            for assignment in self._assignments.values()
            if assignment.global_person_id == global_person_id
        )

    def assign_candidates(
        self,
        candidates: Iterable[IdentityCandidate],
    ) -> tuple[IdentityAssignment, ...]:
        """
        Apply multiple candidate assignments in sequence.
        """
        results: list[IdentityAssignment] = []

        for candidate in candidates:
            results.append(
                self.assign_candidate(candidate)
            )

        return tuple(results)

    def reset(self) -> None:
        """Clear all assignments and restart global ID numbering."""
        self._assignments.clear()
        self._next_global_person_id = 1

    