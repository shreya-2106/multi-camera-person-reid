"""Tests for Phase 10 global identity assignment."""

from __future__ import annotations

import pytest

from src.matching.identity_assigner import (
    GlobalIdentityAssigner,
    IdentityAssignmentError,
    IdentityCandidate,
)
from src.matching.types import IdentityEvidence, TrackReference


def make_evidence(
    mean_similarity: float = 0.8,
) -> IdentityEvidence:
    """Create valid test evidence."""
    return IdentityEvidence(
        mean_similarity=mean_similarity,
        max_similarity=0.9,
        comparison_count=10,
        travel_time_seconds=5.0,
        topology_valid=True,
        temporal_valid=True,
    )


def test_new_track_gets_global_identity() -> None:
    assigner = GlobalIdentityAssigner()

    track = TrackReference(
        camera_id="CAM01",
        track_id=1,
    )

    assignment = assigner.assign_new_track(track)

    assert assignment.global_person_id == 1
    assert assignment.track == track
    assert assignment.assignment_reason == "new_identity"


def test_second_unseen_track_gets_next_global_identity() -> None:
    assigner = GlobalIdentityAssigner()

    track_1 = TrackReference("CAM01", 1)
    track_2 = TrackReference("CAM02", 7)

    first = assigner.assign_new_track(track_1)
    second = assigner.assign_new_track(track_2)

    assert first.global_person_id == 1
    assert second.global_person_id == 2


def test_same_track_is_not_assigned_twice() -> None:
    assigner = GlobalIdentityAssigner()

    track = TrackReference("CAM01", 1)

    first = assigner.assign_new_track(track)
    second = assigner.assign_new_track(track)

    assert first == second
    assert len(assigner.assignments) == 1


def test_candidate_can_join_existing_identity() -> None:
    assigner = GlobalIdentityAssigner()

    first_track = TrackReference("CAM01", 1)
    second_track = TrackReference("CAM02", 7)

    first_assignment = assigner.assign_new_track(first_track)

    candidate = IdentityCandidate(
        track=second_track,
        evidence=make_evidence(),
        target_global_person_id=first_assignment.global_person_id,
    )

    second_assignment = assigner.assign_candidate(candidate)

    assert second_assignment.global_person_id == 1
    assert second_assignment.assignment_reason == (
        "matched_existing_identity"
    )


def test_candidate_requires_existing_identity() -> None:
    assigner = GlobalIdentityAssigner()

    track = TrackReference("CAM02", 7)

    candidate = IdentityCandidate(
        track=track,
        evidence=make_evidence(),
        target_global_person_id=99,
    )

    with pytest.raises(IdentityAssignmentError):
        assigner.assign_candidate(candidate)


def test_track_cannot_switch_global_identity() -> None:
    assigner = GlobalIdentityAssigner()

    track_1 = TrackReference("CAM01", 1)
    track_2 = TrackReference("CAM02", 7)
    track_3 = TrackReference("CAM03", 4)

    identity_1 = assigner.assign_new_track(track_1)
    identity_2 = assigner.assign_new_track(track_3)

    candidate = IdentityCandidate(
        track=track_2,
        evidence=make_evidence(),
        target_global_person_id=identity_1.global_person_id,
    )

    assignment = assigner.assign_candidate(candidate)

    assert assignment.global_person_id == identity_1.global_person_id

    conflicting_candidate = IdentityCandidate(
        track=track_2,
        evidence=make_evidence(),
        target_global_person_id=identity_2.global_person_id,
    )

    with pytest.raises(IdentityAssignmentError):
        assigner.assign_candidate(conflicting_candidate)


def test_get_identity_tracks() -> None:
    assigner = GlobalIdentityAssigner()

    track_1 = TrackReference("CAM01", 1)
    track_2 = TrackReference("CAM02", 7)

    identity = assigner.assign_new_track(track_1)

    candidate = IdentityCandidate(
        track=track_2,
        evidence=make_evidence(),
        target_global_person_id=identity.global_person_id,
    )

    assigner.assign_candidate(candidate)

    tracks = assigner.get_identity_tracks(1)

    assert tracks == (track_1, track_2)


def test_reset_clears_assignments() -> None:
    assigner = GlobalIdentityAssigner()

    track = TrackReference("CAM01", 1)

    assigner.assign_new_track(track)
    assigner.reset()

    assert assigner.assignments == ()

    new_assignment = assigner.assign_new_track(
        TrackReference("CAM03", 4)
    )

    assert new_assignment.global_person_id == 1
    