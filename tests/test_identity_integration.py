"""Integration tests for Phase 10 identity decision and assignment."""

from __future__ import annotations

from src.matching.identity_assigner import (
    GlobalIdentityAssigner,
    IdentityCandidate,
)
from src.matching.identity_decision import (
    IdentityDecision,
    IdentityDecisionMaker,
)
from src.matching.types import IdentityEvidence, TrackReference
from src.temporal.scoring import CandidateScore


def make_score(total_score: float) -> CandidateScore:
    """Create a controlled score for integration testing."""
    return CandidateScore(
        appearance_score=total_score,
        temporal_score=1.0,
        topology_score=1.0,
        total_score=total_score,
    )


def make_evidence(mean_similarity: float = 0.8) -> IdentityEvidence:
    """Create controlled identity evidence."""
    return IdentityEvidence(
        mean_similarity=mean_similarity,
        max_similarity=0.9,
        comparison_count=10,
        travel_time_seconds=5.0,
        topology_valid=True,
        temporal_valid=True,
    )


def test_accepted_candidate_can_join_existing_identity() -> None:
    """An accepted candidate should be assignable to an existing identity."""
    decision_maker = IdentityDecisionMaker()
    assigner = GlobalIdentityAssigner()

    source_track = TrackReference("CAM01", 1)
    destination_track = TrackReference("CAM02", 7)

    source_assignment = assigner.assign_new_track(source_track)

    score = make_score(0.90)
    decision = decision_maker.decide(score)

    assert decision.decision == IdentityDecision.ACCEPT

    candidate = IdentityCandidate(
        track=destination_track,
        evidence=make_evidence(),
        target_global_person_id=source_assignment.global_person_id,
    )

    assignment = assigner.assign_candidate(candidate)

    assert assignment.global_person_id == 1
    assert assignment.track == destination_track


def test_deferred_candidate_is_not_assigned() -> None:
    """A deferred candidate should not automatically change identity."""
    decision_maker = IdentityDecisionMaker()
    assigner = GlobalIdentityAssigner()

    source_track = TrackReference("CAM01", 1)
    destination_track = TrackReference("CAM02", 7)

    source_assignment = assigner.assign_new_track(source_track)

    score = make_score(0.75)
    decision = decision_maker.decide(score)

    assert decision.decision == IdentityDecision.DEFER

    assert assigner.get_assignment(destination_track) is None
    assert source_assignment.global_person_id == 1


def test_rejected_candidate_is_not_assigned() -> None:
    """A rejected candidate should not be assigned to an existing identity."""
    decision_maker = IdentityDecisionMaker()
    assigner = GlobalIdentityAssigner()

    source_track = TrackReference("CAM01", 1)
    destination_track = TrackReference("CAM02", 7)

    source_assignment = assigner.assign_new_track(source_track)

    score = make_score(0.50)
    decision = decision_maker.decide(score)

    assert decision.decision == IdentityDecision.REJECT

    assert assigner.get_assignment(destination_track) is None
    assert source_assignment.global_person_id == 1


def test_multiple_camera_tracks_can_share_one_identity() -> None:
    """Accepted tracks from multiple cameras can belong to one global identity."""
    decision_maker = IdentityDecisionMaker()
    assigner = GlobalIdentityAssigner()

    cam01_track = TrackReference("CAM01", 1)
    cam02_track = TrackReference("CAM02", 7)
    cam03_track = TrackReference("CAM03", 4)

    identity = assigner.assign_new_track(cam01_track)

    for track in (cam02_track, cam03_track):
        score = make_score(0.90)
        decision = decision_maker.decide(score)

        assert decision.decision == IdentityDecision.ACCEPT

        candidate = IdentityCandidate(
            track=track,
            evidence=make_evidence(),
            target_global_person_id=identity.global_person_id,
        )

        assigner.assign_candidate(candidate)

    tracks = assigner.get_identity_tracks(
        identity.global_person_id
    )

    assert tracks == (
        cam01_track,
        cam02_track,
        cam03_track,
    )


def test_deferred_track_can_remain_unassigned() -> None:
    """An ambiguous track can remain unassigned until more evidence arrives."""
    decision_maker = IdentityDecisionMaker()
    assigner = GlobalIdentityAssigner()

    source_track = TrackReference("CAM01", 1)
    uncertain_track = TrackReference("CAM03", 4)

    assigner.assign_new_track(source_track)

    score = make_score(0.70)
    decision = decision_maker.decide(score)

    assert decision.decision == IdentityDecision.DEFER
    assert assigner.get_assignment(uncertain_track) is None


def test_assign_decision_accepts_candidate() -> None:
    """ACCEPT should assign the candidate to the target identity."""
    decision_maker = IdentityDecisionMaker()
    assigner = GlobalIdentityAssigner()

    source_track = TrackReference("CAM01", 1)
    destination_track = TrackReference("CAM02", 7)

    identity = assigner.assign_new_track(source_track)

    decision = decision_maker.decide(
        make_score(0.90)
    )

    candidate = IdentityCandidate(
        track=destination_track,
        evidence=make_evidence(),
        target_global_person_id=identity.global_person_id,
    )

    assignment = assigner.assign_decision(
        candidate,
        decision,
    )

    assert assignment is not None
    assert assignment.global_person_id == 1
    assert assignment.track == destination_track


def test_assign_decision_defers_candidate() -> None:
    """DEFER should leave the candidate unassigned."""
    decision_maker = IdentityDecisionMaker()
    assigner = GlobalIdentityAssigner()

    source_track = TrackReference("CAM01", 1)
    destination_track = TrackReference("CAM02", 7)

    assigner.assign_new_track(source_track)

    decision = decision_maker.decide(
        make_score(0.75)
    )

    candidate = IdentityCandidate(
        track=destination_track,
        evidence=make_evidence(),
        target_global_person_id=1,
    )

    assignment = assigner.assign_decision(
        candidate,
        decision,
    )

    assert assignment is None
    assert assigner.get_assignment(destination_track) is None


def test_assign_decision_rejects_candidate() -> None:
    """REJECT should leave the candidate unassigned."""
    decision_maker = IdentityDecisionMaker()
    assigner = GlobalIdentityAssigner()

    source_track = TrackReference("CAM01", 1)
    destination_track = TrackReference("CAM02", 7)

    assigner.assign_new_track(source_track)

    decision = decision_maker.decide(
        make_score(0.50)
    )

    candidate = IdentityCandidate(
        track=destination_track,
        evidence=make_evidence(),
        target_global_person_id=1,
    )

    assignment = assigner.assign_decision(
        candidate,
        decision,
    )

    assert assignment is None
    assert assigner.get_assignment(destination_track) is None


def test_multi_camera_identity_chain() -> None:
    """One person can retain the same global identity across cameras."""
    decision_maker = IdentityDecisionMaker()
    assigner = GlobalIdentityAssigner()

    cam01_track = TrackReference("CAM01", 1)
    cam02_track = TrackReference("CAM02", 7)
    cam03_track = TrackReference("CAM03", 4)

    # Person first appears in CAM01.
    identity = assigner.assign_new_track(cam01_track)

    assert identity.global_person_id == 1

    # Person transitions from CAM01 to CAM02.
    decision_cam02 = decision_maker.decide(
        make_score(0.90)
    )

    candidate_cam02 = IdentityCandidate(
        track=cam02_track,
        evidence=make_evidence(0.88),
        target_global_person_id=identity.global_person_id,
    )

    assignment_cam02 = assigner.assign_decision(
        candidate_cam02,
        decision_cam02,
    )

    assert assignment_cam02 is not None
    assert assignment_cam02.global_person_id == 1

    # Person later transitions from CAM02 to CAM03.
    decision_cam03 = decision_maker.decide(
        make_score(0.87)
    )

    candidate_cam03 = IdentityCandidate(
        track=cam03_track,
        evidence=make_evidence(0.84),
        target_global_person_id=identity.global_person_id,
    )

    assignment_cam03 = assigner.assign_decision(
        candidate_cam03,
        decision_cam03,
    )

    assert assignment_cam03 is not None
    assert assignment_cam03.global_person_id == 1

    # All three local tracks must belong to the same global identity.
    tracks = assigner.get_identity_tracks(1)

    assert tracks == (
        cam01_track,
        cam02_track,
        cam03_track,
    )
def test_track_cannot_switch_global_identity() -> None:
    """A local track must not switch from one global identity to another."""
    decision_maker = IdentityDecisionMaker()
    assigner = GlobalIdentityAssigner()

    cam01_track = TrackReference("CAM01", 1)
    cam02_track = TrackReference("CAM02", 7)
    cam03_track = TrackReference("CAM03", 4)

    # Create two different global identities.
    identity_one = assigner.assign_new_track(cam01_track)
    identity_two = assigner.assign_new_track(cam02_track)

    assert identity_one.global_person_id == 1
    assert identity_two.global_person_id == 2

    # First assign CAM03 Track 4 to Global Person 1.
    decision_one = decision_maker.decide(
        make_score(0.90)
    )

    candidate_one = IdentityCandidate(
        track=cam03_track,
        evidence=make_evidence(0.88),
        target_global_person_id=identity_one.global_person_id,
    )

    assignment = assigner.assign_decision(
        candidate_one,
        decision_one,
    )

    assert assignment is not None
    assert assignment.global_person_id == 1

    # Now attempt to switch the same CAM03 Track 4 to Global Person 2.
    decision_two = decision_maker.decide(
        make_score(0.92)
    )

    candidate_two = IdentityCandidate(
        track=cam03_track,
        evidence=make_evidence(0.91),
        target_global_person_id=identity_two.global_person_id,
    )

    import pytest

    with pytest.raises(Exception):
        assigner.assign_decision(
            candidate_two,
            decision_two,
        )

    # The original assignment must remain Global Person 1.
    final_assignment = assigner.get_assignment(cam03_track)

    assert final_assignment is not None
    assert final_assignment.global_person_id == 1
def test_scored_candidate_flows_into_identity_assignment() -> None:
    """A strongly supported candidate should flow from scoring to assignment."""
    decision_maker = IdentityDecisionMaker()
    assigner = GlobalIdentityAssigner()

    cam01_track = TrackReference("CAM01", 1)
    cam02_track = TrackReference("CAM02", 7)

    # CAM01 establishes the initial global identity.
    identity = assigner.assign_new_track(cam01_track)

    assert identity.global_person_id == 1

    # Build a strong candidate score from multiple evidence sources.
    score = CandidateScore(
        appearance_score=0.88,
        temporal_score=0.90,
        topology_score=1.00,
        total_score=0.90,
    )

    decision = decision_maker.decide(score)

    assert decision.decision == IdentityDecision.ACCEPT
    assert decision.score.total_score == 0.90

    # Attach the candidate evidence to CAM02.
    candidate = IdentityCandidate(
        track=cam02_track,
        evidence=make_evidence(0.88),
        target_global_person_id=identity.global_person_id,
    )

    assignment = assigner.assign_decision(
        candidate,
        decision,
    )

    assert assignment is not None
    assert assignment.global_person_id == 1
    assert assignment.track == cam02_track

    # Both camera tracks must now belong to the same global identity.
    tracks = assigner.get_identity_tracks(1)

    assert tracks == (
        cam01_track,
        cam02_track,
    )