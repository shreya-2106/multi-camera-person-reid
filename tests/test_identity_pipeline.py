from __future__ import annotations

import pytest

from src.matching.identity_assigner import (
    GlobalIdentityAssigner,
    IdentityAssignmentError,
    IdentityCandidate,
)
from src.matching.identity_decision import (
    IdentityDecision,
    IdentityDecisionMaker,
)
from src.matching.identity_pipeline import (
    GlobalIdentityPipeline,
    IdentityPipelineError,
)
from src.matching.types import IdentityEvidence, TrackReference
from src.temporal.scoring import CandidateScore


def make_score(total_score: float) -> CandidateScore:
    return CandidateScore(
        appearance_score=total_score,
        temporal_score=1.0,
        topology_score=1.0,
        total_score=total_score,
    )


def make_evidence(mean_similarity: float = 0.8) -> IdentityEvidence:
    return IdentityEvidence(
        mean_similarity=mean_similarity,
        max_similarity=0.9,
        comparison_count=10,
        travel_time_seconds=5.0,
        topology_valid=True,
        temporal_valid=True,
    )


def make_candidate(
    camera_id: str = "CAM02",
    track_id: int = 7,
    global_person_id: int = 1,
) -> IdentityCandidate:
    return IdentityCandidate(
        track=TrackReference(
            camera_id=camera_id,
            track_id=track_id,
        ),
        evidence=make_evidence(),
        target_global_person_id=global_person_id,
    )


def test_high_score_candidate_is_accepted_and_assigned() -> None:
    pipeline = GlobalIdentityPipeline()

    first_track = TrackReference("CAM01", 1)
    identity = pipeline.assigner.assign_new_track(first_track)

    candidate = make_candidate(
        camera_id="CAM02",
        track_id=7,
        global_person_id=identity.global_person_id,
    )

    result = pipeline.process_candidate(
        candidate,
        make_score(0.90),
    )

    assert result.accepted is True
    assert result.deferred is False
    assert result.rejected is False
    assert result.decision.decision == IdentityDecision.ACCEPT
    assert result.assignment is not None
    assert result.assignment.global_person_id == 1


def test_middle_score_is_deferred_and_not_assigned() -> None:
    pipeline = GlobalIdentityPipeline()

    first_track = TrackReference("CAM01", 1)
    identity = pipeline.assigner.assign_new_track(first_track)

    candidate = make_candidate(
        camera_id="CAM02",
        track_id=7,
        global_person_id=identity.global_person_id,
    )

    result = pipeline.process_candidate(
        candidate,
        make_score(0.75),
    )

    assert result.accepted is False
    assert result.deferred is True
    assert result.rejected is False
    assert result.decision.decision == IdentityDecision.DEFER
    assert result.assignment is None

    assert pipeline.get_assignment("CAM02", 7) is None


def test_low_score_candidate_is_rejected_and_not_assigned() -> None:
    pipeline = GlobalIdentityPipeline()

    first_track = TrackReference("CAM01", 1)
    identity = pipeline.assigner.assign_new_track(first_track)

    candidate = make_candidate(
        camera_id="CAM02",
        track_id=7,
        global_person_id=identity.global_person_id,
    )

    result = pipeline.process_candidate(
        candidate,
        make_score(0.50),
    )

    assert result.accepted is False
    assert result.deferred is False
    assert result.rejected is True
    assert result.decision.decision == IdentityDecision.REJECT
    assert result.assignment is None

    assert pipeline.get_assignment("CAM02", 7) is None


def test_multiple_camera_tracks_can_share_one_identity() -> None:
    pipeline = GlobalIdentityPipeline()

    first_track = TrackReference("CAM01", 1)
    identity = pipeline.assigner.assign_new_track(first_track)

    candidate_2 = make_candidate(
        camera_id="CAM02",
        track_id=7,
        global_person_id=identity.global_person_id,
    )

    candidate_3 = make_candidate(
        camera_id="CAM03",
        track_id=4,
        global_person_id=identity.global_person_id,
    )

    result_2 = pipeline.process_candidate(
        candidate_2,
        make_score(0.90),
    )

    result_3 = pipeline.process_candidate(
        candidate_3,
        make_score(0.88),
    )

    assert result_2.assignment is not None
    assert result_3.assignment is not None

    tracks = pipeline.get_identity_tracks(1)

    assert tracks == (
        first_track,
        candidate_2.track,
        candidate_3.track,
    )


def test_process_candidates_handles_mixed_decisions() -> None:
    pipeline = GlobalIdentityPipeline()

    first_track = TrackReference("CAM01", 1)
    identity = pipeline.assigner.assign_new_track(first_track)

    candidates = [
        (
            make_candidate(
                camera_id="CAM02",
                track_id=7,
                global_person_id=identity.global_person_id,
            ),
            make_score(0.90),
        ),
        (
            make_candidate(
                camera_id="CAM03",
                track_id=4,
                global_person_id=identity.global_person_id,
            ),
            make_score(0.75),
        ),
        (
            make_candidate(
                camera_id="CAM04",
                track_id=2,
                global_person_id=identity.global_person_id,
            ),
            make_score(0.50),
        ),
    ]

    results = pipeline.process_candidates(candidates)

    assert len(results) == 3

    assert results[0].decision.decision == IdentityDecision.ACCEPT
    assert results[0].assignment is not None

    assert results[1].decision.decision == IdentityDecision.DEFER
    assert results[1].assignment is None

    assert results[2].decision.decision == IdentityDecision.REJECT
    assert results[2].assignment is None


def test_to_dict_contains_pipeline_result_details() -> None:
    pipeline = GlobalIdentityPipeline()

    first_track = TrackReference("CAM01", 1)
    identity = pipeline.assigner.assign_new_track(first_track)

    candidate = make_candidate(
        camera_id="CAM02",
        track_id=7,
        global_person_id=identity.global_person_id,
    )

    result = pipeline.process_candidate(
        candidate,
        make_score(0.90),
    )

    data = result.to_dict()

    assert data["candidate"]["camera_id"] == "CAM02"
    assert data["candidate"]["track_id"] == 7
    assert data["decision"]["decision"] == "accept"
    assert data["assignment"]["global_person_id"] == 1


def test_reset_clears_pipeline_assignments() -> None:
    pipeline = GlobalIdentityPipeline()

    first_track = TrackReference("CAM01", 1)
    pipeline.assigner.assign_new_track(first_track)

    assert pipeline.get_assignment("CAM01", 1) is not None

    pipeline.reset()

    assert pipeline.get_assignment("CAM01", 1) is None


def test_none_candidate_is_rejected() -> None:
    pipeline = GlobalIdentityPipeline()

    with pytest.raises(IdentityPipelineError):
        pipeline.process_candidate(
            None,
            make_score(0.90),
        )


def test_none_score_is_rejected() -> None:
    pipeline = GlobalIdentityPipeline()

    candidate = make_candidate()

    with pytest.raises(IdentityPipelineError):
        pipeline.process_candidate(
            candidate,
            None,
        )


def test_unknown_target_identity_is_reported_as_pipeline_error() -> None:
    pipeline = GlobalIdentityPipeline()

    candidate = make_candidate(
        camera_id="CAM02",
        track_id=7,
        global_person_id=99,
    )

    with pytest.raises(IdentityPipelineError):
        pipeline.process_candidate(
            candidate,
            make_score(0.90),
        )



        