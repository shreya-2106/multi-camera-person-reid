"""Tests for Phase 14 candidate reporting."""

from src.matching.candidate_report import CandidateReport
from src.temporal.scoring import CandidateScore
from src.temporal.transition import TransitionCandidate
from src.temporal.evidence import TransitionEvidence


def test_valid_candidate_report():
    transition = TransitionCandidate(
        from_camera="CAM01",
        from_track_id=1,
        to_camera="CAM02",
        to_track_id=7,
        exit_time=5.0,
        entry_time=10.0,
        travel_time_seconds=5.0,
    )

    evidence = TransitionEvidence(
        from_camera="CAM01",
        from_track_id=1,
        to_camera="CAM02",
        to_track_id=7,
        mean_similarity=0.95,
        max_similarity=0.98,
        comparison_count=10,
        travel_time_seconds=5.0,
        topology_valid=True,
        temporal_valid=True,
    )

    score = CandidateScore(
        appearance_score=0.95,
        temporal_score=1.0,
        topology_score=1.0,
        total_score=0.965,
    )

    report = CandidateReport(
        from_camera="CAM01",
        from_track_id=1,
        to_camera="CAM02",
        to_track_id=7,
        status="valid",
        travel_time_seconds=5.0,
        transition=transition,
        evidence=evidence,
        score=score,
    )

    result = report.to_dict()

    assert result["status"] == "valid"
    assert result["reason"] is None
    assert result["travel_time_seconds"] == 5.0
    assert result["transition"]["travel_time_seconds"] == 5.0
    assert result["evidence"]["mean_similarity"] == 0.95
    assert result["score"]["total_score"] == 0.965


def test_rejected_candidate_report():
    report = CandidateReport(
        from_camera="CAM01",
        from_track_id=7,
        to_camera="CAM02",
        to_track_id=17,
        status="rejected",
        reason="invalid travel time",
        travel_time_seconds=49.72,
    )

    result = report.to_dict()

    assert result["status"] == "rejected"
    assert result["reason"] == "invalid travel time"
    assert result["travel_time_seconds"] == 49.72
    assert result["transition"] is None
    assert result["evidence"] is None
    assert result["score"] is None


def test_rejected_report_requires_reason():
    try:
        CandidateReport(
            from_camera="CAM01",
            from_track_id=1,
            to_camera="CAM02",
            to_track_id=7,
            status="rejected",
        )
    except ValueError as exc:
        assert "reason" in str(exc)
    else:
        raise AssertionError(
            "Rejected candidate without a reason should fail"
        )


def test_same_camera_report_is_rejected():
    try:
        CandidateReport(
            from_camera="CAM01",
            from_track_id=1,
            to_camera="CAM01",
            to_track_id=7,
            status="valid",
        )
    except ValueError as exc:
        assert "cross-camera" in str(exc)
    else:
        raise AssertionError(
            "Same-camera candidate report should fail"
        )