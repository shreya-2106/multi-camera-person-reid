import pytest

from src.reid.track_matcher import TrackSimilarityResult
from src.temporal.evidence import (
    TransitionEvidenceBuilder,
)
from src.temporal.transition import TransitionCandidate


def make_candidate() -> TransitionCandidate:
    return TransitionCandidate(
        from_camera="CAM01",
        from_track_id=1,
        to_camera="CAM02",
        to_track_id=7,
        exit_time=10.0,
        entry_time=12.0,
        travel_time_seconds=2.0,
    )


def make_similarity() -> TrackSimilarityResult:
    return TrackSimilarityResult(
        mean_similarity=0.72,
        max_similarity=0.91,
        comparison_count=90,
    )


def test_build_transition_evidence():
    builder = TransitionEvidenceBuilder()

    evidence = builder.build(
        make_candidate(),
        make_similarity(),
    )

    assert evidence.from_camera == "CAM01"
    assert evidence.from_track_id == 1

    assert evidence.to_camera == "CAM02"
    assert evidence.to_track_id == 7

    assert evidence.mean_similarity == pytest.approx(
        0.72
    )

    assert evidence.max_similarity == pytest.approx(
        0.91
    )

    assert evidence.comparison_count == 90

    assert evidence.travel_time_seconds == pytest.approx(
        2.0
    )

    assert evidence.topology_valid is True
    assert evidence.temporal_valid is True


def test_evidence_to_dict():
    builder = TransitionEvidenceBuilder()

    evidence = builder.build(
        make_candidate(),
        make_similarity(),
    )

    data = evidence.to_dict()

    assert data["from_camera"] == "CAM01"
    assert data["from_track_id"] == 1
    assert data["to_camera"] == "CAM02"
    assert data["to_track_id"] == 7

    assert data["mean_similarity"] == pytest.approx(
        0.72
    )

    assert data["max_similarity"] == pytest.approx(
        0.91
    )

    assert data["comparison_count"] == 90
    assert data["travel_time_seconds"] == pytest.approx(
        2.0
    )

    assert data["topology_valid"] is True
    assert data["temporal_valid"] is True


def test_invalid_mean_similarity():
    with pytest.raises(ValueError):
        from src.temporal.evidence import TransitionEvidence

        TransitionEvidence(
            from_camera="CAM01",
            from_track_id=1,
            to_camera="CAM02",
            to_track_id=7,
            mean_similarity=1.5,
            max_similarity=0.9,
            comparison_count=10,
            travel_time_seconds=2.0,
            topology_valid=True,
            temporal_valid=True,
        )


def test_invalid_comparison_count():
    with pytest.raises(ValueError):
        from src.temporal.evidence import TransitionEvidence

        TransitionEvidence(
            from_camera="CAM01",
            from_track_id=1,
            to_camera="CAM02",
            to_track_id=7,
            mean_similarity=0.7,
            max_similarity=0.9,
            comparison_count=0,
            travel_time_seconds=2.0,
            topology_valid=True,
            temporal_valid=True,
        )


def test_invalid_same_camera():
    with pytest.raises(ValueError):
        from src.temporal.evidence import TransitionEvidence

        TransitionEvidence(
            from_camera="CAM01",
            from_track_id=1,
            to_camera="CAM01",
            to_track_id=7,
            mean_similarity=0.7,
            max_similarity=0.9,
            comparison_count=10,
            travel_time_seconds=2.0,
            topology_valid=True,
            temporal_valid=True,
        )