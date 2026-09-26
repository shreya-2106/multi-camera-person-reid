import pytest

from src.temporal.evidence import TransitionEvidence
from src.temporal.scoring import (
    CandidateScorer,
    ScoringError,
)


def make_evidence(
    mean_similarity: float = 0.7,
    topology_valid: bool = True,
    temporal_valid: bool = True,
) -> TransitionEvidence:
    return TransitionEvidence(
        from_camera="CAM01",
        from_track_id=1,
        to_camera="CAM02",
        to_track_id=2,
        mean_similarity=mean_similarity,
        max_similarity=0.9,
        comparison_count=10,
        travel_time_seconds=5.0,
        topology_valid=topology_valid,
        temporal_valid=temporal_valid,
    )


def test_default_score():
    scorer = CandidateScorer()

    result = scorer.score(
        make_evidence(mean_similarity=0.7)
    )

    assert result.appearance_score == pytest.approx(0.7)
    assert result.temporal_score == pytest.approx(1.0)
    assert result.topology_score == pytest.approx(1.0)

    assert result.total_score == pytest.approx(0.79)


def test_weights_are_normalized():
    scorer = CandidateScorer(
        appearance_weight=7.0,
        temporal_weight=2.0,
        topology_weight=1.0,
    )

    result = scorer.score(
        make_evidence(mean_similarity=0.7)
    )

    assert result.total_score == pytest.approx(0.79)


def test_invalid_temporal_evidence():
    scorer = CandidateScorer()

    result = scorer.score(
        make_evidence(
            mean_similarity=0.7,
            temporal_valid=False,
        )
    )

    assert result.temporal_score == 0.0
    assert result.topology_score == 1.0

    assert result.total_score == pytest.approx(0.59)


def test_invalid_topology_evidence():
    scorer = CandidateScorer()

    result = scorer.score(
        make_evidence(
            mean_similarity=0.7,
            topology_valid=False,
        )
    )

    assert result.temporal_score == 1.0
    assert result.topology_score == 0.0

    assert result.total_score == pytest.approx(0.69)


def test_invalid_both_temporal_and_topology():
    scorer = CandidateScorer()

    result = scorer.score(
        make_evidence(
            mean_similarity=0.7,
            topology_valid=False,
            temporal_valid=False,
        )
    )

    assert result.temporal_score == 0.0
    assert result.topology_score == 0.0

    assert result.total_score == pytest.approx(0.49)


def test_weights_are_rejected_when_negative():
    with pytest.raises(ScoringError):
        CandidateScorer(
            appearance_weight=-1.0,
            temporal_weight=1.0,
            topology_weight=1.0,
        )


def test_all_zero_weights_are_rejected():
    with pytest.raises(ScoringError):
        CandidateScorer(
            appearance_weight=0.0,
            temporal_weight=0.0,
            topology_weight=0.0,
        )


def test_score_to_dict():
    scorer = CandidateScorer()

    result = scorer.score(
        make_evidence(mean_similarity=0.7)
    )

    data = result.to_dict()

    assert data["appearance_score"] == pytest.approx(0.7)
    assert data["temporal_score"] == pytest.approx(1.0)
    assert data["topology_score"] == pytest.approx(1.0)
    assert data["total_score"] == pytest.approx(0.79)