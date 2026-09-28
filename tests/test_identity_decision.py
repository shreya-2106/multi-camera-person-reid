"""Tests for Phase 10 identity decision logic."""

from __future__ import annotations

import pytest

from src.matching.identity_decision import (
    IdentityDecision,
    IdentityDecisionError,
    IdentityDecisionMaker,
)
from src.temporal.scoring import CandidateScore


def make_score(total_score: float) -> CandidateScore:
    """Create a valid candidate score for testing."""
    return CandidateScore(
        appearance_score=total_score,
        temporal_score=1.0,
        topology_score=1.0,
        total_score=total_score,
    )


def test_high_score_is_accepted() -> None:
    decision_maker = IdentityDecisionMaker()

    result = decision_maker.decide(
        make_score(0.90)
    )

    assert result.decision == IdentityDecision.ACCEPT
    assert result.reason == "score_meets_accept_threshold"


def test_boundary_accept_score_is_accepted() -> None:
    decision_maker = IdentityDecisionMaker(
        accept_threshold=0.85,
        defer_threshold=0.65,
    )

    result = decision_maker.decide(
        make_score(0.85)
    )

    assert result.decision == IdentityDecision.ACCEPT


def test_middle_score_is_deferred() -> None:
    decision_maker = IdentityDecisionMaker()

    result = decision_maker.decide(
        make_score(0.75)
    )

    assert result.decision == IdentityDecision.DEFER
    assert result.reason == "score_is_ambiguous"


def test_boundary_defer_score_is_deferred() -> None:
    decision_maker = IdentityDecisionMaker(
        accept_threshold=0.85,
        defer_threshold=0.65,
    )

    result = decision_maker.decide(
        make_score(0.65)
    )

    assert result.decision == IdentityDecision.DEFER


def test_low_score_is_rejected() -> None:
    decision_maker = IdentityDecisionMaker()

    result = decision_maker.decide(
        make_score(0.50)
    )

    assert result.decision == IdentityDecision.REJECT
    assert result.reason == "score_below_defer_threshold"


def test_accept_threshold_must_be_valid() -> None:
    with pytest.raises(IdentityDecisionError):
        IdentityDecisionMaker(
            accept_threshold=1.1,
            defer_threshold=0.65,
        )


def test_defer_threshold_must_be_valid() -> None:
    with pytest.raises(IdentityDecisionError):
        IdentityDecisionMaker(
            accept_threshold=0.85,
            defer_threshold=-0.1,
        )


def test_defer_threshold_must_be_lower_than_accept() -> None:
    with pytest.raises(IdentityDecisionError):
        IdentityDecisionMaker(
            accept_threshold=0.65,
            defer_threshold=0.85,
        )


def test_none_score_is_rejected() -> None:
    decision_maker = IdentityDecisionMaker()

    with pytest.raises(IdentityDecisionError):
        decision_maker.decide(None)  # type: ignore[arg-type]


def test_decision_result_to_dict() -> None:
    decision_maker = IdentityDecisionMaker()

    result = decision_maker.decide(
        make_score(0.90)
    )

    data = result.to_dict()

    assert data["decision"] == "accept"
    assert data["score"]["total_score"] == pytest.approx(0.90)
    assert data["reason"] == "score_meets_accept_threshold"