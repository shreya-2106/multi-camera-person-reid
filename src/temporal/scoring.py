"""Evidence scoring for cross-camera Re-ID candidates."""

from __future__ import annotations

from dataclasses import dataclass

from .evidence import TransitionEvidence


class ScoringError(RuntimeError):
    """Raised when candidate evidence cannot be scored."""


@dataclass(frozen=True)
class CandidateScore:
    """Transparent weighted score for one cross-camera candidate."""

    appearance_score: float
    temporal_score: float
    topology_score: float
    total_score: float

    def __post_init__(self) -> None:
        for name, value in (
            ("appearance_score", self.appearance_score),
            ("temporal_score", self.temporal_score),
            ("topology_score", self.topology_score),
            ("total_score", self.total_score),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1"
                )

    def to_dict(self) -> dict:
        """Return the score as a serializable dictionary."""
        return {
            "appearance_score": self.appearance_score,
            "temporal_score": self.temporal_score,
            "topology_score": self.topology_score,
            "total_score": self.total_score,
        }


class CandidateScorer:
    """Calculate transparent evidence scores."""

    def __init__(
        self,
        appearance_weight: float = 0.7,
        temporal_weight: float = 0.2,
        topology_weight: float = 0.1,
    ) -> None:
        weights = (
            appearance_weight,
            temporal_weight,
            topology_weight,
        )

        if any(weight < 0.0 for weight in weights):
            raise ScoringError(
                "Scoring weights must be non-negative"
            )

        total_weight = sum(weights)

        if total_weight <= 0.0:
            raise ScoringError(
                "At least one scoring weight must be positive"
            )

        self.appearance_weight = (
            appearance_weight / total_weight
        )
        self.temporal_weight = (
            temporal_weight / total_weight
        )
        self.topology_weight = (
            topology_weight / total_weight
        )

    @staticmethod
    def _appearance_score(
        evidence: TransitionEvidence,
    ) -> float:
        """Use mean appearance similarity as the base score."""
        return max(
            0.0,
            min(
                1.0,
                evidence.mean_similarity,
            ),
        )

    @staticmethod
    def _temporal_score(
        evidence: TransitionEvidence,
    ) -> float:
        """Represent temporal validity as a binary evidence signal."""
        return 1.0 if evidence.temporal_valid else 0.0

    @staticmethod
    def _topology_score(
        evidence: TransitionEvidence,
    ) -> float:
        """Represent topology validity as a binary evidence signal."""
        return 1.0 if evidence.topology_valid else 0.0

    def score(
        self,
        evidence: TransitionEvidence,
    ) -> CandidateScore:
        """Calculate a weighted candidate evidence score."""
        if evidence is None:
            raise ScoringError(
                "Evidence cannot be None"
            )

        appearance = self._appearance_score(
            evidence
        )

        temporal = self._temporal_score(
            evidence
        )

        topology = self._topology_score(
            evidence
        )

        total = (
            appearance * self.appearance_weight
            + temporal * self.temporal_weight
            + topology * self.topology_weight
        )

        return CandidateScore(
            appearance_score=appearance,
            temporal_score=temporal,
            topology_score=topology,
            total_score=max(
                0.0,
                min(1.0, total),
            ),
        )