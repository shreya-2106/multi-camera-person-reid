"""Phase 12 cross-camera Re-ID integration.

This module connects the existing Phase 8, Phase 9, and Phase 10
components without replacing their responsibilities.

Flow:

track embeddings
    -> TrackMatcher
    -> TransitionCandidateBuilder
    -> TransitionEvidenceBuilder
    -> CandidateScorer
    -> GlobalIdentityPipeline
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from src.matching.candidate_report import CandidateReport
from src.matching.identity_assigner import IdentityCandidate
from src.matching.identity_pipeline import (
    GlobalIdentityPipeline,
    IdentityPipelineResult,
)
from src.matching.types import IdentityEvidence, TrackReference
from src.reid.track_matcher import (
    TrackMatcher,
    TrackSimilarityResult,
)
from src.temporal.evidence import TransitionEvidenceBuilder
from src.temporal.scoring import CandidateScore, CandidateScorer
from src.temporal.topology import CameraTopology
from src.temporal.transition import (
    TransitionCandidate,
    TransitionCandidateBuilder,
)
from src.temporal.types import TrackSummary


class CrossCameraMatcherError(RuntimeError):
    """Raised when cross-camera matching cannot be completed."""


@dataclass(frozen=True)
class CrossCameraMatchResult:
    """Result of evaluating one cross-camera track pair."""

    from_track: TrackReference
    to_track: TrackReference
    similarity: TrackSimilarityResult
    transition: TransitionCandidate
    score: CandidateScore
    identity_result: IdentityPipelineResult
    report: CandidateReport

    def to_dict(self) -> dict:
        """Convert the result to a serializable dictionary."""
        return {
            "from_track": self.from_track.to_dict(),
            "to_track": self.to_track.to_dict(),
            "similarity": self.similarity.to_dict(),
            "transition": self.transition.to_dict(),
            "score": self.score.to_dict(),
            "identity_result": self.identity_result.to_dict(),
            "report": self.report.to_dict(),
        }


class CrossCameraMatcher:
    """Orchestrate cross-camera Re-ID using existing components."""

    def __init__(
        self,
        topology: CameraTopology,
        track_matcher: TrackMatcher | None = None,
        scorer: CandidateScorer | None = None,
        identity_pipeline: GlobalIdentityPipeline | None = None,
    ) -> None:
        if topology is None:
            raise CrossCameraMatcherError(
                "Camera topology cannot be None"
            )

        self.topology = topology

        self.track_matcher = (
            track_matcher
            if track_matcher is not None
            else TrackMatcher()
        )

        self.scorer = (
            scorer
            if scorer is not None
            else CandidateScorer()
        )

        self.identity_pipeline = (
            identity_pipeline
            if identity_pipeline is not None
            else GlobalIdentityPipeline()
        )

        self.transition_builder = TransitionCandidateBuilder(
            topology=topology
        )

        self.evidence_builder = TransitionEvidenceBuilder()

    @staticmethod
    def load_track_embeddings(
        embeddings_path: str | Path,
    ):
        """Load one track's embeddings using the existing matcher."""
        return TrackMatcher.load_track_embeddings(
            embeddings_path
        )

    def compare_embedding_files(
        self,
        from_embeddings_path: str | Path,
        to_embeddings_path: str | Path,
    ) -> TrackSimilarityResult:
        """Compare two track embedding files."""
        return self.track_matcher.compare_tracks(
            from_embeddings_path,
            to_embeddings_path,
        )

    def build_transition(
        self,
        from_track: TrackSummary,
        to_track: TrackSummary,
        from_offset_seconds: float = 0.0,
        to_offset_seconds: float = 0.0,
    ) -> TransitionCandidate | None:
        """Build a valid topology/time transition candidate."""
        return self.transition_builder.build_pair(
            from_track,
            to_track,
            from_offset_seconds,
            to_offset_seconds,
        )

    def build_candidate(
        self,
        from_track: TrackSummary,
        to_track: TrackSummary,
        similarity: TrackSimilarityResult,
        target_global_person_id: int,
        from_offset_seconds: float = 0.0,
        to_offset_seconds: float = 0.0,
    ) -> tuple[IdentityCandidate, CandidateScore] | None:
        """Build an identity candidate and score it."""
        transition = self.build_transition(
            from_track,
            to_track,
            from_offset_seconds,
            to_offset_seconds,
        )

        if transition is None:
            return None

        evidence = self.evidence_builder.build(
            transition,
            similarity,
        )

        score = self.scorer.score(
            evidence
        )

        identity_evidence = IdentityEvidence(
            mean_similarity=evidence.mean_similarity,
            max_similarity=evidence.max_similarity,
            comparison_count=evidence.comparison_count,
            travel_time_seconds=evidence.travel_time_seconds,
            topology_valid=evidence.topology_valid,
            temporal_valid=evidence.temporal_valid,
        )

        candidate = IdentityCandidate(
            track=TrackReference(
                camera_id=to_track.camera_id,
                track_id=to_track.track_id,
            ),
            evidence=identity_evidence,
            target_global_person_id=target_global_person_id,
        )

        return candidate, score

    def process_candidate(
        self,
        candidate: IdentityCandidate,
        score: CandidateScore,
    ) -> IdentityPipelineResult:
        """Send one candidate through the existing Phase 10 pipeline."""
        return self.identity_pipeline.process_candidate(
            candidate,
            score,
        )

    def match_track_pair(
        self,
        from_track: TrackSummary,
        to_track: TrackSummary,
        from_embeddings_path: str | Path,
        to_embeddings_path: str | Path,
        target_global_person_id: int,
        from_offset_seconds: float = 0.0,
        to_offset_seconds: float = 0.0,
        similarity: TrackSimilarityResult | None = None,
    ) -> CrossCameraMatchResult | None:
        """Evaluate one complete cross-camera track pair."""
        if from_track.camera_id == to_track.camera_id:
            raise CrossCameraMatcherError(
                "Cross-camera matching requires different cameras"
            )

        if similarity is None:
            similarity = self.compare_embedding_files(
                from_embeddings_path,
                to_embeddings_path,
            )

        built = self.build_candidate(
            from_track=from_track,
            to_track=to_track,
            similarity=similarity,
            target_global_person_id=target_global_person_id,
            from_offset_seconds=from_offset_seconds,
            to_offset_seconds=to_offset_seconds,
        )

        if built is None:
            return None

        candidate, score = built

        transition = self.build_transition(
            from_track,
            to_track,
            from_offset_seconds,
            to_offset_seconds,
        )

        if transition is None:
            return None

        identity_result = self.process_candidate(
            candidate,
            score,
        )

        evidence = self.evidence_builder.build(
            transition,
            similarity,
        )

        report = CandidateReport(
            from_camera=from_track.camera_id,
            from_track_id=from_track.track_id,
            to_camera=to_track.camera_id,
            to_track_id=to_track.track_id,
            status="valid",
            travel_time_seconds=transition.travel_time_seconds,
            transition=transition,
            evidence=evidence,
            score=score,
            timing_basis="video_relative_plus_configured_offsets",
            timeline_offsets_seconds={
                from_track.camera_id: from_offset_seconds,
                to_track.camera_id: to_offset_seconds,
            },
            timeline_offset_source="user_provided",
            common_time_origin=True,
        )

        return CrossCameraMatchResult(
            from_track=TrackReference(
                camera_id=from_track.camera_id,
                track_id=from_track.track_id,
            ),
            to_track=TrackReference(
                camera_id=to_track.camera_id,
                track_id=to_track.track_id,
            ),
            similarity=similarity,
            transition=transition,
            score=score,
            identity_result=identity_result,
            report=report,
        )
