"""
Track-level appearance similarity for Phase 8.

This module compares two local tracks using their OSNet embeddings.

It does not:
- assign global person IDs
- apply camera topology
- apply temporal constraints
- make final identity decisions

It only calculates appearance similarity between two tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class TrackMatcherError(RuntimeError):
    """Raised when track-level matching cannot be completed."""


@dataclass(frozen=True)
class TrackSimilarityResult:
    """Aggregated appearance similarity between two tracks."""

    mean_similarity: float
    max_similarity: float
    comparison_count: int

    def to_dict(self) -> dict:
        return {
            "mean_similarity": self.mean_similarity,
            "max_similarity": self.max_similarity,
            "comparison_count": self.comparison_count,
        }


class TrackMatcher:
    """Compare local tracks using OSNet appearance embeddings."""

    @staticmethod
    def load_track_embeddings(
        embeddings_path: str | Path,
    ) -> np.ndarray:
        """Load and validate one track's embeddings.npy file."""
        path = Path(embeddings_path)

        if not path.exists():
            raise TrackMatcherError(
                f"Embeddings file does not exist: {path}"
            )

        embeddings = np.asarray(
            np.load(path),
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise TrackMatcherError(
                f"Track embeddings must have shape (N, D), "
                f"got {embeddings.shape}"
            )

        if embeddings.shape[0] == 0:
            raise TrackMatcherError(
                f"Track contains zero embeddings: {path}"
            )

        if embeddings.shape[1] == 0:
            raise TrackMatcherError(
                f"Track embeddings have zero dimensions: {path}"
            )

        return embeddings

    @staticmethod
    def _normalize(
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """L2-normalize embedding vectors."""
        vectors = np.asarray(
            embeddings,
            dtype=np.float32,
        ).copy()

        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True,
        )

        if np.any(norms <= 0.0):
            raise TrackMatcherError(
                "Track contains an embedding with zero norm"
            )

        return vectors / norms

    def compare_embeddings(
        self,
        track_a: np.ndarray,
        track_b: np.ndarray,
    ) -> TrackSimilarityResult:
        """
        Compare every embedding in track A against every embedding
        in track B.

        Returns aggregate cosine-similarity statistics.
        """
        a = np.asarray(track_a, dtype=np.float32)
        b = np.asarray(track_b, dtype=np.float32)

        if a.ndim != 2 or b.ndim != 2:
            raise TrackMatcherError(
                "Both tracks must have shape (N, D)"
            )

        if a.shape[1] != b.shape[1]:
            raise TrackMatcherError(
                f"Embedding dimensions do not match: "
                f"{a.shape[1]} vs {b.shape[1]}"
            )

        a = self._normalize(a)
        b = self._normalize(b)

        similarity_matrix = a @ b.T

        similarities = similarity_matrix.reshape(-1)

        return TrackSimilarityResult(
            mean_similarity=float(
                np.mean(similarities)
            ),
            max_similarity=float(
                np.max(similarities)
            ),
            comparison_count=int(
                similarities.size
            ),
        )

    def compare_tracks(
        self,
        track_a_path: str | Path,
        track_b_path: str | Path,
    ) -> TrackSimilarityResult:
        """Load two tracks and compare their embeddings."""
        track_a = self.load_track_embeddings(
            track_a_path
        )

        track_b = self.load_track_embeddings(
            track_b_path
        )

        return self.compare_embeddings(
            track_a,
            track_b,
        )