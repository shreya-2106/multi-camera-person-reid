"""
FAISS-based similarity matching for Phase 8.

This module:
1. Loads OSNet embedding vectors from embeddings.npy files.
2. L2-normalizes the vectors.
3. Builds a FAISS IndexFlatIP index.
4. Performs nearest-neighbor similarity search.
5. Returns the associated camera/track/frame metadata.

Important:
- Inner product on L2-normalized vectors is cosine similarity.
- This module performs similarity retrieval only.
- It does NOT assign global person IDs.
- It does NOT perform final cross-camera identity decisions.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np


@dataclass(frozen=True)
class MatchResult:
    """One FAISS nearest-neighbor result."""

    camera_id: str
    track_id: int
    frame_number: int
    crop_path: str
    embedding_index: int
    similarity: float

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "frame_number": self.frame_number,
            "crop_path": self.crop_path,
            "embedding_index": self.embedding_index,
            "similarity": self.similarity,
        }


class MatcherError(RuntimeError):
    """Raised when FAISS matching cannot be completed."""


class FAISSMatcher:
    """FAISS cosine-similarity matcher for OSNet embeddings."""

    def __init__(
        self,
        similarity_threshold: float = 0.6,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold must be between 0.0 and 1.0"
            )

        self.similarity_threshold = similarity_threshold

        self.index: Optional[faiss.Index] = None
        self.metadata: list[dict] = []
        self.embedding_dimension: Optional[int] = None

    def build_index(
        self,
        embeddings: np.ndarray,
        metadata: list[dict],
    ) -> None:
        """
        Build a FAISS inner-product index.

        Embeddings are L2-normalized before indexing so that
        inner product is equivalent to cosine similarity.
        """
        if not isinstance(embeddings, np.ndarray):
            raise MatcherError("embeddings must be a NumPy array")

        if embeddings.ndim != 2:
            raise MatcherError(
                f"embeddings must have shape (N, D), got {embeddings.shape}"
            )

        if len(embeddings) == 0:
            raise MatcherError("Cannot build an index from zero embeddings")

        if len(metadata) != len(embeddings):
            raise MatcherError(
                "Number of metadata records must match number of embeddings"
            )

        vectors = np.asarray(
            embeddings,
            dtype=np.float32,
        ).copy()

        faiss.normalize_L2(vectors)

        self.embedding_dimension = vectors.shape[1]

        self.index = faiss.IndexFlatIP(
            self.embedding_dimension
        )

        self.index.add(vectors)

        self.metadata = list(metadata)

    def build_from_directory(
        self,
        embeddings_root: str | Path,
    ) -> int:
        """
        Load all embeddings.npy files below an embeddings directory
        and build one combined FAISS index.

        Expected structure:

        outputs/
        └── embeddings/
            ├── CAM01/
            │   ├── embeddings.csv
            │   ├── track_001/
            │   │   └── embeddings.npy
            │   └── track_002/
            │       └── embeddings.npy
            └── CAM02/
                └── ...

        Returns:
            Number of indexed embeddings.
        """
        root = Path(embeddings_root)

        if not root.exists():
            raise MatcherError(
                f"Embeddings directory does not exist: {root}"
            )

        embedding_files = sorted(
            root.rglob("embeddings.npy")
        )

        if not embedding_files:
            raise MatcherError(
                f"No embeddings.npy files found under: {root}"
            )

        all_embeddings: list[np.ndarray] = []
        all_metadata: list[dict] = []

        for embedding_path in embedding_files:
            embeddings = np.load(embedding_path)

            if embeddings.ndim != 2:
                raise MatcherError(
                    f"Invalid embedding shape in {embedding_path}: "
                    f"{embeddings.shape}"
                )

            csv_path = (
                embedding_path.parent.parent
                / "embeddings.csv"
            )

            if not csv_path.exists():
                raise MatcherError(
                    f"Missing metadata CSV for {embedding_path}: "
                    f"{csv_path}"
                )

            metadata = self._load_metadata(csv_path)

            track_embedding_count = len(embeddings)

            # Example:
            # track_001 -> track_id 1
            # track_002 -> track_id 2
            # track_003 -> track_id 3
            track_directory_name = embedding_path.parent.name

            try:
                track_id = int(
                    track_directory_name.split("_")[-1]
                )
            except ValueError as exc:
                raise MatcherError(
                    f"Could not determine track ID from directory: "
                    f"{embedding_path.parent}"
                ) from exc

            # The camera-level embeddings.csv contains metadata
            # for all tracks. Select only the rows belonging to
            # the current embeddings.npy file.
            track_metadata = [
                row
                for row in metadata
                if int(row["track_id"]) == track_id
            ]

            if len(track_metadata) != track_embedding_count:
                raise MatcherError(
                    f"Metadata count ({len(track_metadata)}) does not "
                    f"match embeddings ({track_embedding_count}) for "
                    f"{embedding_path}"
                )

            # Make sure metadata order matches the row order in
            # embeddings.npy.
            track_metadata.sort(
                key=lambda row: int(
                    row["embedding_index"]
                )
            )

            all_embeddings.append(
                np.asarray(
                    embeddings,
                    dtype=np.float32,
                )
            )

            all_metadata.extend(track_metadata)

        combined = np.vstack(all_embeddings)

        self.build_index(
            combined,
            all_metadata,
        )

        return int(combined.shape[0])

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[MatchResult]:
        """
        Search the FAISS index for the nearest embeddings.

        Results below similarity_threshold are excluded.
        """
        if self.index is None:
            raise MatcherError(
                "FAISS index has not been built"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.ndim != 2:
            raise MatcherError(
                f"Query embedding must have shape (D,) or (1, D), "
                f"got {query.shape}"
            )

        if query.shape[1] != self.embedding_dimension:
            raise MatcherError(
                f"Query dimension {query.shape[1]} does not match "
                f"index dimension {self.embedding_dimension}"
            )

        faiss.normalize_L2(query)

        k = min(
            top_k,
            self.index.ntotal,
        )

        similarities, indices = self.index.search(
            query,
            k,
        )

        results: list[MatchResult] = []

        for similarity, index_position in zip(
            similarities[0],
            indices[0],
        ):
            if index_position < 0:
                continue

            similarity_value = float(
                similarity
            )

            if similarity_value < self.similarity_threshold:
                continue

            metadata = self.metadata[
                int(index_position)
            ]

            results.append(
                MatchResult(
                    camera_id=str(
                        metadata["camera_id"]
                    ),
                    track_id=int(
                        metadata["track_id"]
                    ),
                    frame_number=int(
                        metadata["frame_number"]
                    ),
                    crop_path=str(
                        metadata["crop_path"]
                    ),
                    embedding_index=int(
                        metadata["embedding_index"]
                    ),
                    similarity=similarity_value,
                )
            )

        return results

    @staticmethod
    def _load_metadata(
        csv_path: Path,
    ) -> list[dict]:
        """Load embedding metadata from embeddings.csv."""
        with csv_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as handle:
            reader = csv.DictReader(handle)

            required_fields = {
                "camera_id",
                "track_id",
                "frame_number",
                "crop_path",
                "embedding_index",
            }

            if not required_fields.issubset(
                set(reader.fieldnames or [])
            ):
                raise MatcherError(
                    f"Missing required metadata fields in {csv_path}"
                )

            return list(reader)

