"""Typed output records for Phase 6 OSNet embedding extraction."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

EMBEDDING_CSV_FIELDNAMES = [
    "camera_id", "track_id", "frame_number", "crop_path", "embedding_index",
]


@dataclass(frozen=True)
class EmbeddingMetadata:
    """Maps one vector in a local-track ``embeddings.npy`` file to its crop."""

    camera_id: str
    track_id: int
    frame_number: int
    crop_path: str
    embedding_index: int

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id must be non-empty")
        if not isinstance(self.track_id, int) or self.track_id < 0:
            raise ValueError("track_id must be a non-negative integer")
        if not isinstance(self.frame_number, int) or self.frame_number < 0:
            raise ValueError("frame_number must be a non-negative integer")
        if not self.crop_path:
            raise ValueError("crop_path must be non-empty")
        if not isinstance(self.embedding_index, int) or self.embedding_index < 0:
            raise ValueError("embedding_index must be a non-negative integer")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EmbeddingExtractionResult:
    """Summary of one directory-level embedding extraction run."""

    input_directory: str
    camera_id: Optional[str]
    status: str
    images_processed: int = 0
    embeddings_generated: int = 0
    images_skipped: int = 0
    embedding_dimension: Optional[int] = None
    device: Optional[str] = None
    output_directory: Optional[str] = None
    metadata_csv_paths: list[str] = field(default_factory=list)
    error: Optional[str] = None


def write_embeddings_csv(metadata: list[EmbeddingMetadata], output_path: Path) -> Path:
    """Write stable embedding metadata, including a header for an empty camera."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EMBEDDING_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(item.to_dict() for item in metadata)
    return output_path
