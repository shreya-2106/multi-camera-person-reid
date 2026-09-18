"""Phase 2 dataset and CCTV video validation utilities."""

from .dataset_validator import validate_datasets
from .video_validator import validate_videos

__all__ = ["validate_datasets", "validate_videos"]
