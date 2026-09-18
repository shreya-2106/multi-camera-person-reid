"""
Person detection module (Phase 3): YOLO-based, person-only, per-frame
detection. No tracking or re-identification happens here.

Public API:
    Detector             - loads a YOLO model and runs detection on frames/videos
    DetectorError        - raised on model load/inference failures
    Detection            - a single person detection (dataclass)
    DetectionVideoResult  - the outcome of running detection over a video
"""

from src.detection.detector import Detector, DetectorError
from src.detection.types import Detection, DetectionVideoResult

__all__ = ["Detector", "DetectorError", "Detection", "DetectionVideoResult"]