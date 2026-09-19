"""Phase 4: feed Phase 3 person detections into Ultralytics ByteTrack."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import numpy as np

from src.detection import Detection, Detector
from src.logging_setup import get_logger
from src.tracking.types import TrackedPerson, TrackingVideoResult, write_tracks_csv

logger = get_logger(__name__)

STATUS_OK = "OK"
STATUS_NOT_FOUND = "NOT_FOUND"
STATUS_UNREADABLE = "UNREADABLE"
STATUS_ERROR = "ERROR"


class TrackingError(Exception):
    """Raised for invalid local-tracking configuration or backend setup."""


class _ByteTrackDetections:
    """Small NumPy results adapter required by Ultralytics' BYTETracker API."""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh, self.conf, self.cls = xywh, conf, cls

    def __getitem__(self, index: Any) -> "_ByteTrackDetections":
        return _ByteTrackDetections(self.xywh[index], self.conf[index], self.cls[index])

    def __len__(self) -> int:
        return len(self.conf)


def _validate_tracking_config(config: dict) -> dict:
    """Validate the supported native Ultralytics ByteTrack options."""
    if config.get("tracker", "bytetrack").lower() != "bytetrack":
        raise TrackingError("Phase 4 supports only tracker='bytetrack'")
    defaults = {
        "track_high_thresh": 0.5, "track_low_thresh": 0.1,
        "new_track_thresh": 0.6, "track_buffer": 30,
        "match_thresh": 0.8, "fuse_score": True,
    }
    merged = {**defaults, **config}
    for key in ("track_high_thresh", "track_low_thresh", "new_track_thresh", "match_thresh"):
        value = merged[key]
        if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise TrackingError(f"tracking.{key} must be a number in [0, 1]")
    if merged["track_low_thresh"] > merged["track_high_thresh"]:
        raise TrackingError("tracking.track_low_thresh must not exceed track_high_thresh")
    if not isinstance(merged["track_buffer"], int) or merged["track_buffer"] < 1:
        raise TrackingError("tracking.track_buffer must be a positive integer")
    if not isinstance(merged["fuse_score"], bool):
        raise TrackingError("tracking.fuse_score must be a boolean")
    return merged


class Tracker:
    """Stateful local ByteTrack wrapper for Phase 3 ``Detection`` objects."""

    def __init__(self, tracking_config: Optional[dict] = None, engine: Any = None):
        self.config = _validate_tracking_config(tracking_config or {})
        self._engine = engine if engine is not None else self._create_engine()

    @classmethod
    def from_config(cls, cfg: Any) -> "Tracker":
        return cls(dict(cfg.get("tracking", {})))

    def _create_engine(self) -> Any:
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
        except ImportError as exc:
            raise TrackingError("Ultralytics with native ByteTrack is required for Phase 4") from exc
        return BYTETracker(SimpleNamespace(**self.config))

    @staticmethod
    def _to_backend_input(detections: Sequence[Detection]) -> _ByteTrackDetections:
        if not detections:
            return _ByteTrackDetections(np.empty((0, 4), dtype=np.float32), np.empty(0), np.empty(0))
        xywh = np.array([
            [(d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2, d.x2 - d.x1, d.y2 - d.y1]
            for d in detections
        ], dtype=np.float32)
        return _ByteTrackDetections(xywh, np.array([d.confidence for d in detections]), np.array([d.class_id for d in detections]))

    def update(self, detections: Sequence[Detection]) -> list[TrackedPerson]:
        """Advance ByteTrack by one frame and return only active local tracks."""
        if detections:
            camera_ids = {d.camera_id for d in detections}
            frames = {d.frame_number for d in detections}
            if len(camera_ids) != 1 or len(frames) != 1:
                raise TrackingError("a tracker update must contain detections from one camera and frame")
        raw_tracks = self._engine.update(self._to_backend_input(detections))
        if raw_tracks is None or len(raw_tracks) == 0:
            return []
        by_class = {d.class_id: d.class_name for d in detections}
        camera_id = detections[0].camera_id if detections else None
        frame_number = detections[0].frame_number if detections else max(self._engine.frame_id - 1, 0)
        timestamp = detections[0].timestamp_seconds if detections else None
        result = []
        for row in np.asarray(raw_tracks):
            x1, y1, x2, y2, track_id, confidence, class_id = row[:7]
            result.append(TrackedPerson(camera_id, int(frame_number), timestamp, int(track_id), float(x1), float(y1), float(x2), float(y2), float(confidence), int(class_id), by_class.get(int(class_id), str(int(class_id)))))
        return result

    def track_video(self, video_path: Path, detector: Detector, camera_id: Optional[str] = None,
                    output_csv: Optional[Path] = None, annotate: bool = False,
                    annotated_output_path: Optional[Path] = None, max_frames: Optional[int] = None) -> TrackingVideoResult:
        """Sequentially process a video using the existing detector and local tracker."""
        import cv2
        video_path = Path(video_path)
        if camera_id is None:
            from src.data.video_validator import infer_camera_id
            camera_id = infer_camera_id(video_path.name)
        if not video_path.is_file():
            return TrackingVideoResult(str(video_path), camera_id, STATUS_NOT_FOUND, error="File not found")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return TrackingVideoResult(str(video_path), camera_id, STATUS_UNREADABLE, error="File could not be opened by OpenCV")
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_value = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps_value) if fps_value and fps_value > 0 else None
        writer = None
        if annotate:
            annotated_output_path = Path(annotated_output_path or video_path.with_name(f"{video_path.stem}_tracked.mp4"))
            annotated_output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(annotated_output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps or 25.0, (width, height))
        result = TrackingVideoResult(str(video_path), camera_id, STATUS_OK, fps=fps, width=width, height=height)
        frame_number, started = 0, time.monotonic()
        try:
            while max_frames is None or frame_number < max_frames:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                detections = detector.detect_frame(frame, camera_id, frame_number, frame_number / fps if fps else None)
                tracks = self.update(detections)
                result.tracks.extend(tracks)
                if writer is not None:
                    writer.write(draw_tracks(frame, tracks))
                frame_number += 1
        except Exception as exc:
            logger.exception("Tracking failed at frame %s", frame_number)
            result.status, result.error = STATUS_ERROR, str(exc)
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        result.frames_processed, result.track_count = frame_number, len(result.tracks)
        if output_csv is not None and result.status != STATUS_ERROR:
            result.output_csv_path = str(write_tracks_csv(result.tracks, Path(output_csv)))
        if writer is not None and result.status != STATUS_ERROR:
            result.annotated_video_path = str(annotated_output_path)
        logger.info("Tracked %s: %d frames, %d active track records in %.1fs", video_path.name, frame_number, result.track_count, time.monotonic() - started)
        return result


def draw_tracks(frame: Any, tracks: Sequence[TrackedPerson]) -> Any:
    import cv2
    annotated = frame.copy()
    for track in tracks:
        p1, p2 = (int(track.x1), int(track.y1)), (int(track.x2), int(track.y2))
        cv2.rectangle(annotated, p1, p2, (255, 180, 0), 2)
        cv2.putText(annotated, f"ID {track.track_id} | {track.class_name} | {track.confidence:.2f}", (p1[0], max(p1[1] - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1)
    return annotated


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Phase 4: YOLO person detection with local ByteTrack IDs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--camera-id")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--annotate", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    from src.config import PROJECT_ROOT, load_config
    from src.logging_setup import setup_logging
    args = parse_args(argv)
    setup_logging(level="INFO")
    cfg = load_config(args.config, apply_env=False)
    tracker = Tracker.from_config(cfg)
    detector = Detector.from_config(cfg)
    video_path = Path(args.input)
    from src.data.video_validator import infer_camera_id
    camera_id = args.camera_id or infer_camera_id(video_path.name) or video_path.stem
    output_dir = Path(args.output_dir or cfg.tracking.get("output_dir", "outputs/tracking"))
    if not output_dir.is_absolute(): output_dir = PROJECT_ROOT / output_dir
    ann_dir = Path(cfg.tracking.get("annotated_output_dir", "outputs/tracking_annotated"))
    if not ann_dir.is_absolute(): ann_dir = PROJECT_ROOT / ann_dir
    result = tracker.track_video(video_path, detector, camera_id, output_dir / f"{camera_id}_tracks.csv", args.annotate, ann_dir / f"{camera_id}_tracked.mp4", args.max_frames)
    detector.close()
    print(f"\nTracking status: {result.status}\nFrames processed: {result.frames_processed}\nTrack records: {result.track_count}\nCSV output: {result.output_csv_path}")
    if result.annotated_video_path: print(f"Annotated video: {result.annotated_video_path}")
    if result.error: print(f"Note: {result.error}")
    return 0 if result.status == STATUS_OK else 1


if __name__ == "__main__":
    sys.exit(main())
