from pathlib import Path

import numpy as np
import pytest

from src.detection.types import Detection
from src.tracking.tracker import STATUS_NOT_FOUND, STATUS_OK, Tracker, TrackingError

cv2 = pytest.importorskip("cv2")


def detection(frame=0, x1=10.0, camera="CAM01"):
    return Detection(camera, frame, frame / 10, 0, "person", 0.9, x1, 20, x1 + 30, 80)


class FakeEngine:
    def __init__(self): self.frame_id = 0; self.last = None
    def update(self, values):
        self.frame_id += 1
        self.last = values
        if not len(values): return np.empty((0, 8))
        return np.array([[10, 20, 40, 80, 12, .9, 0, 0]])


def test_tracker_accepts_detection_and_returns_typed_local_id():
    engine = FakeEngine()
    tracks = Tracker(engine=engine).update([detection()])
    assert tracks[0].track_id == 12
    assert isinstance(tracks[0].track_id, int)
    assert tracks[0].camera_id == "CAM01"
    assert tracks[0].class_name == "person"
    assert engine.last.xywh.shape == (1, 4)


def test_native_ultralytics_bytetrack_assigns_a_local_id():
    """Exercise the installed ByteTrack backend without loading a YOLO model."""
    tracks = Tracker({"tracker": "bytetrack", "new_track_thresh": 0.5}).update([detection()])
    assert len(tracks) == 1
    assert isinstance(tracks[0].track_id, int)


def test_empty_and_multiple_detections_are_safe():
    tracker = Tracker(engine=FakeEngine())
    assert tracker.update([]) == []
    tracks = tracker.update([detection(), detection(x1=60)])
    assert len(tracks) == 1


def test_tracker_rejects_mixed_frame_or_camera_updates():
    with pytest.raises(TrackingError): Tracker(engine=FakeEngine()).update([detection(), detection(frame=1)])
    with pytest.raises(TrackingError): Tracker(engine=FakeEngine()).update([detection(), detection(camera="CAM02")])


class FakeDetector:
    def detect_frame(self, frame, camera_id, frame_number, timestamp_seconds):
        return [Detection(camera_id, frame_number, timestamp_seconds, 0, "person", .9, 5, 5, 25, 30)]


def make_video(path: Path, count=3):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (48, 36))
    for _ in range(count): writer.write(np.zeros((36, 48, 3), dtype=np.uint8))
    writer.release()


def test_track_video_respects_limit_writes_csv_and_annotation(tmp_path):
    video = tmp_path / "cam01.mp4"; make_video(video)
    csv_path, annotated = tmp_path / "tracks.csv", tmp_path / "tracked.mp4"
    result = Tracker(engine=FakeEngine()).track_video(video, FakeDetector(), output_csv=csv_path,
        annotate=True, annotated_output_path=annotated, max_frames=2)
    assert result.status == STATUS_OK
    assert result.frames_processed == 2
    assert result.output_csv_path == str(csv_path) and csv_path.exists()
    assert result.annotated_video_path == str(annotated) and annotated.exists()


def test_invalid_video_is_reported_clearly(tmp_path):
    result = Tracker(engine=FakeEngine()).track_video(tmp_path / "missing.mp4", FakeDetector())
    assert result.status == STATUS_NOT_FOUND
