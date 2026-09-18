"""
Unit tests for src.detection.detector.Detector.

None of these tests download real YOLO weights or run real inference:
Detector._create_backend_model is monkeypatched with a small fake model
object that mimics just enough of the Ultralytics API (`.names`,
`.predict(...)`) for the parsing/filtering/IO logic to be exercised
deterministically and offline.
"""

from pathlib import Path

import numpy as np
import pytest

from src.detection.detector import (
    Detector,
    DetectorError,
    STATUS_ERROR,
    STATUS_NOT_FOUND,
    STATUS_OK,
    STATUS_UNREADABLE,
    resolve_device,
    resolve_model_path,
)

cv2 = pytest.importorskip("cv2")


# ---------------------------------------------------------------------------
# Fake Ultralytics-like backend
# ---------------------------------------------------------------------------


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYOLOModel:
    """
    Minimal stand-in for an ultralytics.YOLO instance.

    `names` mimics the COCO class map (truncated). `predict()` returns a
    single FakeResult with one "person" box and one "bicycle" box unless a
    `classes` filter is passed, in which case only matching boxes are kept
    -- mirroring how the real ultralytics `classes=` kwarg behaves.
    """

    names = {0: "person", 1: "bicycle", 2: "car"}

    def __init__(self):
        self.predict_calls = []

    def predict(self, source, conf, iou, imgsz, device, classes, verbose):
        self.predict_calls.append(
            dict(conf=conf, iou=iou, imgsz=imgsz, device=device, classes=classes)
        )
        all_boxes = [
            # (xyxy, conf, cls)
            ([10.0, 20.0, 50.0, 100.0], 0.91, 0),  # person
            ([200.0, 200.0, 260.0, 260.0], 0.75, 1),  # bicycle
        ]
        if classes is not None:
            all_boxes = [b for b in all_boxes if b[2] in classes]

        if not all_boxes:
            xyxy = np.zeros((0, 4))
            confs = np.zeros((0,))
            clss = np.zeros((0,))
        else:
            xyxy = np.array([b[0] for b in all_boxes])
            confs = np.array([b[1] for b in all_boxes])
            clss = np.array([b[2] for b in all_boxes])

        return [FakeResult(FakeBoxes(xyxy, confs, clss))]


def make_fake_detector(monkeypatch, **kwargs) -> Detector:
    detector = Detector(device="cpu", **kwargs)
    fake_model = FakeYOLOModel()
    monkeypatch.setattr(Detector, "_create_backend_model", lambda self, model_name_or_path: fake_model)
    return detector


def make_valid_video(path: Path, width=64, height=48, fps=10.0, num_frames=5) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype="uint8")
        frame[:, :, 0] = i * 10
        writer.write(frame)
    writer.release()


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------


def test_resolve_device_cpu_is_always_cpu():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_returns_cpu_or_cuda():
    assert resolve_device("auto") in ("cpu", "cuda")


def test_resolve_device_cuda_falls_back_to_cpu_when_unavailable(monkeypatch):
    import src.env_check as env_check

    fake_info = env_check.DeviceInfo(torch_available=True, device="cpu", cuda_available=False)
    monkeypatch.setattr(env_check, "detect_device", lambda prefer_cuda=True: fake_info)
    assert resolve_device("cuda") == "cpu"


# ---------------------------------------------------------------------------
# resolve_model_path
# ---------------------------------------------------------------------------


def test_resolve_model_path_bare_name_is_untouched():
    assert resolve_model_path("yolov8n.pt") == "yolov8n.pt"


def test_resolve_model_path_relative_path_resolved_to_project_root():
    from src.config import PROJECT_ROOT

    resolved = resolve_model_path("models/yolov8n.pt")
    assert resolved == str(PROJECT_ROOT / "models" / "yolov8n.pt")


def test_resolve_model_path_absolute_path_is_untouched(tmp_path):
    abs_path = str(tmp_path / "custom_weights.pt")
    assert resolve_model_path(abs_path) == abs_path


# ---------------------------------------------------------------------------
# Model loading / class resolution
# ---------------------------------------------------------------------------


def test_load_model_resolves_person_class_id(monkeypatch):
    detector = make_fake_detector(monkeypatch)
    detector.load_model()
    assert detector._class_id_filter == [0]  # only "person" -> id 0


def test_load_model_is_idempotent(monkeypatch):
    detector = make_fake_detector(monkeypatch)
    detector.load_model()
    model_ref = detector._model
    detector.load_model()
    assert detector._model is model_ref  # not reloaded


def test_load_model_raises_for_unknown_class(monkeypatch):
    detector = make_fake_detector(monkeypatch, classes=["airplane"])
    with pytest.raises(DetectorError):
        detector.load_model()


def test_close_clears_loaded_model(monkeypatch):
    detector = make_fake_detector(monkeypatch)
    detector.load_model()
    detector.close()
    assert detector._model is None
    assert detector._class_id_filter is None


def test_create_backend_model_missing_ultralytics_raises_detector_error(monkeypatch):
    detector = Detector(device="cpu")
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ultralytics":
            raise ImportError("No module named 'ultralytics'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(DetectorError):
        detector.load_model()


# ---------------------------------------------------------------------------
# Frame-level detection
# ---------------------------------------------------------------------------


def test_detect_frame_filters_to_person_only(monkeypatch):
    detector = make_fake_detector(monkeypatch)
    frame = np.zeros((100, 100, 3), dtype="uint8")

    detections = detector.detect_frame(frame, camera_id="CAM01", frame_number=3, timestamp_seconds=0.3)

    assert len(detections) == 1
    det = detections[0]
    assert det.class_name == "person"
    assert det.class_id == 0
    assert det.camera_id == "CAM01"
    assert det.frame_number == 3
    assert det.timestamp_seconds == 0.3
    assert det.confidence == pytest.approx(0.91)
    assert (det.x1, det.y1, det.x2, det.y2) == (10.0, 20.0, 50.0, 100.0)


def test_detect_frame_passes_configured_thresholds_to_model(monkeypatch):
    detector = make_fake_detector(monkeypatch, confidence=0.7, iou=0.3, imgsz=320)
    frame = np.zeros((50, 50, 3), dtype="uint8")
    detector.detect_frame(frame)
    call = detector._model.predict_calls[-1]
    assert call["conf"] == 0.7
    assert call["iou"] == 0.3
    assert call["imgsz"] == 320
    assert call["classes"] == [0]


def test_detect_frame_empty_results_returns_empty_list(monkeypatch):
    detector = make_fake_detector(monkeypatch, classes=["car"])  # matches id 2, never returned as person
    frame = np.zeros((50, 50, 3), dtype="uint8")
    detections = detector.detect_frame(frame)
    assert detections == []


# ---------------------------------------------------------------------------
# Video-level detection
# ---------------------------------------------------------------------------


def test_detect_video_missing_file_reports_not_found(monkeypatch, tmp_path):
    detector = make_fake_detector(monkeypatch)
    result = detector.detect_video(tmp_path / "missing.mp4")
    assert result.status == STATUS_NOT_FOUND
    assert result.camera_id is None  # "missing" contains no recognizable camera token


def test_detect_video_unreadable_file_reports_unreadable(monkeypatch, tmp_path):
    detector = make_fake_detector(monkeypatch)
    fake_video = tmp_path / "cam01.mp4"
    fake_video.write_bytes(b"not a real video")
    result = detector.detect_video(fake_video)
    assert result.status == STATUS_UNREADABLE
    assert result.camera_id == "CAM01"


def test_detect_video_processes_frames_and_infers_camera_id(monkeypatch, tmp_path):
    detector = make_fake_detector(monkeypatch)
    video_path = tmp_path / "cam02.mp4"
    make_valid_video(video_path, width=64, height=48, fps=10.0, num_frames=4)

    result = detector.detect_video(video_path)

    assert result.status == STATUS_OK
    assert result.camera_id == "CAM02"
    assert result.frames_processed == 4
    assert result.detection_count == 4  # 1 person detection per frame
    assert result.fps == pytest.approx(10.0, rel=0.5)
    # timestamps should increase with frame number
    timestamps = [d.timestamp_seconds for d in result.detections]
    assert timestamps == sorted(timestamps)


def test_detect_video_respects_max_frames(monkeypatch, tmp_path):
    detector = make_fake_detector(monkeypatch)
    video_path = tmp_path / "cam03.mp4"
    make_valid_video(video_path, num_frames=10)

    result = detector.detect_video(video_path, max_frames=3)

    assert result.frames_processed == 3


def test_detect_video_writes_csv_output(monkeypatch, tmp_path):
    detector = make_fake_detector(monkeypatch)
    video_path = tmp_path / "cam04.mp4"
    make_valid_video(video_path, num_frames=2)
    csv_path = tmp_path / "out" / "cam04_detections.csv"

    result = detector.detect_video(video_path, output_csv=csv_path)

    assert result.output_csv_path == str(csv_path)
    assert csv_path.exists()
    content = csv_path.read_text()
    assert "camera_id" in content.splitlines()[0]
    assert "CAM04" in content


def test_detect_video_annotate_writes_video_file(monkeypatch, tmp_path):
    detector = make_fake_detector(monkeypatch)
    video_path = tmp_path / "cam05.mp4"
    make_valid_video(video_path, width=64, height=48, num_frames=3)
    annotated_path = tmp_path / "annotated" / "cam05_annotated.mp4"

    result = detector.detect_video(video_path, annotate=True, annotated_output_path=annotated_path)

    assert result.annotated_video_path == str(annotated_path)
    assert annotated_path.exists()
    assert annotated_path.stat().st_size > 0


def test_detect_video_model_load_failure_reports_error(monkeypatch, tmp_path):
    detector = Detector(device="cpu")

    def raise_error(self, model_name_or_path):
        raise DetectorError("simulated model load failure")

    monkeypatch.setattr(Detector, "_create_backend_model", raise_error)

    video_path = tmp_path / "cam06.mp4"
    make_valid_video(video_path, num_frames=1)

    result = detector.detect_video(video_path)

    assert result.status == STATUS_ERROR
    assert "simulated model load failure" in result.error


def test_detect_video_does_not_load_model_for_missing_file(monkeypatch, tmp_path):
    detector = Detector(device="cpu")
    load_calls = []

    def track_load(self, model_name_or_path):
        load_calls.append(model_name_or_path)
        return FakeYOLOModel()

    monkeypatch.setattr(Detector, "_create_backend_model", track_load)
    detector.detect_video(tmp_path / "missing.mp4")
    assert load_calls == []  # model must not be loaded for a file that doesn't exist


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------


def test_from_config_reads_detection_section():
    from src.config import Config

    cfg = Config(
        {
            "detection": {
                "model": "yolov8s.pt",
                "confidence": 0.6,
                "iou": 0.4,
                "device": "cpu",
                "imgsz": 320,
                "classes": ["person"],
            }
        }
    )
    detector = Detector.from_config(cfg)
    assert detector.model_name_or_path == "yolov8s.pt"
    assert detector.confidence == 0.6
    assert detector.iou == 0.4
    assert detector.imgsz == 320
    assert detector.device == "cpu"


def test_from_config_uses_defaults_when_section_missing():
    from src.config import Config

    cfg = Config({})
    detector = Detector.from_config(cfg)
    assert detector.model_name_or_path == "yolov8n.pt"
    assert detector.confidence == 0.5
