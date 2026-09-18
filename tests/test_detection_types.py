"""Unit tests for src.detection.types."""

import csv
from pathlib import Path

import pytest

from src.detection.types import CSV_FIELDNAMES, Detection, DetectionVideoResult, write_detections_csv


def make_detection(**overrides) -> Detection:
    defaults = dict(
        camera_id="CAM01",
        frame_number=0,
        timestamp_seconds=0.0,
        class_id=0,
        class_name="person",
        confidence=0.9,
        x1=10.0,
        y1=20.0,
        x2=50.0,
        y2=100.0,
    )
    defaults.update(overrides)
    return Detection(**defaults)


def test_detection_valid_construction():
    det = make_detection()
    assert det.camera_id == "CAM01"
    assert det.width == 40.0
    assert det.height == 80.0


def test_detection_rejects_confidence_out_of_range():
    with pytest.raises(ValueError):
        make_detection(confidence=1.5)
    with pytest.raises(ValueError):
        make_detection(confidence=-0.1)


def test_detection_rejects_invalid_box_x():
    with pytest.raises(ValueError):
        make_detection(x1=50.0, x2=10.0)


def test_detection_rejects_invalid_box_y():
    with pytest.raises(ValueError):
        make_detection(y1=100.0, y2=20.0)


def test_detection_allows_zero_area_box():
    # A degenerate (point) box is not a coordinate-ordering error.
    det = make_detection(x1=10.0, x2=10.0, y1=20.0, y2=20.0)
    assert det.width == 0.0
    assert det.height == 0.0


def test_detection_to_dict_has_expected_keys():
    det = make_detection()
    d = det.to_dict()
    assert set(CSV_FIELDNAMES).issubset(d.keys())


def test_detection_video_result_defaults():
    result = DetectionVideoResult(video_path="data/videos/cam01.mp4", camera_id="CAM01", status="OK")
    assert result.detections == []
    assert result.frames_processed == 0
    assert result.detection_count == 0


def test_write_detections_csv_creates_file_with_header_and_rows(tmp_path):
    detections = [make_detection(frame_number=i) for i in range(3)]
    out_path = tmp_path / "nested" / "cam01_detections.csv"

    result_path = write_detections_csv(detections, out_path)

    assert result_path == out_path
    assert out_path.exists()

    with open(out_path, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_FIELDNAMES
        rows = list(reader)
    assert len(rows) == 3
    assert rows[0]["camera_id"] == "CAM01"
    assert rows[1]["frame_number"] == "1"


def test_write_detections_csv_empty_list_still_writes_header(tmp_path):
    out_path = tmp_path / "empty.csv"
    write_detections_csv([], out_path)
    with open(out_path, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_FIELDNAMES
        assert list(reader) == []
