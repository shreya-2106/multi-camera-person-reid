"""Tests for Phase 2 local data validation."""
import json
from pathlib import Path

import cv2
import numpy as np

from src.data.dataset_validator import find_images, validate_datasets, validate_market1501, validate_msmt17
from src.data.validate import write_report
from src.data.video_validator import validate_videos


def make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.zeros((8, 12, 3), dtype=np.uint8))


def test_missing_datasets_are_not_found(tmp_path: Path) -> None:
    report = validate_datasets(tmp_path / "missing")
    assert report["datasets"]["market1501"]["status"] == "NOT_FOUND"
    assert report["datasets"]["msmt17"]["status"] == "NOT_FOUND"


def test_market_images_and_corruption(tmp_path: Path) -> None:
    directory = tmp_path / "market1501" / "bounding_box_train"
    make_image(directory / "0001_c1.jpg")
    (directory / "0002_c1.PNG").write_bytes(b"invalid image")
    assert len(find_images(directory)) == 2
    report = validate_market1501(tmp_path / "market1501")
    assert report["train_images"] == 2
    assert report["identities"] == 2
    assert report["corrupt_images"] == 1
    assert report["image_dimensions"] == [{"width": 12, "height": 8}]


def test_market_empty_split_structure(tmp_path: Path) -> None:
    root = tmp_path / "market1501"
    for name in ("bounding_box_train", "bounding_box_test", "query", "gt_bbox"):
        (root / name).mkdir(parents=True)
    report = validate_market1501(root)
    assert report["status"] == "FOUND"
    assert report["train_images"] == 0
    assert report["missing_directories"] == []


def test_msmt_verified_lists(tmp_path: Path) -> None:
    root = tmp_path / "msmt17"
    make_image(root / "train" / "0000_00.jpg")
    make_image(root / "test" / "0001_00.jpg")
    (root / "list_train.txt").write_text("train/0000_00.jpg 0\n", encoding="utf-8")
    (root / "list_query.txt").write_text("test/0001_00.jpg 1\n", encoding="utf-8")
    report = validate_msmt17(root)
    assert report["train_images"] == 1
    assert report["query_images"] == 1
    assert report["gallery_images"] == 1


def test_videos_missing_and_invalid(tmp_path: Path) -> None:
    assert validate_videos(tmp_path / "missing")["status"] == "NOT_FOUND"
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "cam02.mp4").write_bytes(b"not a video")
    report = validate_videos(videos)
    assert report["videos"][0]["camera_id"] == "CAM02"
    assert report["videos"][0]["status"] == "UNREADABLE"


def test_video_metadata_and_report_output(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    writer = cv2.VideoWriter(str(videos / "cam01.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (16, 10))
    assert writer.isOpened()
    writer.write(np.zeros((10, 16, 3), dtype=np.uint8))
    writer.release()
    record = validate_videos(videos)["videos"][0]
    assert record["status"] == "READABLE"
    assert record["duration_seconds"] == 0.1
    destination = tmp_path / "outputs" / "report.json"
    write_report(destination, {"path": "data/reid"})
    assert json.loads(destination.read_text(encoding="utf-8"))["path"] == "data/reid"
