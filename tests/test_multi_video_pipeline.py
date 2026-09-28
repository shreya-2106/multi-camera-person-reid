"""Tests for the Phase 11 multi-video pipeline."""

from pathlib import Path

import pytest

from src.pipeline.multi_video_pipeline import (
    MultiVideoPipeline,
    MultiVideoPipelineError,
)
from src.pipeline.types import VideoInput


def create_video_file(
    directory: Path,
    filename: str,
) -> Path:
    """Create a temporary file with a supported video extension."""
    path = directory / filename
    path.write_bytes(b"test-video-placeholder")
    return path


def test_pipeline_requires_at_least_two_videos() -> None:
    pipeline = MultiVideoPipeline()

    video = VideoInput(
        video_path="video1.mp4",
        camera_id="CAM01",
    )

    with pytest.raises(
        MultiVideoPipelineError,
        match="At least 2 videos are required",
    ):
        pipeline.validate_inputs([video])


def test_pipeline_accepts_two_videos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video1 = create_video_file(tmp_path, "cam01.mp4")
    video2 = create_video_file(tmp_path, "cam02.mp4")

    def fake_validate_video(path: Path) -> dict:
        return {
            "filename": path.name,
            "status": "READABLE",
        }

    monkeypatch.setattr(
        "src.pipeline.multi_video_pipeline.validate_video",
        fake_validate_video,
    )

    pipeline = MultiVideoPipeline()

    inputs = [
        VideoInput(
            video_path=str(video1),
            camera_id="CAM01",
        ),
        VideoInput(
            video_path=str(video2),
            camera_id="CAM02",
        ),
    ]

    result = pipeline.prepare_analysis(inputs)

    assert result.status == "ready"
    assert len(result.videos) == 2
    assert result.unique_global_people == 0


def test_pipeline_accepts_three_videos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video1 = create_video_file(tmp_path, "cam01.mp4")
    video2 = create_video_file(tmp_path, "cam02.mp4")
    video3 = create_video_file(tmp_path, "cam03.mp4")

    def fake_validate_video(path: Path) -> dict:
        return {
            "filename": path.name,
            "status": "READABLE",
        }

    monkeypatch.setattr(
        "src.pipeline.multi_video_pipeline.validate_video",
        fake_validate_video,
    )

    pipeline = MultiVideoPipeline()

    inputs = [
        VideoInput(str(video1), "CAM01"),
        VideoInput(str(video2), "CAM02"),
        VideoInput(str(video3), "CAM03"),
    ]

    result = pipeline.prepare_analysis(inputs)

    assert result.status == "ready"
    assert len(result.videos) == 3


def test_pipeline_rejects_more_than_three_videos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    videos = [
        create_video_file(tmp_path, f"cam0{index}.mp4")
        for index in range(1, 5)
    ]

    def fake_validate_video(path: Path) -> dict:
        return {
            "filename": path.name,
            "status": "READABLE",
        }

    monkeypatch.setattr(
        "src.pipeline.multi_video_pipeline.validate_video",
        fake_validate_video,
    )

    pipeline = MultiVideoPipeline()

    inputs = [
        VideoInput(
            video_path=str(path),
            camera_id=f"CAM0{index}",
        )
        for index, path in enumerate(videos, start=1)
    ]

    with pytest.raises(
        MultiVideoPipelineError,
        match="At most 3 videos are supported",
    ):
        pipeline.validate_inputs(inputs)


def test_pipeline_rejects_duplicate_camera_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video1 = create_video_file(tmp_path, "cam01.mp4")
    video2 = create_video_file(tmp_path, "cam02.mp4")

    def fake_validate_video(path: Path) -> dict:
        return {
            "filename": path.name,
            "status": "READABLE",
        }

    monkeypatch.setattr(
        "src.pipeline.multi_video_pipeline.validate_video",
        fake_validate_video,
    )

    pipeline = MultiVideoPipeline()

    inputs = [
        VideoInput(str(video1), "CAM01"),
        VideoInput(str(video2), "CAM01"),
    ]

    with pytest.raises(
        MultiVideoPipelineError,
        match="Duplicate camera_id",
    ):
        pipeline.validate_inputs(inputs)


def test_pipeline_rejects_missing_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video1 = create_video_file(tmp_path, "cam01.mp4")
    missing_video = tmp_path / "cam02.mp4"

    def fake_validate_video(path: Path) -> dict:
        return {
            "filename": path.name,
            "status": "READABLE",
        }

    monkeypatch.setattr(
        "src.pipeline.multi_video_pipeline.validate_video",
        fake_validate_video,
    )

    pipeline = MultiVideoPipeline()

    inputs = [
        VideoInput(str(video1), "CAM01"),
        VideoInput(str(missing_video), "CAM02"),
    ]

    with pytest.raises(
        MultiVideoPipelineError,
        match="Video file does not exist",
    ):
        pipeline.validate_inputs(inputs)


def test_pipeline_rejects_unreadable_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video1 = create_video_file(tmp_path, "cam01.mp4")
    video2 = create_video_file(tmp_path, "cam02.mp4")

    def fake_validate_video(path: Path) -> dict:
        if path.name == "cam02.mp4":
            return {
                "filename": path.name,
                "status": "UNREADABLE",
            }

        return {
            "filename": path.name,
            "status": "READABLE",
        }

    monkeypatch.setattr(
        "src.pipeline.multi_video_pipeline.validate_video",
        fake_validate_video,
    )

    pipeline = MultiVideoPipeline()

    inputs = [
        VideoInput(str(video1), "CAM01"),
        VideoInput(str(video2), "CAM02"),
    ]

    with pytest.raises(
        MultiVideoPipelineError,
        match="Video is unreadable",
    ):
        pipeline.validate_inputs(inputs)


def test_pipeline_rejects_unsupported_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video1 = create_video_file(tmp_path, "cam01.mp4")
    invalid_video = tmp_path / "cam02.txt"
    invalid_video.write_text("not a video", encoding="utf-8")

    def fake_validate_video(path: Path) -> dict:
        return {
            "filename": path.name,
            "status": "READABLE",
        }

    monkeypatch.setattr(
        "src.pipeline.multi_video_pipeline.validate_video",
        fake_validate_video,
    )

    pipeline = MultiVideoPipeline()

    inputs = [
        VideoInput(str(video1), "CAM01"),
        VideoInput(str(invalid_video), "CAM02"),
    ]

    with pytest.raises(
        MultiVideoPipelineError,
        match="Unsupported video format",
    ):
        pipeline.validate_inputs(inputs)


def test_pipeline_rejects_none_inputs() -> None:
    pipeline = MultiVideoPipeline()

    with pytest.raises(
        MultiVideoPipelineError,
        match="Video inputs cannot be None",
    ):
        pipeline.validate_inputs(None)


def test_pipeline_result_contains_video_information(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video1 = create_video_file(tmp_path, "cam01.mp4")
    video2 = create_video_file(tmp_path, "cam02.mp4")

    def fake_validate_video(path: Path) -> dict:
        return {
            "filename": path.name,
            "status": "READABLE",
        }

    monkeypatch.setattr(
        "src.pipeline.multi_video_pipeline.validate_video",
        fake_validate_video,
    )

    pipeline = MultiVideoPipeline()

    inputs = [
        VideoInput(str(video1), "CAM01"),
        VideoInput(str(video2), "CAM02"),
    ]

    result = pipeline.prepare_analysis(inputs)

    assert result.videos[0].camera_id == "CAM01"
    assert result.videos[1].camera_id == "CAM02"
    assert result.videos[0].status == "ready"
    assert result.videos[1].status == "ready"