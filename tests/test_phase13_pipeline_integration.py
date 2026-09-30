"""Controlled Phase 13 pipeline integration test."""

from pathlib import Path

import numpy as np

from src.matching.types import TrackReference
from src.pipeline.multi_video_pipeline import MultiVideoPipeline
from src.pipeline.types import VideoAnalysisResult, VideoInput


def write_tracks_csv(
    path: Path,
    camera_id: str,
    track_id: int,
    start_frame: int,
    end_frame: int,
    start_time: float,
    end_time: float,
) -> None:
    """Create a minimal valid tracking CSV for one track."""

    lines = [
        "camera_id,frame_number,timestamp_seconds,track_id,x1,y1,x2,y2"
    ]

    frame_count = end_frame - start_frame + 1

    for index, frame_number in enumerate(
        range(start_frame, end_frame + 1)
    ):
        if frame_count == 1:
            timestamp = start_time
        else:
            timestamp = start_time + (
                (end_time - start_time)
                * index
                / (frame_count - 1)
            )

        lines.append(
            f"{camera_id},"
            f"{frame_number},"
            f"{timestamp:.6f},"
            f"{track_id},"
            "100,100,180,280"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_embeddings(
    path: Path,
    embedding: np.ndarray,
) -> None:
    """Create one track-level embedding file."""

    path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = np.stack(
        [embedding, embedding, embedding],
        axis=0,
    )

    np.save(path, embeddings)


def test_phase13_pipeline_assigns_same_global_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify the complete Phase 13 cross-camera integration."""

    # --------------------------------------------------------------
    # Keep all generated outputs isolated inside pytest's temp folder.
    # --------------------------------------------------------------

    monkeypatch.chdir(tmp_path)

    # --------------------------------------------------------------
    # Create synthetic CAM01 tracking data.
    # CAM01 Track 1 ends at 5 seconds.
    # --------------------------------------------------------------

    write_tracks_csv(
        tmp_path
        / "outputs"
        / "tracking"
        / "CAM01"
        / "tracks.csv",
        camera_id="CAM01",
        track_id=1,
        start_frame=0,
        end_frame=4,
        start_time=0.0,
        end_time=5.0,
    )

    # --------------------------------------------------------------
    # Create synthetic CAM02 tracking data.
    # CAM02 Track 7 starts at 10 seconds.
    #
    # Travel time = 10 - 5 = 5 seconds.
    # Existing topology allows 2-30 seconds.
    # --------------------------------------------------------------

    write_tracks_csv(
        tmp_path
        / "outputs"
        / "tracking"
        / "CAM02"
        / "tracks.csv",
        camera_id="CAM02",
        track_id=7,
        start_frame=10,
        end_frame=14,
        start_time=10.0,
        end_time=15.0,
    )

    # --------------------------------------------------------------
    # Create identical synthetic Re-ID embeddings.
    # This should produce cosine similarity ~= 1.0.
    # --------------------------------------------------------------

    embedding = np.zeros(
        512,
        dtype=np.float32,
    )

    embedding[0] = 1.0

    write_embeddings(
        tmp_path
        / "outputs"
        / "embeddings"
        / "CAM01"
        / "track_001"
        / "embeddings.npy",
        embedding,
    )

    write_embeddings(
        tmp_path
        / "outputs"
        / "embeddings"
        / "CAM02"
        / "track_007"
        / "embeddings.npy",
        embedding,
    )

    # --------------------------------------------------------------
    # Create the pipeline using the real project configuration.
    # --------------------------------------------------------------

    repo_root = Path(__file__).resolve().parents[1]

    pipeline = MultiVideoPipeline(
        config_path=repo_root / "configs" / "config.yaml",
        reid_config_path=repo_root / "configs" / "reid.yaml",
    )

    videos = (
        VideoInput(
            video_path="cam01.mp4",
            camera_id="CAM01",
        ),
        VideoInput(
            video_path="cam02.mp4",
            camera_id="CAM02",
        ),
    )

    video_results = (
        VideoAnalysisResult(
            camera_id="CAM01",
            video_path="cam01.mp4",
            status="completed",
        ),
        VideoAnalysisResult(
            camera_id="CAM02",
            video_path="cam02.mp4",
            status="completed",
        ),
    )

    # --------------------------------------------------------------
    # Run only the Phase 13 integration layer.
    # --------------------------------------------------------------

    results = pipeline._run_cross_camera_matching(
        videos=videos,
        video_results=video_results,
    )

    # --------------------------------------------------------------
    # We expect exactly two track-level results:
    #
    # CAM01 Track 1 -> new global identity
    # CAM02 Track 7 -> same global identity
    # --------------------------------------------------------------

    assert len(results) == 2

    cam01_result = next(
        result
        for result in results
        if result.camera_id == "CAM01"
    )

    cam02_result = next(
        result
        for result in results
        if result.camera_id == "CAM02"
    )

    assert cam01_result.track_id == 1
    assert cam02_result.track_id == 7

    assert (
        cam01_result.global_person_id
        == cam02_result.global_person_id
    )

    # The cross-camera match should be accepted.
    assert cam02_result.decision == "accept"

    # Similarity should be essentially perfect.
    assert (
        cam02_result.evidence["mean_similarity"]
        > 0.99
    )

    assert (
        cam02_result.evidence["max_similarity"]
        > 0.99
    )

    # Travel time should be 5 seconds.
    assert (
        cam02_result.evidence["travel_time_seconds"]
        == 5.0
    )

    # Both topology and temporal constraints should be valid.
    assert (
        cam02_result.evidence["topology_valid"]
        is True
    )

    assert (
        cam02_result.evidence["temporal_valid"]
        is True
    )
