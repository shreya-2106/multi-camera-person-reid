from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.pipeline.multi_video_pipeline import MultiVideoPipeline
from src.pipeline.types import (
    VideoAnalysisResult,
    VideoInput,
)


def write_tracks_csv(
    path: Path,
    *,
    camera_id: str,
    track_id: int,
    start_frame: int,
    end_frame: int,
    start_time: float,
    end_time: float,
) -> None:
    """Create a deterministic synthetic tracking CSV for the test."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_count = end_frame - start_frame + 1

    if frame_count <= 0:
        raise ValueError(
            "end_frame must be greater than or equal to start_frame"
        )

    if frame_count == 1:
        timestamps = [start_time]
    else:
        timestamps = np.linspace(
            start_time,
            end_time,
            frame_count,
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "camera_id",
                "frame_number",
                "timestamp_seconds",
                "track_id",
                "x1",
                "y1",
                "x2",
                "y2",
                "confidence",
                "class_id",
                "class_name",
            ],
        )

        writer.writeheader()

        for index, frame_number in enumerate(
            range(start_frame, end_frame + 1)
        ):
            writer.writerow(
                {
                    "camera_id": camera_id,
                    "frame_number": frame_number,
                    "timestamp_seconds": float(
                        timestamps[index]
                    ),
                    "track_id": track_id,
                    "x1": 100.0,
                    "y1": 50.0,
                    "x2": 180.0,
                    "y2": 200.0,
                    "confidence": 0.95,
                    "class_id": 0,
                    "class_name": "person",
                }
            )


def write_embeddings(
    path: Path,
    embedding: np.ndarray,
) -> None:
    """Write one synthetic Re-ID embedding."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        path,
        embedding.astype(np.float32),
    )


@pytest.mark.parametrize(
    ("offsets", "expected_accept"),
    [((0.0, 0.0), True), ((None, None), False)],
)
def test_phase13_pipeline_assigns_same_global_identity(
    tmp_path: Path,
    monkeypatch,
    offsets: tuple[float | None, float | None],
    expected_accept: bool,
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
    #
    # CAM02 Track 7 starts at 10 seconds.
    #
    # Travel time:
    #
    #     10 - 5 = 5 seconds
    #
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
    #
    # The matcher expects embeddings in the shape:
    #
    #     (number_of_embeddings, embedding_dimension)
    #
    # Therefore one 512-dimensional embedding must be:
    #
    #     (1, 512)
    #
    # rather than:
    #
    #     (512,)
    #
    # This should produce cosine similarity ~= 1.0.
    # --------------------------------------------------------------

    embedding = np.zeros(
        (1, 512),
        dtype=np.float32,
    )

    embedding[0, 0] = 1.0

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
        config_path=repo_root
        / "configs"
        / "config.yaml",
        reid_config_path=repo_root
        / "configs"
        / "reid.yaml",
    )

    videos = (
        VideoInput(
            video_path="cam01.mp4",
            camera_id="CAM01",
            timeline_offset_seconds=offsets[0],
        ),
        VideoInput(
            video_path="cam02.mp4",
            camera_id="CAM02",
            timeline_offset_seconds=offsets[1],
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
    # Run the cross-camera integration layer.
    #
    # Phase 15 returns:
    #
    #     (
    #         global_identity_results,
    #         candidate_reports,
    #     )
    #
    # --------------------------------------------------------------

    results, candidate_reports = (
        pipeline._run_cross_camera_matching(
            videos=videos,
            video_results=video_results,
        )
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

    # --------------------------------------------------------------
    # CAM01 should create the initial global identity.
    # --------------------------------------------------------------

    assert cam01_result.track_id == 1
    assert cam01_result.global_person_id == 1
    assert cam01_result.decision == "new_identity"

    # --------------------------------------------------------------
    # CAM02 should be matched to the existing CAM01 identity.
    # --------------------------------------------------------------

    assert cam02_result.track_id == 7
    if expected_accept:
        assert cam02_result.global_person_id == 1
        assert cam02_result.decision == "accept"
    else:
        assert cam02_result.global_person_id != 1
        assert cam02_result.decision == "new_identity"
        assert any(report["status"] == "timing_unverified" for report in candidate_reports)
        unresolved = next(
            report for report in candidate_reports
            if report["status"] == "timing_unverified"
        )
        assert unresolved["final_decision"] == "unresolved"
        assert unresolved["appearance"]["mean_similarity"] > 0.99
        return

    # --------------------------------------------------------------
    # Verify the matching evidence.
    # --------------------------------------------------------------

    assert cam02_result.score
    assert cam02_result.evidence

    assert (
        cam02_result.evidence["mean_similarity"]
        > 0.99
    )

    assert (
        cam02_result.evidence["travel_time_seconds"]
        == 5.0
    )

    assert (
        cam02_result.evidence["topology_valid"]
        is True
    )

    assert (
        cam02_result.evidence["temporal_valid"]
        is True
    )

    # --------------------------------------------------------------
    # Phase 15 candidate-report collection should also contain
    # the accepted cross-camera candidate.
    # --------------------------------------------------------------

    assert candidate_reports

    accepted_reports = [
        report
        for report in candidate_reports
        if (
            report["from_camera"] == "CAM01"
            and report["from_track_id"] == 1
            and report["to_camera"] == "CAM02"
            and report["to_track_id"] == 7
        )
    ]

    assert accepted_reports

    accepted_report = accepted_reports[0]

    assert accepted_report["status"] == "valid"

    assert (
        accepted_report["travel_time_seconds"]
        == 5.0
    )

    assert accepted_report["transition"] is not None
    assert accepted_report["evidence"] is not None
    assert accepted_report["score"] is not None

    assert (
        accepted_report["evidence"]["mean_similarity"]
        > 0.99
    )
