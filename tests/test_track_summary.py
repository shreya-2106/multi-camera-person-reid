import csv

import pytest

from src.temporal.track_summary import (
    TrackSummaryBuilder,
    TrackSummaryError,
)


def create_tracking_fixture(tmp_path):
    csv_path = tmp_path / "CAM01_tracks.csv"

    rows = []

    # Track 1: frames 0-19
    # 20 frames, timestamps 0.0 to 0.6339666667
    for frame in range(20):
        timestamp = frame * (0.6339666667 / 19)

        rows.append(
            {
                "camera_id": "CAM01",
                "frame_number": frame,
                "timestamp_seconds": timestamp,
                "track_id": 1,
                "x1": 10.0 + frame,
                "y1": 20.0,
                "x2": 30.0 + frame,
                "y2": 60.0,
            }
        )

    # Track 2: frames 16-19
    # 4 frames, timestamps 0.5338666667 to 0.6339666667
    for frame in range(16, 20):
        timestamp = 0.5338666667 + (
            (frame - 16) * (0.1001 / 3)
        )

        rows.append(
            {
                "camera_id": "CAM01",
                "frame_number": frame,
                "timestamp_seconds": timestamp,
                "track_id": 2,
                "x1": 50.0 + frame,
                "y1": 30.0,
                "x2": 70.0 + frame,
                "y2": 70.0,
            }
        )

    with csv_path.open(
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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def test_build_real_tracking_summaries(tmp_path):
    csv_path = create_tracking_fixture(tmp_path)

    builder = TrackSummaryBuilder(csv_path)

    summaries = builder.build()

    assert len(summaries) == 2

    track_1 = summaries[0]
    track_2 = summaries[1]

    assert track_1.camera_id == "CAM01"
    assert track_1.track_id == 1
    assert track_1.start_frame == 0
    assert track_1.end_frame == 19
    assert track_1.frames_seen == 20
    assert track_1.start_time == pytest.approx(0.0)
    assert track_1.end_time == pytest.approx(0.6339666667)

    assert track_2.camera_id == "CAM01"
    assert track_2.track_id == 2
    assert track_2.start_frame == 16
    assert track_2.end_frame == 19
    assert track_2.frames_seen == 4
    assert track_2.start_time == pytest.approx(0.5338666667)
    assert track_2.end_time == pytest.approx(0.6339666667)


def test_missing_tracking_file():
    builder = TrackSummaryBuilder(
        "outputs/tracking/does_not_exist.csv"
    )

    with pytest.raises(TrackSummaryError):
        builder.build()


def test_invalid_csv_header(tmp_path):
    csv_path = tmp_path / "invalid.csv"

    csv_path.write_text(
        "camera_id,track_id\n"
        "CAM01,1\n",
        encoding="utf-8",
    )

    builder = TrackSummaryBuilder(csv_path)

    with pytest.raises(TrackSummaryError):
        builder.build()


def test_empty_csv(tmp_path):
    csv_path = tmp_path / "empty.csv"

    csv_path.write_text(
        "camera_id,frame_number,timestamp_seconds,track_id,"
        "x1,y1,x2,y2\n",
        encoding="utf-8",
    )

    builder = TrackSummaryBuilder(csv_path)

    with pytest.raises(TrackSummaryError):
        builder.build()


def test_track_summary_duration_and_displacement(tmp_path):
    csv_path = create_tracking_fixture(tmp_path)

    builder = TrackSummaryBuilder(csv_path)

    summaries = builder.build()

    track_1 = summaries[0]

    assert track_1.duration_seconds == pytest.approx(
        0.6339666667
    )

    assert track_1.displacement_pixels > 0.0


def test_track_summary_to_dict(tmp_path):
    csv_path = create_tracking_fixture(tmp_path)

    builder = TrackSummaryBuilder(csv_path)

    summary = builder.build()[0]
    data = summary.to_dict()

    assert data["camera_id"] == "CAM01"
    assert data["track_id"] == 1
    assert "duration_seconds" in data
    assert "displacement_pixels" in data