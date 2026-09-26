import pytest

from src.temporal.track_summary import (
    TrackSummaryBuilder,
    TrackSummaryError,
)


def test_build_real_tracking_summaries():
    builder = TrackSummaryBuilder(
        "outputs/tracking/CAM01_tracks.csv"
    )

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


def test_track_summary_duration_and_displacement():
    builder = TrackSummaryBuilder(
        "outputs/tracking/CAM01_tracks.csv"
    )

    summaries = builder.build()

    track_1 = summaries[0]

    assert track_1.duration_seconds == pytest.approx(
        0.6339666667
    )

    assert track_1.displacement_pixels > 0.0


def test_track_summary_to_dict():
    builder = TrackSummaryBuilder(
        "outputs/tracking/CAM01_tracks.csv"
    )

    summary = builder.build()[0]
    data = summary.to_dict()

    assert data["camera_id"] == "CAM01"
    assert data["track_id"] == 1
    assert "duration_seconds" in data
    assert "displacement_pixels" in data