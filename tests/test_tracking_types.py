import csv
import pytest

from src.tracking.types import TRACK_CSV_FIELDNAMES, TrackedPerson, write_tracks_csv


def make_track(**changes):
    values = dict(camera_id="CAM01", frame_number=2, timestamp_seconds=0.2, track_id=7,
                  x1=1.0, y1=2.0, x2=10.0, y2=20.0, confidence=0.9,
                  class_id=0, class_name="person")
    values.update(changes)
    return TrackedPerson(**values)


def test_tracked_person_has_required_local_tracking_fields():
    track = make_track()
    assert track.track_id == 7
    assert set(TRACK_CSV_FIELDNAMES) == set(track.to_dict())


def test_tracked_person_rejects_invalid_id_confidence_and_box():
    with pytest.raises(ValueError): make_track(track_id="7")
    with pytest.raises(ValueError): make_track(confidence=2)
    with pytest.raises(ValueError): make_track(x1=11)


def test_write_tracks_csv_writes_header_and_records(tmp_path):
    path = write_tracks_csv([make_track()], tmp_path / "nested" / "tracks.csv")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["camera_id"] == "CAM01"
    assert rows[0]["track_id"] == "7"


def test_write_tracks_csv_handles_no_tracks(tmp_path):
    path = write_tracks_csv([], tmp_path / "tracks.csv")
    with path.open(newline="") as handle:
        assert csv.DictReader(handle).fieldnames == TRACK_CSV_FIELDNAMES
