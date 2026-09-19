import pytest

from src.config import load_config
from src.tracking.tracker import Tracker, TrackingError, _validate_tracking_config


def test_real_config_enables_bytetrack_with_supported_settings():
    tracking = load_config(apply_env=False).tracking
    assert tracking.enabled is True
    assert tracking.tracker == "bytetrack"
    assert _validate_tracking_config(dict(tracking))["track_buffer"] == 30


@pytest.mark.parametrize("config", [
    {"tracker": "botsort"}, {"track_high_thresh": 1.1},
    {"track_low_thresh": 0.7, "track_high_thresh": 0.5}, {"track_buffer": 0},
])
def test_invalid_tracking_configuration_is_rejected(config):
    with pytest.raises(TrackingError):
        _validate_tracking_config(config)


def test_tracker_initializes_without_loading_yolo():
    tracker = Tracker({"tracker": "bytetrack"})
    assert tracker.config["tracker"] == "bytetrack"
