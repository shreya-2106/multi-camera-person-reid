import pytest

from src.temporal.topology import CameraTopology
from src.temporal.transition import (
    TransitionCandidateBuilder,
)
from src.temporal.types import TrackSummary


TOPOLOGY_CONFIG = "configs/topology.yaml"


def make_track(
    camera_id: str,
    track_id: int,
    start_time: float,
    end_time: float,
) -> TrackSummary:
    return TrackSummary(
        camera_id=camera_id,
        track_id=track_id,
        start_frame=0,
        end_frame=10,
        start_time=start_time,
        end_time=end_time,
        start_center_x=100.0,
        start_center_y=100.0,
        end_center_x=200.0,
        end_center_y=200.0,
        frames_seen=11,
    )


def test_valid_transition():
    topology = CameraTopology.from_yaml(
        TOPOLOGY_CONFIG
    )

    builder = TransitionCandidateBuilder(topology)

    cam01_track = make_track(
        "CAM01",
        1,
        0.0,
        10.0,
    )

    cam02_track = make_track(
        "CAM02",
        7,
        12.0,
        20.0,
    )

    candidate = builder.build_pair(
        cam01_track,
        cam02_track,
    )

    assert candidate is not None

    assert candidate.from_camera == "CAM01"
    assert candidate.from_track_id == 1

    assert candidate.to_camera == "CAM02"
    assert candidate.to_track_id == 7

    assert candidate.exit_time == pytest.approx(10.0)
    assert candidate.entry_time == pytest.approx(12.0)
    assert candidate.travel_time_seconds == pytest.approx(2.0)


def test_transition_too_fast():
    topology = CameraTopology.from_yaml(
        TOPOLOGY_CONFIG
    )

    builder = TransitionCandidateBuilder(topology)

    cam01_track = make_track(
        "CAM01",
        1,
        0.0,
        10.0,
    )

    cam02_track = make_track(
        "CAM02",
        7,
        11.0,
        20.0,
    )

    candidate = builder.build_pair(
        cam01_track,
        cam02_track,
    )

    assert candidate is None


def test_transition_too_slow():
    topology = CameraTopology.from_yaml(
        TOPOLOGY_CONFIG
    )

    builder = TransitionCandidateBuilder(topology)

    cam01_track = make_track(
        "CAM01",
        1,
        0.0,
        10.0,
    )

    cam02_track = make_track(
        "CAM02",
        7,
        41.0,
        50.0,
    )

    candidate = builder.build_pair(
        cam01_track,
        cam02_track,
    )

    assert candidate is None


def test_invalid_camera_connection():
    topology = CameraTopology.from_yaml(
        TOPOLOGY_CONFIG
    )

    builder = TransitionCandidateBuilder(topology)

    cam01_track = make_track(
        "CAM01",
        1,
        0.0,
        10.0,
    )

    cam04_track = make_track(
        "CAM04",
        7,
        12.0,
        20.0,
    )

    candidate = builder.build_pair(
        cam01_track,
        cam04_track,
    )

    assert candidate is None


def test_same_camera_is_not_transition():
    topology = CameraTopology.from_yaml(
        TOPOLOGY_CONFIG
    )

    builder = TransitionCandidateBuilder(topology)

    first_track = make_track(
        "CAM01",
        1,
        0.0,
        10.0,
    )

    second_track = make_track(
        "CAM01",
        2,
        12.0,
        20.0,
    )

    candidate = builder.build_pair(
        first_track,
        second_track,
    )

    assert candidate is None


def test_build_multiple_tracks():
    topology = CameraTopology.from_yaml(
        TOPOLOGY_CONFIG
    )

    builder = TransitionCandidateBuilder(topology)

    tracks = [
        make_track(
            "CAM01",
            1,
            0.0,
            10.0,
        ),
        make_track(
            "CAM02",
            2,
            12.0,
            20.0,
        ),
        make_track(
            "CAM03",
            3,
            13.0,
            21.0,
        ),
        make_track(
            "CAM04",
            4,
            20.0,
            30.0,
        ),
    ]

    candidates = builder.build(tracks)

    assert len(candidates) == 2

    pairs = {
        (
            candidate.from_camera,
            candidate.from_track_id,
            candidate.to_camera,
            candidate.to_track_id,
        )
        for candidate in candidates
    }

    assert (
        "CAM01",
        1,
        "CAM02",
        2,
    ) in pairs

    assert (
        "CAM01",
        1,
        "CAM03",
        3,
    ) in pairs


def test_candidate_to_dict():
    topology = CameraTopology.from_yaml(
        TOPOLOGY_CONFIG
    )

    builder = TransitionCandidateBuilder(topology)

    first_track = make_track(
        "CAM01",
        1,
        0.0,
        10.0,
    )

    second_track = make_track(
        "CAM02",
        2,
        12.0,
        20.0,
    )

    candidate = builder.build_pair(
        first_track,
        second_track,
    )

    assert candidate is not None

    data = candidate.to_dict()

    assert data["from_camera"] == "CAM01"
    assert data["from_track_id"] == 1
    assert data["to_camera"] == "CAM02"
    assert data["to_track_id"] == 2
    assert data["travel_time_seconds"] == pytest.approx(2.0)