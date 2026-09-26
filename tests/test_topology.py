import pytest

from src.temporal.topology import (
    CameraTopology,
    CameraTransition,
    TopologyError,
)


TOPOLOGY_CONFIG = "configs/topology.yaml"


def test_load_real_topology():
    topology = CameraTopology.from_yaml(TOPOLOGY_CONFIG)

    assert set(topology.cameras) == {
        "CAM01",
        "CAM02",
        "CAM03",
        "CAM04",
    }

    assert topology.cameras["CAM01"] == [
        "CAM02",
        "CAM03",
    ]

    assert topology.cameras["CAM02"] == ["CAM04"]
    assert topology.cameras["CAM03"] == ["CAM04"]
    assert topology.cameras["CAM04"] == []


def test_real_transitions():
    topology = CameraTopology.from_yaml(TOPOLOGY_CONFIG)

    assert len(topology.transitions) == 4

    assert topology.can_transition(
        "CAM01",
        "CAM02",
    )

    assert topology.can_transition(
        "CAM01",
        "CAM03",
    )

    assert topology.can_transition(
        "CAM02",
        "CAM04",
    )

    assert topology.can_transition(
        "CAM03",
        "CAM04",
    )


def test_invalid_direct_transition():
    topology = CameraTopology.from_yaml(TOPOLOGY_CONFIG)

    assert not topology.can_transition(
        "CAM01",
        "CAM04",
    )


def test_get_transition():
    topology = CameraTopology.from_yaml(TOPOLOGY_CONFIG)

    transition = topology.get_transition(
        "CAM01",
        "CAM02",
    )

    assert transition is not None
    assert transition.from_camera == "CAM01"
    assert transition.to_camera == "CAM02"
    assert transition.min_travel_time_seconds == pytest.approx(2.0)
    assert transition.max_travel_time_seconds == pytest.approx(30.0)


def test_valid_travel_time():
    topology = CameraTopology.from_yaml(TOPOLOGY_CONFIG)

    assert topology.is_travel_time_valid(
        "CAM01",
        "CAM02",
        2.0,
    )

    assert topology.is_travel_time_valid(
        "CAM01",
        "CAM02",
        10.0,
    )

    assert topology.is_travel_time_valid(
        "CAM01",
        "CAM02",
        30.0,
    )


def test_invalid_travel_time():
    topology = CameraTopology.from_yaml(TOPOLOGY_CONFIG)

    assert not topology.is_travel_time_valid(
        "CAM01",
        "CAM02",
        1.0,
    )

    assert not topology.is_travel_time_valid(
        "CAM01",
        "CAM02",
        31.0,
    )


def test_unknown_camera_transition():
    topology = CameraTopology.from_yaml(TOPOLOGY_CONFIG)

    assert not topology.can_transition(
        "CAM01",
        "CAM99",
    )

    assert not topology.is_travel_time_valid(
        "CAM01",
        "CAM99",
        5.0,
    )


def test_invalid_camera_transition_definition():
    with pytest.raises(ValueError):
        CameraTransition(
            from_camera="CAM01",
            to_camera="CAM01",
            min_travel_time_seconds=2.0,
            max_travel_time_seconds=30.0,
        )


def test_invalid_travel_time_range():
    with pytest.raises(ValueError):
        CameraTransition(
            from_camera="CAM01",
            to_camera="CAM02",
            min_travel_time_seconds=30.0,
            max_travel_time_seconds=2.0,
        )


def test_missing_topology_file():
    with pytest.raises(TopologyError):
        CameraTopology.from_yaml(
            "configs/does_not_exist.yaml"
        )
        