"""Camera topology and travel-time constraints for Phase 9."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class TopologyError(RuntimeError):
    """Raised when camera topology configuration is invalid."""


@dataclass(frozen=True)
class CameraTransition:
    """Allowed transition between two cameras."""

    from_camera: str
    to_camera: str
    min_travel_time_seconds: float
    max_travel_time_seconds: float

    def __post_init__(self) -> None:
        if not self.from_camera:
            raise ValueError("from_camera must be non-empty")

        if not self.to_camera:
            raise ValueError("to_camera must be non-empty")

        if self.from_camera == self.to_camera:
            raise ValueError(
                "from_camera and to_camera must be different"
            )

        if self.min_travel_time_seconds < 0.0:
            raise ValueError(
                "min_travel_time_seconds must be non-negative"
            )

        if self.max_travel_time_seconds < 0.0:
            raise ValueError(
                "max_travel_time_seconds must be non-negative"
            )

        if (
            self.min_travel_time_seconds
            > self.max_travel_time_seconds
        ):
            raise ValueError(
                "min_travel_time_seconds cannot exceed "
                "max_travel_time_seconds"
            )


class CameraTopology:
    """Validated camera connectivity and temporal constraints."""

    def __init__(
        self,
        cameras: dict[str, list[str]],
        transitions: dict[str, CameraTransition],
    ) -> None:
        self._cameras = cameras
        self._transitions = transitions

        self._validate()

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
    ) -> "CameraTopology":
        """Load topology configuration from a YAML file."""
        path = Path(config_path)

        if not path.exists():
            raise TopologyError(
                f"Topology configuration does not exist: {path}"
            )

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                config = yaml.safe_load(handle)

        except (OSError, yaml.YAMLError) as exc:
            raise TopologyError(
                f"Could not read topology configuration: {path}"
            ) from exc

        if not isinstance(config, dict):
            raise TopologyError(
                "Topology configuration must be a mapping"
            )

        topology = config.get("topology")

        if not isinstance(topology, dict):
            raise TopologyError(
                "Missing 'topology' configuration"
            )

        if not topology.get("enabled", False):
            raise TopologyError(
                "Camera topology is disabled"
            )

        cameras_config = topology.get("cameras")

        if not isinstance(cameras_config, dict):
            raise TopologyError(
                "'cameras' must be a mapping"
            )

        cameras: dict[str, list[str]] = {}

        for camera_id, camera_config in cameras_config.items():
            if not isinstance(camera_id, str) or not camera_id:
                raise TopologyError(
                    "Camera IDs must be non-empty strings"
                )

            if not isinstance(camera_config, dict):
                raise TopologyError(
                    f"Configuration for {camera_id} must be a mapping"
                )

            outgoing = camera_config.get("outgoing", [])

            if not isinstance(outgoing, list):
                raise TopologyError(
                    f"Outgoing cameras for {camera_id} "
                    "must be a list"
                )

            cameras[camera_id] = [
                str(destination)
                for destination in outgoing
            ]

        transitions_config = topology.get("transitions", {})

        if not isinstance(transitions_config, dict):
            raise TopologyError(
                "'transitions' must be a mapping"
            )

        transitions: dict[str, CameraTransition] = {}

        for transition_id, transition_config in transitions_config.items():
            if not isinstance(transition_config, dict):
                raise TopologyError(
                    f"Transition {transition_id} must be a mapping"
                )

            try:
                transition = CameraTransition(
                    from_camera=str(
                        transition_config["from_camera"]
                    ),
                    to_camera=str(
                        transition_config["to_camera"]
                    ),
                    min_travel_time_seconds=float(
                        transition_config[
                            "min_travel_time_seconds"
                        ]
                    ),
                    max_travel_time_seconds=float(
                        transition_config[
                            "max_travel_time_seconds"
                        ]
                    ),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise TopologyError(
                    f"Invalid transition: {transition_id}"
                ) from exc

            transitions[str(transition_id)] = transition

        return cls(
            cameras=cameras,
            transitions=transitions,
        )

    def _validate(self) -> None:
        """Validate camera references and transition consistency."""
        camera_ids = set(self._cameras)

        for camera_id, outgoing in self._cameras.items():
            for destination in outgoing:
                if destination not in camera_ids:
                    raise TopologyError(
                        f"{camera_id} references unknown camera "
                        f"{destination}"
                    )

                if destination == camera_id:
                    raise TopologyError(
                        f"{camera_id} cannot connect to itself"
                    )

        for transition_id, transition in self._transitions.items():
            if transition.from_camera not in camera_ids:
                raise TopologyError(
                    f"{transition_id} references unknown source "
                    f"camera {transition.from_camera}"
                )

            if transition.to_camera not in camera_ids:
                raise TopologyError(
                    f"{transition_id} references unknown destination "
                    f"camera {transition.to_camera}"
                )

            outgoing = self._cameras[transition.from_camera]

            if transition.to_camera not in outgoing:
                raise TopologyError(
                    f"{transition_id} is not declared in "
                    f"{transition.from_camera}.outgoing"
                )

    @property
    def cameras(self) -> dict[str, list[str]]:
        """Return a copy of the camera connectivity mapping."""
        return {
            camera_id: list(outgoing)
            for camera_id, outgoing in self._cameras.items()
        }

    @property
    def transitions(self) -> dict[str, CameraTransition]:
        """Return a copy of the transition mapping."""
        return dict(self._transitions)

    def can_transition(
        self,
        from_camera: str,
        to_camera: str,
    ) -> bool:
        """Return whether a direct camera transition is allowed."""
        return to_camera in self._cameras.get(
            from_camera,
            [],
        )

    def get_transition(
        self,
        from_camera: str,
        to_camera: str,
    ) -> CameraTransition | None:
        """Return transition constraints for a camera pair."""
        for transition in self._transitions.values():
            if (
                transition.from_camera == from_camera
                and transition.to_camera == to_camera
            ):
                return transition

        return None

    def is_travel_time_valid(
        self,
        from_camera: str,
        to_camera: str,
        travel_time_seconds: float,
    ) -> bool:
        """Check whether a travel time fits the configured transition."""
        transition = self.get_transition(
            from_camera,
            to_camera,
        )

        if transition is None:
            return False

        if travel_time_seconds < 0.0:
            return False

        return (
            transition.min_travel_time_seconds
            <= travel_time_seconds
            <= transition.max_travel_time_seconds
        )