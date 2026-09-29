from __future__ import annotations

import numpy as np
import pytest

from src.matching.cross_camera_matcher import (
    CrossCameraMatcher,
)
from src.matching.identity_decision import IdentityDecision
from src.temporal.topology import CameraTopology, CameraTransition
from src.temporal.types import TrackSummary
from src.matching.types import TrackReference


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
        end_center_x=120.0,
        end_center_y=110.0,
        frames_seen=11,
    )


def make_topology() -> CameraTopology:
    return CameraTopology(
        cameras={
            "CAM01": ["CAM02"],
            "CAM02": [],
        },
        transitions={
            "CAM01_to_CAM02": CameraTransition(
                from_camera="CAM01",
                to_camera="CAM02",
                min_travel_time_seconds=2.0,
                max_travel_time_seconds=30.0,
            ),
        },
    )


def make_embeddings(
    path,
    vector,
    samples=4,
) -> None:
    embeddings = np.tile(
        np.asarray(vector, dtype=np.float32),
        (samples, 1),
    )

    embeddings = embeddings / np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    np.save(path, embeddings)


def test_cross_camera_match_builds_valid_candidate(
    tmp_path,
) -> None:
    from_track = make_track(
        camera_id="CAM01",
        track_id=1,
        start_time=0.0,
        end_time=5.0,
    )

    to_track = make_track(
        camera_id="CAM02",
        track_id=7,
        start_time=10.0,
        end_time=15.0,
    )

    from_path = tmp_path / "cam01_track1.npy"
    to_path = tmp_path / "cam02_track7.npy"

    make_embeddings(
        from_path,
        [1.0, 0.0, 0.0, 0.0],
    )

    make_embeddings(
        to_path,
        [1.0, 0.0, 0.0, 0.0],
    )

    matcher = CrossCameraMatcher(
        topology=make_topology(),
    )

    # Create the existing global identity from the source-camera track.
    identity = matcher.identity_pipeline.assigner.assign_new_track(
        TrackReference(
            camera_id="CAM01",
            track_id=1,
        )
    )

    result = matcher.match_track_pair(
        from_track=from_track,
        to_track=to_track,
        from_embeddings_path=from_path,
        to_embeddings_path=to_path,
        target_global_person_id=identity.global_person_id,
    )

    assert result is not None

    assert result.from_track.camera_id == "CAM01"
    assert result.from_track.track_id == 1

    assert result.to_track.camera_id == "CAM02"
    assert result.to_track.track_id == 7

    assert result.similarity.mean_similarity > 0.99
    assert result.similarity.max_similarity > 0.99

    assert result.transition.from_camera == "CAM01"
    assert result.transition.to_camera == "CAM02"
    assert result.transition.travel_time_seconds == 5.0

    assert result.score.total_score >= 0.85

    assert (
        result.identity_result.decision.decision
        == IdentityDecision.ACCEPT
    )

    assert result.identity_result.assignment is not None

    assert (
        result.identity_result.assignment.global_person_id
        == identity.global_person_id
    )


def test_invalid_camera_transition_returns_none(
    tmp_path,
) -> None:
    from_track = make_track(
        camera_id="CAM01",
        track_id=1,
        start_time=0.0,
        end_time=5.0,
    )

    to_track = make_track(
        camera_id="CAM03",
        track_id=7,
        start_time=10.0,
        end_time=15.0,
    )

    from_path = tmp_path / "cam01_track1.npy"
    to_path = tmp_path / "cam03_track7.npy"

    make_embeddings(
        from_path,
        [1.0, 0.0, 0.0, 0.0],
    )

    make_embeddings(
        to_path,
        [1.0, 0.0, 0.0, 0.0],
    )

    matcher = CrossCameraMatcher(
        topology=make_topology(),
    )

    result = matcher.match_track_pair(
        from_track=from_track,
        to_track=to_track,
        from_embeddings_path=from_path,
        to_embeddings_path=to_path,
        target_global_person_id=1,
    )

    assert result is None


def test_invalid_travel_time_returns_none(
    tmp_path,
) -> None:
    from_track = make_track(
        camera_id="CAM01",
        track_id=1,
        start_time=0.0,
        end_time=5.0,
    )

    to_track = make_track(
        camera_id="CAM02",
        track_id=7,
        start_time=40.0,
        end_time=45.0,
    )

    from_path = tmp_path / "cam01_track1.npy"
    to_path = tmp_path / "cam02_track7.npy"

    make_embeddings(
        from_path,
        [1.0, 0.0, 0.0, 0.0],
    )

    make_embeddings(
        to_path,
        [1.0, 0.0, 0.0, 0.0],
    )

    matcher = CrossCameraMatcher(
        topology=make_topology(),
    )

    result = matcher.match_track_pair(
        from_track=from_track,
        to_track=to_track,
        from_embeddings_path=from_path,
        to_embeddings_path=to_path,
        target_global_person_id=1,
    )

    assert result is None


def test_same_camera_pair_is_rejected(
    tmp_path,
) -> None:
    from_track = make_track(
        camera_id="CAM01",
        track_id=1,
        start_time=0.0,
        end_time=5.0,
    )

    to_track = make_track(
        camera_id="CAM01",
        track_id=7,
        start_time=10.0,
        end_time=15.0,
    )

    from_path = tmp_path / "cam01_track1.npy"
    to_path = tmp_path / "cam01_track7.npy"

    make_embeddings(
        from_path,
        [1.0, 0.0, 0.0, 0.0],
    )

    make_embeddings(
        to_path,
        [1.0, 0.0, 0.0, 0.0],
    )

    matcher = CrossCameraMatcher(
        topology=make_topology(),
    )

    try:
        matcher.match_track_pair(
            from_track=from_track,
            to_track=to_track,
            from_embeddings_path=from_path,
            to_embeddings_path=to_path,
            target_global_person_id=1,
        )
    except Exception as exc:
        assert "different cameras" in str(exc)
    else:
        raise AssertionError(
            "Expected same-camera matching to raise an error"
        )


def test_low_similarity_is_not_automatically_accepted(
    tmp_path,
) -> None:
    from_track = make_track(
        camera_id="CAM01",
        track_id=1,
        start_time=0.0,
        end_time=5.0,
    )

    to_track = make_track(
        camera_id="CAM02",
        track_id=7,
        start_time=10.0,
        end_time=15.0,
    )

    from_path = tmp_path / "cam01_track1.npy"
    to_path = tmp_path / "cam02_track7.npy"

    make_embeddings(
        from_path,
        [1.0, 0.0, 0.0, 0.0],
    )

    make_embeddings(
        to_path,
        [0.0, 1.0, 0.0, 0.0],
    )

    matcher = CrossCameraMatcher(
        topology=make_topology(),
    )

    result = matcher.match_track_pair(
        from_track=from_track,
        to_track=to_track,
        from_embeddings_path=from_path,
        to_embeddings_path=to_path,
        target_global_person_id=1,
    )

    assert result is not None
    assert result.similarity.mean_similarity == 0.0
    assert result.score.total_score == pytest.approx(0.3)
    assert (
        result.identity_result.decision.decision
        == IdentityDecision.REJECT
    )
    assert result.identity_result.assignment is None
