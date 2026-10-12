from __future__ import annotations

from backend.app.api import searches
from src.pipeline.types import (
    GlobalIdentityResult,
    MultiVideoAnalysisResult,
    VideoAnalysisResult,
    build_matched_person_summary,
)
from src.pipeline.multi_video_pipeline import MultiVideoPipeline
from src.tracking.types import TrackedPerson


def _identity(global_id, camera, track, decision="new_identity", **extra):
    return {
        "global_person_id": global_id,
        "camera_id": camera,
        "track_id": track,
        "decision": decision,
        "score": extra.get("score"),
        "evidence": extra.get("evidence"),
    }


def _video(camera, tracks, observations):
    return {
        "camera_id": camera,
        "status": "completed",
        "tracks_detected": tracks,
        "detection_count": tracks * 4,
        "track_observations": observations,
    }


def _observation(track, start, end):
    return {
        "track_id": track,
        "first_frame": int(start * 10),
        "last_frame": int(end * 10),
        "first_time_seconds": start,
        "last_time_seconds": end,
        "duration_seconds": end - start,
        "timestamp_basis": "video_relative_seconds_from_frame_index_and_fps",
    }


def test_one_accepted_global_identity_across_two_cameras_counts_once():
    videos = [
        _video("CAM01", 13, [_observation(7, 4.2, 8.5)]),
        _video("CAM02", 13, [_observation(10, 10.0, 13.6)]),
    ]
    identities = [
        _identity(4, "CAM01", 7),
        _identity(4, "CAM02", 10, "accept", score={"total_score": 0.87}),
    ]
    summary = build_matched_person_summary(videos, identities)

    assert summary["matched_people_count"] == 1
    assert summary["total_tracks"] == 26
    assert summary["total_detections"] == 104
    assert summary["camera_observations_count"] == 2
    person = summary["matched_people"][0]
    assert person["global_person_id"] == 4
    assert person["camera_ids"] == ["CAM01", "CAM02"]
    assert [row["track_id"] for row in person["observations"]] == [7, 10]
    assert person["observations"][0]["first_time_seconds"] == 4.2
    assert person["observations"][1]["last_time_seconds"] == 13.6
    assert person["observations"][0]["absolute_capture_start"] is None


def test_candidates_and_single_camera_or_rejected_assignments_do_not_count():
    videos = [_video("CAM01", 1, [_observation(3, 0.0, 1.0)])]
    identities = [
        _identity(1, "CAM01", 3, "accept"),
        _identity(2, "CAM01", 4, "new_identity"),
        _identity(2, "CAM02", 8, "reject"),
    ]
    summary = build_matched_person_summary(videos, identities)
    assert summary["matched_people_count"] == 0
    assert summary["camera_observations_count"] == 0
    assert summary["matched_people"] == []


def test_duplicate_identity_rows_do_not_inflate_camera_observations():
    videos = [
        _video("CAM01", 2, [_observation(3, 0.0, 1.0)]),
        _video("CAM02", 2, [_observation(8, 2.0, 3.0)]),
    ]
    identities = [
        _identity(1, "CAM01", 3),
        _identity(1, "CAM01", 3),
        _identity(1, "CAM02", 8, "accept", evidence={"mean_similarity": 0.91}),
    ]
    summary = build_matched_person_summary(videos, identities)
    assert summary["matched_people_count"] == 1
    assert summary["camera_observations_count"] == 2
    assert len(summary["matched_people"][0]["observations"]) == 2


def test_multiple_accepted_global_ids_are_counted_independently():
    identities = [
        _identity(1, "CAM01", 1), _identity(1, "CAM02", 2, "accept"),
        _identity(2, "CAM01", 3), _identity(2, "CAM02", 4, "accept"),
    ]
    summary = build_matched_person_summary([], identities)
    assert summary["matched_people_count"] == 2
    assert summary["camera_observations_count"] == 4


def test_missing_detection_count_is_reported_as_unavailable():
    summary = build_matched_person_summary(
        [{"camera_id": "CAM01", "status": "completed", "tracks_detected": 0}], []
    )
    assert summary["total_detections"] is None
    assert summary["matched_people_count"] == 0


def test_partial_camera_processing_does_not_report_incomplete_totals_as_final():
    summary = build_matched_person_summary(
        [
            _video("CAM01", 2, []),
            {"camera_id": "CAM02", "status": "failed", "tracks_detected": 0, "detection_count": 0},
        ],
        [],
    )
    assert summary["total_tracks"] is None
    assert summary["total_detections"] is None
    assert summary["cameras_processed_count"] == 1


def test_pipeline_serialization_exposes_authoritative_summary_not_candidate_count():
    result = MultiVideoAnalysisResult(
        status="completed",
        videos=(
            VideoAnalysisResult(
                "CAM01", "a.mp4", "completed", tracks_detected=1,
                detection_count=5, track_observations=(_observation(7, 1.0, 2.0),),
            ),
            VideoAnalysisResult(
                "CAM02", "b.mp4", "completed", tracks_detected=1,
                detection_count=6, track_observations=(_observation(10, 3.0, 4.0),),
            ),
        ),
        global_identities=(
            GlobalIdentityResult(1, "CAM01", 7, "new_identity", None, None, "seed"),
            GlobalIdentityResult(1, "CAM02", 10, "accept", {"total_score": .9}, {"mean_similarity": .9}, "accepted"),
        ),
        candidate_reports=({"status": "valid"}, {"status": "valid"}),
    )

    serialized = result.to_dict()
    assert serialized["summary"]["matched_people_count"] == 1
    assert serialized["summary"]["total_detections"] == 11
    assert serialized["summary"]["camera_observations_count"] == 2


def test_track_timestamp_summary_uses_frame_order_and_rejects_non_monotonic_time():
    tracks = [
        TrackedPerson("CAM01", 4, 0.4, 7, 0, 0, 10, 10, .9, 0, "person"),
        TrackedPerson("CAM01", 1, 0.1, 7, 0, 0, 10, 10, .9, 0, "person"),
        TrackedPerson("CAM01", 6, 0.6, 7, 0, 0, 10, 10, .9, 0, "person"),
        TrackedPerson("CAM02", 1, 2.0, 7, 0, 0, 10, 10, .9, 0, "person"),
        TrackedPerson("CAM02", 2, 1.0, 7, 0, 0, 10, 10, .9, 0, "person"),
    ]
    rows = MultiVideoPipeline._summarize_track_observations(tracks)
    first_camera, second_camera = rows
    assert (first_camera["first_frame"], first_camera["last_frame"]) == (1, 6)
    assert (first_camera["first_time_seconds"], first_camera["last_time_seconds"]) == (.1, .6)
    assert first_camera["duration_seconds"] == .5
    assert second_camera["first_time_seconds"] is None
    assert second_camera["timestamp_basis"] == "unavailable"


def test_legacy_saved_result_is_enriched_from_existing_tracking_csv(tmp_path, monkeypatch):
    search_id = 931
    analysis_root = tmp_path / "analyses"
    camera_dir = analysis_root / f"search_{search_id}" / "outputs" / "tracking" / "CAM01"
    camera_dir.mkdir(parents=True)
    csv_path = camera_dir / "tracks.csv"
    csv_path.write_text(
        "camera_id,frame_number,timestamp_seconds,track_id,x1,y1,x2,y2\n"
        "CAM01,0,0.0,7,0,0,20,40\n"
        "CAM01,20,2.0,7,0,0,20,40\n",
        encoding="utf-8",
    )
    camera_2_dir = analysis_root / f"search_{search_id}" / "outputs" / "tracking" / "CAM02"
    camera_2_dir.mkdir(parents=True)
    (camera_2_dir / "tracks.csv").write_text(
        "camera_id,frame_number,timestamp_seconds,track_id,x1,y1,x2,y2\n"
        "CAM02,10,1.0,10,0,0,20,40\n"
        "CAM02,30,3.0,10,0,0,20,40\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(searches, "ANALYSIS_STORAGE_DIR", analysis_root)
    payload = {
        "videos": [
            {"camera_id": "CAM01", "status": "completed", "tracks_detected": 1},
            {"camera_id": "CAM02", "status": "completed", "tracks_detected": 1},
        ],
        "result": {
            "videos": [
                {"camera_id": "CAM01", "status": "completed", "tracks_detected": 1},
                {"camera_id": "CAM02", "status": "completed", "tracks_detected": 1},
            ],
            "global_identities": [
                _identity(4, "CAM01", 7),
                _identity(4, "CAM02", 10, "accept"),
            ],
        },
    }
    enriched = searches._add_legacy_summary(search_id, payload)
    summary = enriched["result"]["summary"]
    assert summary["matched_people_count"] == 1
    observations = summary["matched_people"][0]["observations"]
    assert [(row["camera_id"], row["first_time_seconds"], row["last_time_seconds"]) for row in observations] == [
        ("CAM01", 0.0, 2.0),
        ("CAM02", 1.0, 3.0),
    ]
