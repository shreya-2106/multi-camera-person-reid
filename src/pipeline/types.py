"""Data types for the multi-video analysis pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class VideoInput:
    """One video supplied to the analysis pipeline."""

    video_path: str
    camera_id: str

    def __post_init__(self) -> None:
        if not self.video_path:
            raise ValueError("video_path must be non-empty")

        if not self.camera_id:
            raise ValueError("camera_id must be non-empty")

    def to_dict(self) -> dict:
        """Convert the video input to a dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class VideoAnalysisResult:
    """Result of processing one uploaded video."""

    camera_id: str
    video_path: str
    status: str

    frames_processed: int = 0
    tracks_detected: int = 0
    crops_generated: int = 0
    embeddings_generated: int = 0

    output_directory: Optional[str] = None
    error: Optional[str] = None
    detection_count: int = 0
    fps: Optional[float] = None
    track_observations: tuple[dict, ...] = ()

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id must be non-empty")

        if not self.video_path:
            raise ValueError("video_path must be non-empty")

        if not self.status:
            raise ValueError("status must be non-empty")

        for field_name in (
            "frames_processed",
            "tracks_detected",
            "crops_generated",
            "embeddings_generated",
            "detection_count",
        ):
            value = getattr(self, field_name)

            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"{field_name} must be a non-negative integer"
                )

        if not isinstance(self.track_observations, tuple):
            raise ValueError("track_observations must be a tuple")

    def to_dict(self) -> dict:
        """Convert the video analysis result to a dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class GlobalIdentityResult:
    """Serializable result for one global person identity."""

    global_person_id: int
    camera_id: str
    track_id: int
    decision: str
    score: Optional[dict]
    evidence: Optional[dict]
    assignment_reason: str

    def __post_init__(self) -> None:
        if self.global_person_id < 1:
            raise ValueError(
                "global_person_id must be a positive integer"
            )

        if not self.camera_id:
            raise ValueError("camera_id must be non-empty")

        if self.track_id < 0:
            raise ValueError(
                "track_id must be a non-negative integer"
            )

        if not self.decision:
            raise ValueError("decision must be non-empty")

        if self.score is not None and not isinstance(self.score, dict):
            raise ValueError("score must be a dictionary or None")
        if self.evidence is not None and not isinstance(self.evidence, dict):
            raise ValueError("evidence must be a dictionary or None")

        if not self.assignment_reason:
            raise ValueError(
                "assignment_reason must be non-empty"
            )

    def to_dict(self) -> dict:
        """Convert the identity result to a dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class MultiVideoAnalysisResult:
    """Overall result for a multi-video analysis request."""

    status: str
    videos: tuple[VideoAnalysisResult, ...]

    unique_global_people: int = 0
    global_identities: tuple[GlobalIdentityResult, ...] = ()
    candidate_reports: tuple[dict, ...] = ()

    error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.status:
            raise ValueError("status must be non-empty")

        if self.unique_global_people < 0:
            raise ValueError(
                "unique_global_people must be non-negative"
            )

        if not isinstance(self.candidate_reports, tuple):
            raise ValueError(
                "candidate_reports must be a tuple"
            )

        for report in self.candidate_reports:
            if not isinstance(report, dict):
                raise ValueError(
                    "Each candidate report must be a dictionary"
                )

    def to_dict(self) -> dict:
        """Convert the complete result to a dictionary."""
        videos = [video.to_dict() for video in self.videos]
        identities = [identity.to_dict() for identity in self.global_identities]
        return {
            "status": self.status,
            "videos": videos,
            "unique_global_people": self.unique_global_people,
            "global_identities": [
                identity.to_dict()
                for identity in self.global_identities
            ],
            "candidate_reports": list(
                self.candidate_reports
            ),
            "summary": build_matched_person_summary(videos, identities),
            "error": self.error,
        }


def build_matched_person_summary(
    videos: list[dict], identities: list[dict]
) -> dict:
    """Build deduplicated matched-person and observation counts from assignments."""
    assignments: dict[int, dict[tuple[str, int], dict]] = {}
    accepted_ids: set[int] = set()
    for identity in identities:
        if not isinstance(identity, dict):
            continue
        global_id = identity.get("global_person_id")
        camera_id = identity.get("camera_id")
        track_id = identity.get("track_id")
        if (
            isinstance(global_id, bool) or not isinstance(global_id, int)
            or global_id <= 0 or not isinstance(camera_id, str) or not camera_id
            or isinstance(track_id, bool) or not isinstance(track_id, int)
            or track_id < 0
        ):
            continue
        key = (camera_id, track_id)
        existing = assignments.setdefault(global_id, {}).get(key)
        if existing is None or identity.get("decision") == "accept":
            assignments[global_id][key] = identity
        if identity.get("decision") == "accept":
            accepted_ids.add(global_id)

    timing_by_track = {}
    for video in videos:
        camera_id = video.get("camera_id")
        if not isinstance(camera_id, str):
            continue
        for observation in video.get("track_observations", ()):
            if not isinstance(observation, dict):
                continue
            track_id = observation.get("track_id")
            if isinstance(track_id, int) and not isinstance(track_id, bool):
                timing_by_track[(camera_id, track_id)] = observation

    matched_people = []
    for global_id in sorted(accepted_ids):
        grouped = assignments.get(global_id, {})
        camera_ids = sorted({camera for camera, _ in grouped})
        if len(camera_ids) < 2:
            continue
        observations = []
        evidence = []
        for (camera_id, track_id), identity in sorted(grouped.items()):
            timing = timing_by_track.get((camera_id, track_id), {})
            observations.append({
                "camera_id": camera_id,
                "track_id": track_id,
                "assignment_decision": identity.get("decision"),
                "first_frame": timing.get("first_frame"),
                "last_frame": timing.get("last_frame"),
                "first_time_seconds": timing.get("first_time_seconds"),
                "last_time_seconds": timing.get("last_time_seconds"),
                "duration_seconds": timing.get("duration_seconds"),
                "absolute_capture_start": None,
                "absolute_capture_end": None,
                "timestamp_basis": timing.get("timestamp_basis", "unavailable"),
            })
            if identity.get("decision") == "accept":
                evidence.append({
                    "camera_id": camera_id,
                    "track_id": track_id,
                    "score": identity.get("score"),
                    "evidence": identity.get("evidence"),
                })
        matched_people.append({
            "global_person_id": global_id,
            "match_status": "accepted",
            "distinct_camera_count": len(camera_ids),
            "camera_ids": camera_ids,
            "observations": observations,
            "matching_evidence": evidence,
        })

    return {
        "matched_people_count": len(matched_people),
        "total_tracks": (
            sum(video["tracks_detected"] for video in videos)
            if all(
                video.get("status") == "completed"
                and isinstance(video.get("tracks_detected"), int)
                and not isinstance(video.get("tracks_detected"), bool)
                for video in videos
            )
            else None
        ),
        "total_detections": (
            sum(video["detection_count"] for video in videos)
            if all(
                video.get("status") == "completed"
                and isinstance(video.get("detection_count"), int)
                and not isinstance(video.get("detection_count"), bool)
                for video in videos
            )
            else None
        ),
        "camera_observations_count": sum(
            len(person["observations"]) for person in matched_people
        ),
        "cameras_processed_count": sum(
            video.get("status") == "completed" for video in videos
        ),
        "matched_people": matched_people,
    }
