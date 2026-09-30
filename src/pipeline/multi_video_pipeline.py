"""Multi-video analysis pipeline.

Phase 11:
    video validation
        -> YOLO person detection
        -> ByteTrack tracking
        -> person crops
        -> OSNet Re-ID embeddings

Phase 13:
    track summaries
        -> cross-camera Re-ID matching
        -> temporal/topology evidence
        -> global identity assignment
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import yaml

from src.crops.crop_extractor import CropExtractor
from src.data.video_validator import validate_video
from src.detection.detector import Detector
from src.matching.cross_camera_matcher import CrossCameraMatcher
from src.matching.types import TrackReference
from src.pipeline.types import (
    GlobalIdentityResult,
    MultiVideoAnalysisResult,
    VideoAnalysisResult,
    VideoInput,
)
from src.reid.embedder import ReIDEmbedder
from src.temporal.topology import CameraTopology
from src.temporal.track_summary import TrackSummaryBuilder
from src.tracking.tracker import Tracker


class MultiVideoPipelineError(RuntimeError):
    """Raised when multi-video analysis cannot be completed."""


class MultiVideoPipeline:
    """Coordinate multi-camera video analysis."""

    def __init__(
        self,
        config_path: str | Path = "configs/config.yaml",
        reid_config_path: str | Path = "configs/reid.yaml",
    ) -> None:
        self.config_path = Path(config_path)
        self.reid_config_path = Path(reid_config_path)

        self.config = self._load_yaml(self.config_path)
        self.reid_config = self._load_yaml(self.reid_config_path)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        """Load a YAML configuration file."""

        if not path.exists():
            raise MultiVideoPipelineError(
                f"Configuration file does not exist: {path}"
            )

        try:
            with path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
        except Exception as exc:
            raise MultiVideoPipelineError(
                f"Failed to load configuration file: {path}"
            ) from exc

        if data is None:
            return {}

        if not isinstance(data, dict):
            raise MultiVideoPipelineError(
                f"Configuration must contain a mapping: {path}"
            )

        return data

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def validate_inputs(
        self,
        video_inputs: (
            list[VideoInput]
            | tuple[VideoInput, ...]
            | None
        ),
    ) -> tuple[VideoInput, ...]:
        """Validate and normalize 2–3 video inputs."""

        if video_inputs is None:
            raise MultiVideoPipelineError(
                "Video inputs cannot be None"
            )

        inputs = tuple(video_inputs)

        if len(inputs) < 2:
            raise MultiVideoPipelineError(
                "At least 2 videos are required"
            )

        if len(inputs) > 3:
            raise MultiVideoPipelineError(
                "At most 3 videos are supported"
            )

        camera_ids = [video.camera_id for video in inputs]

        if len(camera_ids) != len(set(camera_ids)):
            raise MultiVideoPipelineError(
                "Duplicate camera_id values are not allowed"
            )

        supported_extensions = {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".m4v",
        }

        for video in inputs:
            path = Path(video.video_path)

            if not path.exists():
                raise MultiVideoPipelineError(
                    f"Video file does not exist: {path}"
                )

            if not path.is_file():
                raise MultiVideoPipelineError(
                    f"Video path is not a file: {path}"
                )

            if path.suffix.lower() not in supported_extensions:
                raise MultiVideoPipelineError(
                    f"Unsupported video format: {path.suffix}"
                )

            validation = validate_video(path)

            if validation.get("status") != "READABLE":
                raise MultiVideoPipelineError(
                    f"Video is unreadable: {path}"
                )

        return inputs

    # ------------------------------------------------------------------
    # Phase 11 preparation API
    # ------------------------------------------------------------------

    def prepare_analysis(
        self,
        video_inputs: (
            list[VideoInput]
            | tuple[VideoInput, ...]
            | None
        ),
    ) -> MultiVideoAnalysisResult:
        """Validate videos and prepare an analysis result."""

        validated_inputs = self.validate_inputs(video_inputs)

        videos = tuple(
            VideoAnalysisResult(
                camera_id=video.camera_id,
                video_path=str(video.video_path),
                status="ready",
            )
            for video in validated_inputs
        )

        return MultiVideoAnalysisResult(
            status="ready",
            videos=videos,
            unique_global_people=0,
            global_identities=(),
            error=None,
        )

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _get_detection_config(self) -> dict[str, Any]:
        """Return detection configuration."""

        detection = self.config.get("detection", {})

        if not isinstance(detection, dict):
            return {}

        return detection

    def _get_tracking_config(self) -> dict[str, Any]:
        """Return tracking configuration."""

        tracking = self.config.get("tracking", {})

        if not isinstance(tracking, dict):
            return {}

        return tracking

    def _get_crop_config(self) -> dict[str, Any]:
        """Return crop configuration."""

        crops = self.config.get("crops", {})

        if not isinstance(crops, dict):
            return {}

        return crops

    def _get_reid_config(self) -> dict[str, Any]:
        """Return Re-ID configuration."""

        reid = self.reid_config.get("reid", {})

        if not isinstance(reid, dict):
            return {}

        return reid

    # ------------------------------------------------------------------
    # Phase 11 video processing
    # ------------------------------------------------------------------

    def _process_video(
        self,
        video: VideoInput,
    ) -> VideoAnalysisResult:
        """Run detection, tracking, crops and Re-ID on one video."""

        video_path = Path(video.video_path)

        detector: Detector | None = None
        tracker: Tracker | None = None
        crop_extractor: CropExtractor | None = None
        embedder: ReIDEmbedder | None = None

        try:
            tracking_config = self._get_tracking_config()
            crop_config = self._get_crop_config()

            # ----------------------------------------------------------
            # Initialize detector
            # ----------------------------------------------------------

            detector = Detector.from_config(self.config)

            # ----------------------------------------------------------
            # Initialize tracker
            # ----------------------------------------------------------

            tracker = Tracker.from_config(self.config)

            # ----------------------------------------------------------
            # Initialize crop extractor
            # ----------------------------------------------------------

            crop_extractor = CropExtractor.from_config(
                self.config,
                detector=detector,
                tracker=tracker,
            )

            # ----------------------------------------------------------
            # Prepare output directories
            # ----------------------------------------------------------

            crop_output_root = Path(
                str(
                    crop_config.get(
                        "output_dir",
                        "outputs/crops",
                    )
                )
            )

            camera_crop_dir = (
                crop_output_root / video.camera_id
            )

            camera_crop_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            tracking_output_root = Path(
                str(
                    tracking_config.get(
                        "output_dir",
                        "outputs/tracking",
                    )
                )
            )

            tracking_csv = (
                tracking_output_root
                / video.camera_id
                / "tracks.csv"
            )

            tracking_csv.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            # ----------------------------------------------------------
            # Remove stale crop data for this camera
            # ----------------------------------------------------------

            self._clear_camera_crop_directory(
                camera_crop_dir
            )

            # ----------------------------------------------------------
            # Run detection + tracking
            # ----------------------------------------------------------

            tracking_result = tracker.track_video(
                video_path=video_path,
                detector=detector,
                camera_id=video.camera_id,
                output_csv=tracking_csv,
                annotate=False,
                max_frames=None,
            )

            tracks = getattr(
                tracking_result,
                "tracks",
                (),
            )

            frames_processed = int(
                getattr(
                    tracking_result,
                    "frames_processed",
                    0,
                )
            )

            unique_track_ids = {
                int(getattr(track, "track_id"))
                for track in tracks
                if getattr(track, "track_id", None) is not None
            }

            # ----------------------------------------------------------
            # Extract person crops
            # ----------------------------------------------------------

            crops_generated = self._extract_crops_from_tracks(
                video=video,
                tracks=tracks,
                crop_extractor=crop_extractor,
                camera_crop_dir=camera_crop_dir,
            )
            self._filter_reid_quality_tracks(
                camera_crop_dir
            )

            # ----------------------------------------------------------
            # Generate OSNet embeddings
            # ----------------------------------------------------------

            embeddings_generated = 0

            if camera_crop_dir.exists():
                embedder = ReIDEmbedder.from_config(
                    self.reid_config
                )

                embedding_output_root = Path(
                    str(
                        self._get_reid_config().get(
                            "output_dir",
                            "outputs/embeddings",
                        )
                    )
                )

                embedding_output_root.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                camera_embedding_dir = (
                    embedding_output_root
                    / video.camera_id
                )

                self._clear_directory_contents(
                    camera_embedding_dir
                )

                embedder.extract_directory(
                    camera_crop_dir,
                    embedding_output_root,
                )

                embeddings_generated = (
                    self._count_embedding_files(
                        camera_embedding_dir
                    )
                )

            return VideoAnalysisResult(
                camera_id=video.camera_id,
                video_path=str(video_path),
                status="completed",
                frames_processed=frames_processed,
                tracks_detected=len(unique_track_ids),
                crops_generated=crops_generated,
                embeddings_generated=embeddings_generated,
                output_directory=str(
                    camera_crop_dir.resolve()
                ),
                error=None,
            )

        except Exception as exc:
            return VideoAnalysisResult(
                camera_id=video.camera_id,
                video_path=str(video_path),
                status="failed",
                frames_processed=0,
                tracks_detected=0,
                crops_generated=0,
                embeddings_generated=0,
                output_directory=None,
                error=str(exc),
            )

        finally:
            self._safe_close(embedder)
            self._safe_close(crop_extractor)
            self._safe_close(tracker)
            self._safe_close(detector)

    # ------------------------------------------------------------------
    # Crop helpers
    # ------------------------------------------------------------------

    def _extract_crops_from_tracks(
        self,
        video: VideoInput,
        tracks: Any,
        crop_extractor: CropExtractor,
        camera_crop_dir: Path,
    ) -> int:
        """Extract crops from the tracks produced by the existing tracker pass."""

        video_path = Path(video.video_path)

        if not video_path.is_file():
            return 0

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            cap.release()
            return 0

        # Reuse the exact TrackedPerson objects produced by the tracker.
        # Group them by frame number so no second detection/tracking pass occurs.
        tracks_by_frame: dict[int, list[Any]] = {}

        for track in tracks:
            frame_number = getattr(track, "frame_number", None)

            if frame_number is None:
                continue

            tracks_by_frame.setdefault(int(frame_number), []).append(track)

        generated = 0
        frame_number = 0
        save_every = int(
            crop_extractor.config["save_every_n_frames"]
        )

        try:
            while True:
                ok, frame = cap.read()

                if not ok or frame is None:
                    break

                frame_tracks = tracks_by_frame.get(frame_number, [])

                if frame_number % save_every == 0:
                    for track in frame_tracks:
                        _, reason = crop_extractor.extract_track_crop(
                            frame=frame,
                            track=track,
                            source_video=video_path,
                            output_directory=camera_crop_dir,
                        )

                        if reason is None:
                            generated += 1

                frame_number += 1

        finally:
            cap.release()

        return generated
    def _filter_reid_quality_tracks(
        self,
        camera_crop_dir: Path,
    ) -> int:
        """Remove crop images and tracks that do not meet Re-ID quality requirements."""

        quality = self._get_reid_config().get(
            "quality",
            {},
        )

        if not isinstance(quality, dict):
            return 0

        if not quality.get("enabled", False):
            return 0

        min_valid_crops = int(
            quality.get("min_valid_crops", 3)
        )
        min_crop_width = int(
            quality.get("min_crop_width", 40)
        )
        min_crop_height = int(
            quality.get("min_crop_height", 80)
        )
        min_crop_area = int(
            quality.get("min_crop_area", 3200)
        )

        if not camera_crop_dir.is_dir():
            return 0

        import cv2

        removed_tracks = 0

        for track_dir in sorted(
            camera_crop_dir.glob("track_*")
        ):
            if not track_dir.is_dir():
                continue

            valid_crops = 0

            for crop_path in sorted(
                track_dir.glob("*.jpg")
            ):
                image = cv2.imread(
                    str(crop_path),
                    cv2.IMREAD_UNCHANGED,
                )

                if image is None:
                    crop_path.unlink(
                        missing_ok=True
                    )
                    continue

                height, width = image.shape[:2]
                area = width * height

                if (
                    width < min_crop_width
                    or height < min_crop_height
                    or area < min_crop_area
                ):
                    crop_path.unlink(
                        missing_ok=True
                    )
                    continue

                valid_crops += 1

            if valid_crops < min_valid_crops:
                for remaining_crop in track_dir.glob(
                    "*.jpg"
                ):
                    remaining_crop.unlink(
                        missing_ok=True
                    )

                try:
                    track_dir.rmdir()
                except OSError:
                    pass

                removed_tracks += 1

        return removed_tracks

    # Phase 13 cross-camera integration
    # ------------------------------------------------------------------

    def _run_cross_camera_matching(
        self,
        videos: tuple[VideoInput, ...],
        video_results: tuple[VideoAnalysisResult, ...],
    ) -> tuple[GlobalIdentityResult, ...]:
        """Match tracks across cameras and assign global identities."""

        successful_cameras = {
            result.camera_id
            for result in video_results
            if result.status == "completed"
        }

        if len(successful_cameras) < 2:
            return ()

        topology = CameraTopology.from_yaml(
            self.config_path
        )

        matcher = CrossCameraMatcher(
            topology=topology
        )

        tracks_by_camera: dict[
            str,
            list[Any],
        ] = {}

        embeddings_by_camera: dict[
            str,
            dict[int, Path],
        ] = {}

        # --------------------------------------------------------------
        # Build track summaries and discover embeddings.
        # --------------------------------------------------------------

        for video in videos:
            if video.camera_id not in successful_cameras:
                continue

            tracks_csv = (
                Path("outputs")
                / "tracking"
                / video.camera_id
                / "tracks.csv"
            )

            if not tracks_csv.exists():
                continue

            # TrackSummaryBuilder requires the CSV path in __init__.
            track_summary_builder = TrackSummaryBuilder(
                tracks_csv
            )

            summaries = track_summary_builder.build()

            tracks_by_camera[
                video.camera_id
            ] = summaries

            embeddings_by_camera[
                video.camera_id
            ] = self._discover_track_embeddings(
                video.camera_id
            )

        if not tracks_by_camera:
            return ()

        # --------------------------------------------------------------
        # Preserve the user's supplied camera order.
        # --------------------------------------------------------------

        ordered_cameras = [
            video.camera_id
            for video in videos
            if video.camera_id in successful_cameras
        ]

        identity_results: list[
            GlobalIdentityResult
        ] = []

        # --------------------------------------------------------------
        # Seed global identities using the first camera that has
        # usable track embeddings.
        # --------------------------------------------------------------

        seeded_camera: str | None = None

        for camera_id in ordered_cameras:
            camera_tracks = tracks_by_camera.get(
                camera_id,
                [],
            )

            camera_embeddings = embeddings_by_camera.get(
                camera_id,
                {},
            )

            if not camera_embeddings:
                continue

            for summary in camera_tracks:
                track_id = int(summary.track_id)

                if track_id not in camera_embeddings:
                    continue

                reference = TrackReference(
                    camera_id=camera_id,
                    track_id=track_id,
                )

                assignment = (
                    matcher.identity_pipeline
                    .assigner
                    .assign_new_track(
                        reference
                    )
                )

                identity_results.append(
                    GlobalIdentityResult(
                        global_person_id=(
                            assignment.global_person_id
                        ),
                        camera_id=camera_id,
                        track_id=track_id,
                        decision="new_identity",
                        score={},
                        evidence={},
                        assignment_reason=(
                            assignment.assignment_reason
                        ),
                    )
                )

            seeded_camera = camera_id
            break

        if seeded_camera is None:
            return ()

        # --------------------------------------------------------------
        # Process remaining cameras.
        # --------------------------------------------------------------

        seeded_index = ordered_cameras.index(
            seeded_camera
        )

        for camera_index in range(
            seeded_index + 1,
            len(ordered_cameras),
        ):
            to_camera = ordered_cameras[
                camera_index
            ]

            to_tracks = tracks_by_camera.get(
                to_camera,
                [],
            )

            to_embeddings = embeddings_by_camera.get(
                to_camera,
                {},
            )

            if not to_embeddings:
                continue

            previous_cameras = ordered_cameras[
                :camera_index
            ]

            for to_summary in to_tracks:
                to_track_id = int(
                    to_summary.track_id
                )

                to_embedding_path = to_embeddings.get(
                    to_track_id
                )

                if to_embedding_path is None:
                    continue

                best_accept: Any | None = None
                best_defer: Any | None = None

                # ------------------------------------------------------
                # Compare this track against tracks from earlier
                # cameras. Topology and temporal constraints are
                # enforced by CrossCameraMatcher.
                # ------------------------------------------------------

                for from_camera in previous_cameras:
                    from_tracks = tracks_by_camera.get(
                        from_camera,
                        [],
                    )

                    from_embeddings = (
                        embeddings_by_camera.get(
                            from_camera,
                            {},
                        )
                    )

                    for from_summary in from_tracks:
                        from_track_id = int(
                            from_summary.track_id
                        )

                        from_embedding_path = (
                            from_embeddings.get(
                                from_track_id
                            )
                        )

                        if from_embedding_path is None:
                            continue

                        from_reference = TrackReference(
                            camera_id=from_camera,
                            track_id=from_track_id,
                        )

                        existing_assignment = (
                            matcher.identity_pipeline
                            .assigner
                            .get_assignment(
                                from_reference
                            )
                        )

                        if existing_assignment is None:
                            continue

                        try:
                            match_result = (
                                matcher.match_track_pair(
                                    from_track=from_summary,
                                    to_track=to_summary,
                                    from_embeddings_path=(
                                        from_embedding_path
                                    ),
                                    to_embeddings_path=(
                                        to_embedding_path
                                    ),
                                    target_global_person_id=(
                                        existing_assignment
                                        .global_person_id
                                    ),
                                )
                            )
                        except Exception:
                            continue

                        if match_result is None:
                            continue

                        if (
                            match_result.identity_result
                            .accepted
                        ):
                            if (
                                best_accept is None
                                or (
                                    match_result.score.total_score
                                    > best_accept.score.total_score
                                )
                            ):
                                best_accept = match_result

                        elif (
                            match_result.identity_result
                            .deferred
                        ):
                            if (
                                best_defer is None
                                or (
                                    match_result.score.total_score
                                    > best_defer.score.total_score
                                )
                            ):
                                best_defer = match_result

                # ------------------------------------------------------
                # ACCEPT takes priority over DEFER.
                # ------------------------------------------------------

                selected = (
                    best_accept
                    if best_accept is not None
                    else best_defer
                )

                if selected is not None:
                    identity_result = (
                        selected.identity_result
                    )

                    assignment = (
                        identity_result.assignment
                    )

                    if assignment is not None:
                        identity_results.append(
                            GlobalIdentityResult(
                                global_person_id=(
                                    assignment
                                    .global_person_id
                                ),
                                camera_id=to_camera,
                                track_id=to_track_id,
                                decision=(
                                    identity_result
                                    .decision
                                    .decision
                                    .value
                                ),
                                score=(
                                    selected.score.to_dict()
                                ),
                                evidence=(
                                    identity_result
                                    .candidate
                                    .evidence
                                    .to_dict()
                                ),
                                assignment_reason=(
                                    assignment
                                    .assignment_reason
                                ),
                            )
                        )

                        continue

                # ------------------------------------------------------
                # No accepted/deferred candidate:
                # create a new global identity.
                # ------------------------------------------------------

                to_reference = TrackReference(
                    camera_id=to_camera,
                    track_id=to_track_id,
                )

                new_assignment = (
                    matcher.identity_pipeline
                    .assigner
                    .assign_new_track(
                        to_reference
                    )
                )

                identity_results.append(
                    GlobalIdentityResult(
                        global_person_id=(
                            new_assignment
                            .global_person_id
                        ),
                        camera_id=to_camera,
                        track_id=to_track_id,
                        decision="new_identity",
                        score={},
                        evidence={},
                        assignment_reason=(
                            new_assignment
                            .assignment_reason
                        ),
                    )
                )

        return tuple(identity_results)

    # ------------------------------------------------------------------
    # Embedding discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _discover_track_embeddings(
        camera_id: str,
    ) -> dict[int, Path]:
        """Discover track-level embedding files for one camera."""

        root = (
            Path("outputs")
            / "embeddings"
            / camera_id
        )

        if not root.exists():
            return {}

        result: dict[int, Path] = {}

        for embedding_path in root.glob(
            "track_*/embeddings.npy"
        ):
            track_directory = embedding_path.parent

            try:
                track_id = int(
                    track_directory.name.split("_")[-1]
                )
            except ValueError:
                continue

            result[track_id] = embedding_path

        return result

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def run(
        self,
        video_inputs: (
            list[VideoInput]
            | tuple[VideoInput, ...]
            | None
        ),
    ) -> MultiVideoAnalysisResult:
        """Run the complete multi-video analysis pipeline."""

        try:
            validated_inputs = self.validate_inputs(
                video_inputs
            )

            results: list[
                VideoAnalysisResult
            ] = []

            for video in validated_inputs:
                result = self._process_video(
                    video
                )

                results.append(result)

                if result.status != "completed":
                    return MultiVideoAnalysisResult(
                        status="failed",
                        videos=tuple(results),
                        unique_global_people=0,
                        global_identities=(),
                        error=(
                            result.error
                            or (
                                "Video processing failed: "
                                f"{video.video_path}"
                            )
                        ),
                    )

            video_results = tuple(results)

            # ----------------------------------------------------------
            # Phase 13
            # ----------------------------------------------------------

            global_identities = (
                self._run_cross_camera_matching(
                    videos=validated_inputs,
                    video_results=video_results,
                )
            )

            unique_global_people = len(
                {
                    identity.global_person_id
                    for identity in global_identities
                }
            )

            return MultiVideoAnalysisResult(
                status="completed",
                videos=video_results,
                unique_global_people=(
                    unique_global_people
                ),
                global_identities=(
                    global_identities
                ),
                error=None,
            )

        except MultiVideoPipelineError:
            raise

        except Exception as exc:
            return MultiVideoAnalysisResult(
                status="failed",
                videos=(),
                unique_global_people=0,
                global_identities=(),
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Filesystem helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clear_camera_crop_directory(
        directory: Path,
    ) -> None:
        """Remove stale crop files/directories for one camera."""

        if not directory.exists():
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )
            return

        for child in directory.iterdir():
            if child.is_dir():
                MultiVideoPipeline._remove_directory(
                    child
                )
            else:
                child.unlink(
                    missing_ok=True
                )

    @staticmethod
    def _clear_directory_contents(
        directory: Path,
    ) -> None:
        """Clear all files/directories below a path."""

        if not directory.exists():
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )
            return

        for child in directory.iterdir():
            if child.is_dir():
                MultiVideoPipeline._remove_directory(
                    child
                )
            else:
                child.unlink(
                    missing_ok=True
                )

    @staticmethod
    def _remove_directory(
        directory: Path,
    ) -> None:
        """Recursively remove a directory."""

        if not directory.exists():
            return

        for child in directory.iterdir():
            if child.is_dir():
                MultiVideoPipeline._remove_directory(
                    child
                )
            else:
                child.unlink(
                    missing_ok=True
                )

        directory.rmdir()

    @staticmethod
    def _count_crop_images(
        directory: Path,
    ) -> int:
        """Count generated crop images."""

        if not directory.exists():
            return 0

        extensions = {
            ".jpg",
            ".jpeg",
            ".png",
        }

        return sum(
            1
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in extensions
        )

    @staticmethod
    def _count_embedding_files(
        directory: Path,
    ) -> int:
        """Count track-level embedding files."""

        if not directory.exists():
            return 0

        return sum(
            1
            for path in directory.rglob(
                "embeddings.npy"
            )
            if path.is_file()
        )

    @staticmethod
    def _safe_close(
        component: Any,
    ) -> None:
        """Safely close a pipeline component if supported."""

        if component is None:
            return

        close_method = getattr(
            component,
            "close",
            None,
        )

        if callable(close_method):
            try:
                close_method()
            except Exception:
                pass