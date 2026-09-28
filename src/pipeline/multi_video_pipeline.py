"""Multi-video analysis pipeline controller.

Phase 11:
    Validate uploaded videos
        -> YOLO person detection
        -> ByteTrack local tracking
        -> sampled person crop extraction
        -> trained OSNet Re-ID embeddings

FAISS matching, temporal evidence, and global identity assignment
remain separate phases.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterable, Optional

import cv2

from src.config import PROJECT_ROOT, load_config
from src.crops import CropExtractor
from src.crops.crop_extractor import should_save_frame
from src.crops.types import write_crops_csv
from src.data.video_validator import VIDEO_EXTENSIONS, validate_video
from src.detection import Detector
from src.pipeline.types import (
    MultiVideoAnalysisResult,
    VideoAnalysisResult,
    VideoInput,
)
from src.reid import ReIDEmbedder
from src.tracking import Tracker
from src.tracking.types import TrackedPerson, write_tracks_csv


logger = logging.getLogger(__name__)


class MultiVideoPipelineError(RuntimeError):
    """Raised when the multi-video pipeline cannot be initialized or executed."""


class MultiVideoPipeline:
    """Validate and coordinate multiple uploaded videos."""

    def __init__(
        self,
        min_videos: int = 2,
        max_videos: int = 3,
        config: Optional[dict] = None,
    ) -> None:
        if min_videos < 1:
            raise MultiVideoPipelineError(
                "min_videos must be at least 1"
            )

        if max_videos < min_videos:
            raise MultiVideoPipelineError(
                "max_videos must be greater than or equal to min_videos"
            )

        self.min_videos = min_videos
        self.max_videos = max_videos
        self.config = config

    def validate_inputs(
        self,
        videos: Iterable[VideoInput],
    ) -> tuple[VideoInput, ...]:
        """Validate the uploaded video collection."""

        if videos is None:
            raise MultiVideoPipelineError(
                "Video inputs cannot be None"
            )

        video_list = tuple(videos)

        if len(video_list) < self.min_videos:
            raise MultiVideoPipelineError(
                f"At least {self.min_videos} videos are required; "
                f"received {len(video_list)}"
            )

        if len(video_list) > self.max_videos:
            raise MultiVideoPipelineError(
                f"At most {self.max_videos} videos are supported; "
                f"received {len(video_list)}"
            )

        camera_ids: set[str] = set()

        for video in video_list:
            if not isinstance(video, VideoInput):
                raise MultiVideoPipelineError(
                    "All inputs must be VideoInput objects"
                )

            if video.camera_id in camera_ids:
                raise MultiVideoPipelineError(
                    f"Duplicate camera_id: {video.camera_id}"
                )

            camera_ids.add(video.camera_id)

            path = Path(video.video_path)

            if path.suffix.lower() not in VIDEO_EXTENSIONS:
                raise MultiVideoPipelineError(
                    f"Unsupported video format for "
                    f"'{video.video_path}'. Supported formats: "
                    f"{', '.join(sorted(VIDEO_EXTENSIONS))}"
                )

            if not path.exists():
                raise MultiVideoPipelineError(
                    f"Video file does not exist: {video.video_path}"
                )

            if not path.is_file():
                raise MultiVideoPipelineError(
                    f"Video path is not a file: {video.video_path}"
                )

            validation = validate_video(path)

            if validation["status"] != "READABLE":
                raise MultiVideoPipelineError(
                    f"Video is unreadable: {video.video_path}"
                )

        return video_list

    def prepare_analysis(
        self,
        videos: Iterable[VideoInput],
    ) -> MultiVideoAnalysisResult:
        """Validate videos and create an initial analysis result."""

        validated_videos = self.validate_inputs(videos)

        results = tuple(
            VideoAnalysisResult(
                camera_id=video.camera_id,
                video_path=video.video_path,
                status="ready",
            )
            for video in validated_videos
        )

        return MultiVideoAnalysisResult(
            status="ready",
            videos=results,
            unique_global_people=0,
        )

    def _load_pipeline_config(
        self,
        config_path: str | Path = "configs/config.yaml",
    ) -> dict:
        """Load the main application configuration."""

        if self.config is not None:
            return self.config

        return load_config(
            str(config_path),
            apply_env=False,
        )

    @staticmethod
    def _load_reid_config(
        reid_config_path: str | Path = "configs/reid.yaml",
    ) -> dict:
        """Load the dedicated trained Re-ID configuration."""

        config = load_config(
            str(reid_config_path),
            apply_env=False,
        )

        reid_config = config.get("reid", {})

        if not isinstance(reid_config, dict):
            raise MultiVideoPipelineError(
                "The reid section in configs/reid.yaml must be a mapping"
            )

        return reid_config

    @staticmethod
    def _resolve_output_directory(
        configured_directory: str | Path,
        camera_id: str,
    ) -> Path:
        """Resolve the camera-specific output directory."""

        output = Path(configured_directory)

        if not output.is_absolute():
            output = PROJECT_ROOT / output

        return output / camera_id

    @staticmethod
    def _clean_camera_outputs(
        camera_id: str,
        cfg: dict,
        reid_config: dict,
    ) -> None:
        """Remove stale generated outputs for one camera before a new run.

        Only generated outputs belonging to the specified camera are removed.
        Source videos, trained models, configurations, source code, and tests
        are never touched.
        """

        output_directories = [
            MultiVideoPipeline._resolve_output_directory(
                cfg.get("crops", {}).get(
                    "output_dir",
                    "outputs/crops",
                ),
                camera_id,
            ),
            MultiVideoPipeline._resolve_output_directory(
                cfg.get("tracking", {}).get(
                    "output_dir",
                    "outputs/tracking",
                ),
                camera_id,
            ),
            MultiVideoPipeline._resolve_output_directory(
                reid_config.get(
                    "output_dir",
                    "outputs/embeddings",
                ),
                camera_id,
            ),
        ]

        for directory in output_directories:
            if directory.exists():
                if not directory.is_dir():
                    raise MultiVideoPipelineError(
                        f"Expected camera output directory but found "
                        f"a non-directory path: {directory}"
                    )

                logger.info(
                    "Removing stale camera output: %s",
                    directory,
                )

                shutil.rmtree(directory)

    def _process_video(
        self,
        video: VideoInput,
        detector: Detector,
        cfg: dict,
        max_frames: Optional[int] = None,
    ) -> VideoAnalysisResult:
        """Run detection, tracking, and crop extraction for one video."""

        video_path = Path(video.video_path)

        crop_config = dict(cfg.get("crops", {}))

        crop_extractor = CropExtractor(
            crop_config=crop_config,
            detector=detector,
        )

        crop_output = self._resolve_output_directory(
            crop_config.get(
                "output_dir",
                "outputs/crops",
            ),
            video.camera_id,
        )

        crop_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        # IMPORTANT:
        # Every video gets its own ByteTrack instance.
        tracker = Tracker.from_config(cfg)

        metadata = []
        all_tracks: list[TrackedPerson] = []
        track_ids: set[int] = set()

        frames_processed = 0
        crops_generated = 0
        crops_rejected = 0

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            cap.release()

            return VideoAnalysisResult(
                camera_id=video.camera_id,
                video_path=str(video_path),
                status="error",
                output_directory=str(crop_output),
                error="Video could not be opened by OpenCV",
            )

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

        try:
            while max_frames is None or frames_processed < max_frames:
                ok, frame = cap.read()

                if not ok or frame is None:
                    break

                timestamp_seconds = (
                    frames_processed / fps
                    if fps > 0
                    else None
                )

                detections = detector.detect_frame(
                    frame=frame,
                    camera_id=video.camera_id,
                    frame_number=frames_processed,
                    timestamp_seconds=timestamp_seconds,
                )

                tracks = tracker.update(detections)

                # Preserve every ByteTrack record for later
                # temporal and cross-camera analysis.
                all_tracks.extend(tracks)

                for track in tracks:
                    track_ids.add(track.track_id)

                if should_save_frame(
                    frames_processed,
                    crop_extractor.config["save_every_n_frames"],
                ):
                    for track in tracks:
                        item, reason = crop_extractor.extract_track_crop(
                            frame=frame,
                            track=track,
                            source_video=video_path,
                            output_directory=crop_output,
                        )

                        if item is not None:
                            metadata.append(item)
                            crops_generated += 1
                        else:
                            crops_rejected += 1

                            logger.debug(
                                "Rejected crop: camera=%s track=%s "
                                "frame=%s reason=%s",
                                video.camera_id,
                                track.track_id,
                                track.frame_number,
                                reason,
                            )

                frames_processed += 1

        except Exception as exc:
            logger.exception(
                "Pipeline processing failed for camera %s",
                video.camera_id,
            )

            return VideoAnalysisResult(
                camera_id=video.camera_id,
                video_path=str(video_path),
                status="error",
                frames_processed=frames_processed,
                tracks_detected=len(track_ids),
                crops_generated=crops_generated,
                embeddings_generated=0,
                output_directory=str(crop_output),
                error=str(exc),
            )

        finally:
            cap.release()

        # ---------------------------------------------------------
        # Persist ByteTrack tracking metadata.
        # ---------------------------------------------------------
        tracking_config = dict(cfg.get("tracking", {}))

        tracking_output = self._resolve_output_directory(
            tracking_config.get(
                "output_dir",
                "outputs/tracking",
            ),
            video.camera_id,
        )

        tracking_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        tracks_csv = tracking_output / "tracks.csv"

        try:
            write_tracks_csv(
                all_tracks,
                tracks_csv,
            )

        except Exception as exc:
            logger.exception(
                "Could not write tracking metadata for camera %s",
                video.camera_id,
            )

            return VideoAnalysisResult(
                camera_id=video.camera_id,
                video_path=str(video_path),
                status="error",
                frames_processed=frames_processed,
                tracks_detected=len(track_ids),
                crops_generated=crops_generated,
                embeddings_generated=0,
                output_directory=str(crop_output),
                error=f"Could not write tracking metadata: {exc}",
            )

        # ---------------------------------------------------------
        # Persist crop metadata.
        # ---------------------------------------------------------
        crops_csv = crop_output / "crops.csv"

        try:
            write_crops_csv(
                metadata,
                crops_csv,
            )

        except Exception as exc:
            logger.exception(
                "Could not write crop metadata for camera %s",
                video.camera_id,
            )

            return VideoAnalysisResult(
                camera_id=video.camera_id,
                video_path=str(video_path),
                status="error",
                frames_processed=frames_processed,
                tracks_detected=len(track_ids),
                crops_generated=crops_generated,
                embeddings_generated=0,
                output_directory=str(crop_output),
                error=f"Could not write crop metadata: {exc}",
            )

        logger.info(
            "Camera %s preprocessing complete: "
            "frames=%d tracks=%d crops=%d rejected_crops=%d",
            video.camera_id,
            frames_processed,
            len(track_ids),
            crops_generated,
            crops_rejected,
        )

        return VideoAnalysisResult(
            camera_id=video.camera_id,
            video_path=str(video_path),
            status="completed",
            frames_processed=frames_processed,
            tracks_detected=len(track_ids),
            crops_generated=crops_generated,
            embeddings_generated=0,
            output_directory=str(crop_output),
        )

    def _generate_embeddings(
        self,
        video_result: VideoAnalysisResult,
        reid_embedder: ReIDEmbedder,
        reid_config: dict,
        max_images: Optional[int] = None,
    ) -> VideoAnalysisResult:
        """Generate trained OSNet embeddings for one camera's crops."""

        crop_directory = Path(
            video_result.output_directory or ""
        )

        if not crop_directory.is_dir():
            return VideoAnalysisResult(
                camera_id=video_result.camera_id,
                video_path=video_result.video_path,
                status="error",
                frames_processed=video_result.frames_processed,
                tracks_detected=video_result.tracks_detected,
                crops_generated=video_result.crops_generated,
                embeddings_generated=0,
                output_directory=video_result.output_directory,
                error=(
                    "Crop output directory does not exist: "
                    f"{crop_directory}"
                ),
            )

        embedding_root = Path(
            reid_config.get(
                "output_dir",
                "outputs/embeddings",
            )
        )

        if not embedding_root.is_absolute():
            embedding_root = PROJECT_ROOT / embedding_root

        result = reid_embedder.extract_directory(
            input_directory=crop_directory,
            output_dir=embedding_root,
            max_images=max_images,
        )

        if result.status != "OK":
            return VideoAnalysisResult(
                camera_id=video_result.camera_id,
                video_path=video_result.video_path,
                status="error",
                frames_processed=video_result.frames_processed,
                tracks_detected=video_result.tracks_detected,
                crops_generated=video_result.crops_generated,
                embeddings_generated=result.embeddings_generated,
                output_directory=video_result.output_directory,
                error=(
                    "Re-ID embedding extraction failed: "
                    f"{result.error or result.status}"
                ),
            )

        logger.info(
            "Camera %s embeddings complete: "
            "embeddings=%d dimension=%s device=%s",
            video_result.camera_id,
            result.embeddings_generated,
            result.embedding_dimension,
            result.device,
        )

        return VideoAnalysisResult(
            camera_id=video_result.camera_id,
            video_path=video_result.video_path,
            status="completed",
            frames_processed=video_result.frames_processed,
            tracks_detected=video_result.tracks_detected,
            crops_generated=video_result.crops_generated,
            embeddings_generated=result.embeddings_generated,
            output_directory=video_result.output_directory,
        )

    def run(
        self,
        videos: Iterable[VideoInput],
        max_frames: Optional[int] = None,
        config_path: str | Path = "configs/config.yaml",
        reid_config_path: str | Path = "configs/reid.yaml",
    ) -> MultiVideoAnalysisResult:
        """Run multi-video preprocessing and trained OSNet embeddings."""

        if max_frames is not None:
            if (
                not isinstance(max_frames, int)
                or isinstance(max_frames, bool)
                or max_frames < 1
            ):
                raise MultiVideoPipelineError(
                    "max_frames must be a positive integer or None"
                )

        validated_videos = self.validate_inputs(videos)

        cfg = self._load_pipeline_config(config_path)

        detection_config = dict(
            cfg.get("detection", {})
        )

        if detection_config.get("enabled", True) is not True:
            raise MultiVideoPipelineError(
                "Detection is disabled in the pipeline configuration"
            )

        reid_config = self._load_reid_config(
            reid_config_path
        )

        if reid_config.get("enabled", True) is not True:
            raise MultiVideoPipelineError(
                "Re-ID is disabled in configs/reid.yaml"
            )

        detector = Detector.from_config(cfg)

        # Build the Re-ID configuration in the format expected by
        # ReIDEmbedder.from_config().
        reid_cfg = {
            "reid": reid_config,
        }

        reid_embedder = ReIDEmbedder.from_config(
            reid_cfg
        )

        video_results: list[VideoAnalysisResult] = []

        try:
            for video in validated_videos:
                # Remove stale generated data for this camera before
                # starting a new Phase 11 analysis.
                self._clean_camera_outputs(
                    camera_id=video.camera_id,
                    cfg=cfg,
                    reid_config=reid_config,
                )

                preprocessing_result = self._process_video(
                    video=video,
                    detector=detector,
                    cfg=cfg,
                    max_frames=max_frames,
                )

                if preprocessing_result.status != "completed":
                    video_results.append(
                        preprocessing_result
                    )
                    continue

                embedding_result = self._generate_embeddings(
                    video_result=preprocessing_result,
                    reid_embedder=reid_embedder,
                    reid_config=reid_config,
                )

                video_results.append(
                    embedding_result
                )

        finally:
            detector.close()

        failed = [
            result
            for result in video_results
            if result.status != "completed"
        ]

        if failed:
            return MultiVideoAnalysisResult(
                status="error",
                videos=tuple(video_results),
                unique_global_people=0,
                error=(
                    f"{len(failed)} of {len(video_results)} "
                    "video(s) failed during analysis"
                ),
            )

        return MultiVideoAnalysisResult(
            status="completed",
            videos=tuple(video_results),
            unique_global_people=0,
        )

    def analyze(
        self,
        videos: Iterable[VideoInput],
        max_frames: Optional[int] = None,
        config_path: str | Path = "configs/config.yaml",
        reid_config_path: str | Path = "configs/reid.yaml",
    ) -> MultiVideoAnalysisResult:
        """Alias for run()."""

        return self.run(
            videos=videos,
            max_frames=max_frames,
            config_path=config_path,
            reid_config_path=reid_config_path,
        )


__all__ = [
    "MultiVideoPipeline",
    "MultiVideoPipelineError",
]