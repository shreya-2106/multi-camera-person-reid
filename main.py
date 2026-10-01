#!/usr/bin/env python3
"""
Application entry point for the complete multi-camera Re-ID pipeline.

Pipeline:
    video validation
        -> YOLO person detection
        -> ByteTrack tracking
        -> person crops
        -> OSNet Re-ID embeddings
        -> cross-camera matching
        -> temporal/topology evidence
        -> global identity assignment
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python main.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Config, load_config, ConfigError  # noqa: E402
from src.logging_setup import setup_logging, get_logger  # noqa: E402
from src.env_check import run_environment_check  # noqa: E402
from src.pipeline.multi_video_pipeline import (  # noqa: E402
    MultiVideoPipeline,
    MultiVideoPipelineError,
)
from src.pipeline.types import VideoInput  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-Camera Person Re-Identification & Tracking System"
        )
    )

    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the main config YAML.",
    )

    parser.add_argument(
        "--reid-config",
        default="configs/reid.yaml",
        help="Path to the Re-ID config YAML.",
    )

    parser.add_argument(
        "--videos-dir",
        default="data/videos",
        help="Directory containing camera videos.",
    )

    parser.add_argument(
        "--log-level",
        default=None,
        help="Override logging level.",
    )

    return parser.parse_args(argv)


def ensure_directories(cfg: Config) -> None:
    """Create configured directories if they do not exist."""
    paths = cfg.get("paths", {})

    for key in (
        "data_dir",
        "videos_dir",
        "reid_data_dir",
        "models_dir",
        "logs_dir",
        "output_dir",
        "reports_dir",
    ):
        rel_path = paths.get(key)

        if not rel_path:
            continue

        Path(rel_path).mkdir(
            parents=True,
            exist_ok=True,
        )


def bootstrap(
    config_path: str,
    reid_config_path: str,
    log_level_override: str | None = None,
) -> Config:
    """Load configuration and initialize logging."""

    cfg = load_config(
        config_path=config_path,
        extra_config_path=reid_config_path,
    )

    log_cfg = cfg.get("logging", {})

    level = (
        log_level_override
        or log_cfg.get("level", "INFO")
    )

    log_dir = (
        cfg.get("paths", {}).get("logs_dir")
        if log_cfg.get("log_to_file", True)
        else None
    )

    setup_logging(
        level=level,
        log_dir=log_dir,
        log_file=log_cfg.get(
            "log_file",
            "app.log",
        ),
        max_bytes=log_cfg.get(
            "max_bytes",
            5 * 1024 * 1024,
        ),
        backup_count=log_cfg.get(
            "backup_count",
            3,
        ),
    )

    ensure_directories(cfg)

    return cfg


def discover_videos(
    videos_dir: Path,
    logger,
) -> tuple[VideoInput, ...]:
    """
    Discover camera videos from the configured directory.

    Camera IDs are derived from filenames:
        cam01.mp4 -> CAM01
        cam02.mp4 -> CAM02
    """

    supported_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".m4v",
    }

    if not videos_dir.exists():
        raise MultiVideoPipelineError(
            f"Videos directory does not exist: {videos_dir}"
        )

    video_paths = sorted(
        path
        for path in videos_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in supported_extensions
    )

    if len(video_paths) < 2:
        raise MultiVideoPipelineError(
            "At least 2 camera videos are required. "
            f"Found {len(video_paths)} in {videos_dir}"
        )

    if len(video_paths) > 3:
        raise MultiVideoPipelineError(
            "At most 3 camera videos are supported. "
            f"Found {len(video_paths)} in {videos_dir}"
        )

    inputs: list[VideoInput] = []

    for video_path in video_paths:
        camera_id = video_path.stem.upper()

        logger.info(
            "Discovered video: camera=%s path=%s",
            camera_id,
            video_path,
        )

        inputs.append(
            VideoInput(
                video_path=str(video_path),
                camera_id=camera_id,
            )
        )

    return tuple(inputs)


def log_final_result(
    result,
    logger,
) -> None:
    """Log the complete pipeline result."""

    logger.info(
        "Pipeline status: %s",
        result.status,
    )

    for video in result.videos:
        logger.info(
            (
                "Camera %s | status=%s | frames=%d | "
                "tracks=%d | crops=%d | embeddings=%d"
            ),
            video.camera_id,
            video.status,
            video.frames_processed,
            video.tracks_detected,
            video.crops_generated,
            video.embeddings_generated,
        )

        if video.error:
            logger.error(
                "Camera %s error: %s",
                video.camera_id,
                video.error,
            )

    logger.info(
        "Unique global people: %d",
        result.unique_global_people,
    )

    for identity in result.global_identities:
        logger.info(
            (
                "Global Person %d | %s Track %d | "
                "decision=%s | reason=%s"
            ),
            identity.global_person_id,
            identity.camera_id,
            identity.track_id,
            identity.decision,
            identity.assignment_reason,
        )

    if result.error:
        logger.error(
            "Pipeline error: %s",
            result.error,
        )


def save_result(
    result,
    output_path: Path,
    logger,
) -> None:
    """Save the final pipeline result as JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result.to_dict(),
            file,
            indent=2,
        )

    logger.info(
        "Final result saved to: %s",
        output_path,
    )


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        cfg = bootstrap(
            args.config,
            args.reid_config,
            args.log_level,
        )

    except ConfigError as exc:
        print(
            f"[FATAL] Configuration error: {exc}",
            file=sys.stderr,
        )
        return 1

    logger = get_logger("main")

    app_name = cfg.get(
        "app",
        {},
    ).get(
        "name",
        "multi-camera-reid-tracking",
    )

    app_version = cfg.get(
        "app",
        {},
    ).get(
        "version",
        "0.0.0",
    )

    app_env = cfg.get(
        "app",
        {},
    ).get(
        "env",
        "development",
    )

    logger.info(
        "Starting %s v%s [%s]",
        app_name,
        app_version,
        app_env,
    )

    logger.info(
        "Loaded config from '%s' (+ '%s')",
        args.config,
        args.reid_config,
    )

    env_report = run_environment_check(
        required_packages=["yaml"]
    )

    logger.info(
        "Python version: %s (ok=%s)",
        env_report.python_version,
        env_report.python_ok,
    )

    logger.info(
        "Platform: %s",
        env_report.platform_name,
    )

    if env_report.missing_packages:
        logger.warning(
            "Missing required packages: %s",
            ", ".join(env_report.missing_packages),
        )
    else:
        logger.info(
            "All required packages present."
        )

    logger.info(
        "Device: %s",
        env_report.device_info.summary(),
    )

    logger.info(
        "Videos directory: %s",
        Path(args.videos_dir).resolve(),
    )

    if not env_report.ok:
        logger.error(
            "Environment check failed. "
            "Pipeline will not start."
        )
        return 1

    try:
        video_inputs = discover_videos(
            Path(args.videos_dir),
            logger,
        )

        logger.info(
            "Starting complete multi-camera analysis "
            "with %d videos.",
            len(video_inputs),
        )

        pipeline = MultiVideoPipeline(
            config_path=args.config,
            reid_config_path=args.reid_config,
        )

        result = pipeline.run(video_inputs)

        log_final_result(
            result,
            logger,
        )

        result_path = (
            Path("outputs")
            / "reports"
            / "final_result.json"
        )

        save_result(
            result,
            result_path,
            logger,
        )

        if result.status != "completed":
            logger.error(
                "Multi-camera analysis did not complete successfully."
            )
            return 1

        logger.info(
            "Complete multi-camera analysis finished successfully."
        )

        return 0

    except MultiVideoPipelineError as exc:
        logger.error(
            "Pipeline error: %s",
            exc,
        )
        return 1

    except Exception:
        logger.exception(
            "Unexpected error while running "
            "the multi-camera pipeline."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())