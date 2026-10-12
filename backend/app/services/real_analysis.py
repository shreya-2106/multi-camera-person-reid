from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Sequence

import yaml

from src.pipeline.multi_video_pipeline import MultiVideoPipeline
from src.pipeline.types import MultiVideoAnalysisResult, VideoInput

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
REID_CONFIG_PATH = PROJECT_ROOT / "configs" / "reid.yaml"

ANALYSIS_ROOT = PROJECT_ROOT / "backend" / "storage" / "analyses"

_PIPELINE_LOCK = threading.Lock()


def _prepare_config_files(
    workspace: Path,
) -> tuple[Path, Path]:
    """Create pipeline configs whose model paths remain valid."""

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle) or {}

    with REID_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        reid_config = yaml.safe_load(handle) or {}

    detection = config.get("detection", {})

    model_path = detection.get("model")
    if model_path:
        detection["model"] = str(
            (PROJECT_ROOT / model_path).resolve()
        )

    config["detection"] = detection

    reid_section = reid_config.get("reid", {})

    checkpoint = reid_section.get("checkpoint")
    if checkpoint:
        reid_section["checkpoint"] = str(
            (PROJECT_ROOT / checkpoint).resolve()
        )

    reid_config["reid"] = reid_section

    config_path = workspace / "config.yaml"
    reid_config_path = workspace / "reid.yaml"

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        yaml.safe_dump(
            config,
            handle,
            sort_keys=False,
        )

    with reid_config_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        yaml.safe_dump(
            reid_config,
            handle,
            sort_keys=False,
        )

    return config_path, reid_config_path


def run_real_analysis(
    *,
    video_paths: Sequence[Path],
    camera_ids: Sequence[str],
    analysis_id: int | str = "local",
) -> MultiVideoAnalysisResult:
    """Run the existing Phase 1-14 ML pipeline in an isolated workspace."""

    if len(video_paths) != len(camera_ids):
        raise ValueError(
            "video_paths and camera_ids must contain the same number of items."
        )

    if not 2 <= len(video_paths) <= 3:
        raise ValueError(
            "Real multi-camera analysis requires 2 to 3 videos."
        )

    config_started = time.perf_counter()
    workspace = (
        ANALYSIS_ROOT
        / f"search_{analysis_id}"
    )

    workspace.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_path, reid_config_path = _prepare_config_files(
        workspace
    )
    logger.info(
        "analysis_stage_timing stage=config_preparation seconds=%.3f",
        time.perf_counter() - config_started,
    )

    inputs = tuple(
        VideoInput(
            video_path=str(Path(video_path).resolve()),
            camera_id=camera_id,
        )
        for video_path, camera_id in zip(
            video_paths,
            camera_ids,
        )
    )

    original_cwd = Path.cwd()
    pipeline_init_started = time.perf_counter()
    pipeline = MultiVideoPipeline(
        config_path=config_path,
        reid_config_path=reid_config_path,
    )
    logger.info(
        "analysis_stage_timing stage=pipeline_initialization seconds=%.3f",
        time.perf_counter() - pipeline_init_started,
    )
    lock_started = time.perf_counter()

    with _PIPELINE_LOCK:
        try:
            logger.info(
                "analysis_stage_timing stage=pipeline_lock_wait seconds=%.3f",
                time.perf_counter() - lock_started,
            )
            os.chdir(workspace)
            pipeline_started = time.perf_counter()
            result = pipeline.run(inputs)
            logger.info(
                "analysis_stage_timing stage=real_analysis_total seconds=%.3f",
                time.perf_counter() - pipeline_started,
            )
            return result
        finally:
            os.chdir(original_cwd)
