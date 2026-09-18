"""Run Phase 2 Re-ID dataset and CCTV video validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import ConfigError, load_config
from .dataset_validator import validate_datasets
from .video_validator import validate_videos


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    try:
        cfg = load_config(extra_config_path="configs/reid.yaml")
        datasets = validate_datasets(cfg.paths.get_path("reid_data_dir"))
        videos = validate_videos(cfg.paths.get_path("videos_dir"))
        reports_root = cfg.paths.get_path("reports_dir")
    except ConfigError as error:
        print(f"Configuration error: {error}")
        return 2
    write_report(reports_root / "dataset_report.json", datasets)
    write_report(reports_root / "video_report.json", videos)
    for name, dataset in datasets["datasets"].items():
        print(f"{name}: {dataset['status']}")
    print(f"videos: {videos['status']} ({len(videos['videos'])} found)")
    print(f"Reports written to {reports_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
