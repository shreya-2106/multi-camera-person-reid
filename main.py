#!/usr/bin/env python3
"""
Application entry point.

Phase 1 scope only:
    - Load configuration (configs/config.yaml [+ configs/reid.yaml])
    - Set up logging
    - Run environment checks (Python version, required packages)
    - Detect CPU/CUDA device
    - Ensure configured data directories exist
    - Print a startup summary

Detection / tracking / re-id / matching / dashboard are NOT started here yet;
those hook in during later phases.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python main.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Config, load_config, ConfigError  # noqa: E402
from src.logging_setup import setup_logging, get_logger  # noqa: E402
from src.env_check import run_environment_check  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Camera Person Re-Identification & Tracking System (Phase 1: setup only)"
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the main config YAML (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--reid-config",
        default="configs/reid.yaml",
        help="Path to the reid config YAML to merge in (default: configs/reid.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override the log level from config (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args(argv)


def ensure_directories(cfg: Config) -> None:
    """Create configured data/log/output directories if they don't exist."""
    paths = cfg.get("paths", {})
    for key in ("data_dir", "videos_dir", "reid_data_dir", "models_dir", "logs_dir", "output_dir", "reports_dir"):
        rel_path = paths.get(key)
        if not rel_path:
            continue
        Path(rel_path).mkdir(parents=True, exist_ok=True)


def bootstrap(config_path: str, reid_config_path: str, log_level_override: str = None) -> Config:
    """Load config, set up logging, and return the merged Config object."""
    cfg = load_config(config_path=config_path, extra_config_path=reid_config_path)

    log_cfg = cfg.get("logging", {})
    level = log_level_override or log_cfg.get("level", "INFO")
    log_dir = cfg.get("paths", {}).get("logs_dir") if log_cfg.get("log_to_file", True) else None

    setup_logging(
        level=level,
        log_dir=log_dir,
        log_file=log_cfg.get("log_file", "app.log"),
        max_bytes=log_cfg.get("max_bytes", 5 * 1024 * 1024),
        backup_count=log_cfg.get("backup_count", 3),
    )

    ensure_directories(cfg)
    return cfg


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        cfg = bootstrap(args.config, args.reid_config, args.log_level)
    except ConfigError as exc:
        print(f"[FATAL] Configuration error: {exc}", file=sys.stderr)
        return 1

    logger = get_logger("main")

    app_name = cfg.get("app", {}).get("name", "app")
    app_version = cfg.get("app", {}).get("version", "0.0.0")
    app_env = cfg.get("app", {}).get("env", "development")

    logger.info("Starting %s v%s [%s]", app_name, app_version, app_env)
    logger.info("Loaded config from '%s' (+ '%s')", args.config, args.reid_config)

    env_report = run_environment_check(required_packages=["yaml"])

    logger.info("Python version: %s (ok=%s)", env_report.python_version, env_report.python_ok)
    logger.info("Platform: %s", env_report.platform_name)
    if env_report.missing_packages:
        logger.warning("Missing required packages: %s", ", ".join(env_report.missing_packages))
    else:
        logger.info("All required packages present.")

    logger.info("Device: %s", env_report.device_info.summary())

    logger.info(
        "Data directories ready: videos=%s reid=%s models=%s",
        cfg.paths.get_path("videos_dir"),
        cfg.paths.get_path("reid_data_dir"),
        cfg.paths.get_path("models_dir"),
    )

    logger.info(
        "Phase 1 setup complete. Detection/Tracking/ReID/Matching/Dashboard "
        "are disabled placeholders until their respective phases are implemented."
    )

    if not env_report.ok:
        logger.error("Environment check failed. See warnings above.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
