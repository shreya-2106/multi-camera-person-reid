"""
Logging system.

Provides a single `setup_logging()` entry point that configures the root
logger (console handler + optional rotating file handler) based on values
from the loaded config, and a `get_logger()` helper for modules to fetch
a named logger.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

_CONFIGURED = False

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: str = "app.log",
    fmt: str = DEFAULT_FORMAT,
    datefmt: str = DEFAULT_DATEFMT,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
    force: bool = False,
) -> logging.Logger:
    """
    Configure the root logger with a console handler, and a rotating file
    handler if `log_dir` is provided.

    Safe to call multiple times; only configures once unless `force=True`.
    """
    global _CONFIGURED

    root_logger = logging.getLogger()

    if _CONFIGURED and not force:
        return root_logger

    level_value = getattr(logging, str(level).upper(), logging.INFO)
    root_logger.setLevel(level_value)

    # Clear existing handlers if we're re-configuring
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level_value)
    root_logger.addHandler(console_handler)

    if log_dir:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        file_path = log_dir_path / log_file
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(file_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level_value)
        root_logger.addHandler(file_handler)

    _CONFIGURED = True
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Call setup_logging() first for full config."""
    return logging.getLogger(name)
