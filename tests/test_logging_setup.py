"""Unit tests for src.logging_setup."""

import logging

import src.logging_setup as logging_setup
from src.logging_setup import setup_logging, get_logger


def reset_logging_state():
    """Helper to reset the module-level configured flag and root handlers between tests."""
    logging_setup._CONFIGURED = False
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)


def test_setup_logging_configures_console_handler():
    reset_logging_state()
    logger = setup_logging(level="DEBUG", log_dir=None, force=True)
    assert logger.level == logging.DEBUG
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


def test_setup_logging_is_idempotent_without_force():
    reset_logging_state()
    logger1 = setup_logging(level="INFO", log_dir=None)
    handler_count_1 = len(logger1.handlers)
    logger2 = setup_logging(level="DEBUG", log_dir=None)  # should be a no-op
    handler_count_2 = len(logger2.handlers)
    assert handler_count_1 == handler_count_2
    assert logger2.level == logging.INFO  # unchanged, since second call was skipped


def test_setup_logging_creates_file_handler(tmp_path):
    reset_logging_state()
    log_dir = tmp_path / "logs"
    setup_logging(level="INFO", log_dir=str(log_dir), log_file="test.log", force=True)
    log_file_path = log_dir / "test.log"
    logger = get_logger("test_logger")
    logger.info("hello world")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert log_file_path.exists()
    content = log_file_path.read_text()
    assert "hello world" in content


def test_get_logger_returns_named_logger():
    logger = get_logger("my.module")
    assert logger.name == "my.module"
