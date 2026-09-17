"""Sanity tests that the Phase 1 project structure is in place."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_DIRS = [
    "data",
    "data/videos",
    "data/reid",
    "models",
    "src",
    "src/detection",
    "src/tracking",
    "src/reid",
    "src/matching",
    "src/topology",
    "src/temporal",
    "src/trajectory",
    "src/database",
    "src/analytics",
    "training",
    "dashboard",
    "configs",
    "tests",
    "notebooks",
]

EXPECTED_FILES = [
    "requirements.txt",
    ".gitignore",
    "README.md",
    "configs/config.yaml",
    "configs/reid.yaml",
    "main.py",
    "src/__init__.py",
    "src/config.py",
    "src/logging_setup.py",
    "src/env_check.py",
]

EXPECTED_PACKAGE_INITS = [
    "src/detection/__init__.py",
    "src/tracking/__init__.py",
    "src/reid/__init__.py",
    "src/matching/__init__.py",
    "src/topology/__init__.py",
    "src/temporal/__init__.py",
    "src/trajectory/__init__.py",
    "src/database/__init__.py",
    "src/analytics/__init__.py",
]


def test_expected_directories_exist():
    for rel_dir in EXPECTED_DIRS:
        d = PROJECT_ROOT / rel_dir
        assert d.is_dir(), f"Expected directory missing: {rel_dir}"


def test_expected_files_exist():
    for rel_file in EXPECTED_FILES:
        f = PROJECT_ROOT / rel_file
        assert f.is_file(), f"Expected file missing: {rel_file}"


def test_expected_package_inits_exist():
    for rel_file in EXPECTED_PACKAGE_INITS:
        f = PROJECT_ROOT / rel_file
        assert f.is_file(), f"Expected __init__.py missing: {rel_file}"


def test_src_is_importable_package():
    import src  # noqa: F401
    import src.detection  # noqa: F401
    import src.tracking  # noqa: F401
    import src.reid  # noqa: F401
    import src.matching  # noqa: F401
    import src.topology  # noqa: F401
    import src.temporal  # noqa: F401
    import src.trajectory  # noqa: F401
    import src.database  # noqa: F401
    import src.analytics  # noqa: F401
