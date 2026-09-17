"""Unit tests for src.env_check."""

from src.env_check import (
    check_python_version,
    detect_device,
    check_required_packages,
    run_environment_check,
    DeviceInfo,
    EnvironmentReport,
)


def test_check_python_version_passes_for_low_minimum():
    assert check_python_version((3, 0)) is True


def test_check_python_version_fails_for_unreasonably_high_minimum():
    assert check_python_version((99, 0)) is False


def test_detect_device_returns_device_info():
    info = detect_device()
    assert isinstance(info, DeviceInfo)
    assert info.device in ("cpu", "cuda")
    # torch may or may not be installed in this environment; either is valid.
    assert isinstance(info.torch_available, bool)


def test_detect_device_summary_is_non_empty_string():
    info = detect_device()
    summary = info.summary()
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_check_required_packages_detects_present_package():
    missing = check_required_packages(["yaml"])  # PyYAML is a Phase 1 dependency
    assert missing == []


def test_check_required_packages_detects_missing_package():
    missing = check_required_packages(["this_package_does_not_exist_12345"])
    assert missing == ["this_package_does_not_exist_12345"]


def test_run_environment_check_returns_report():
    report = run_environment_check(required_packages=["yaml"])
    assert isinstance(report, EnvironmentReport)
    assert report.python_ok is True
    assert report.missing_packages == []
    assert report.device_info is not None
    assert report.ok is True


def test_run_environment_check_flags_missing_package():
    report = run_environment_check(required_packages=["this_package_does_not_exist_12345"])
    assert report.ok is False
    assert "this_package_does_not_exist_12345" in report.missing_packages
