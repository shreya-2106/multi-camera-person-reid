"""Tests that the Phase 3 `detection:` config section is present and sane
in the project's real configs/config.yaml (not just in synthetic configs)."""

from src.config import load_config
from src.detection.detector import Detector


def test_real_config_has_detection_section():
    cfg = load_config(apply_env=False)
    assert "detection" in cfg
    det_cfg = cfg.detection
    assert det_cfg.get("model")
    assert 0.0 < det_cfg.get("confidence", 0) <= 1.0
    assert "person" in det_cfg.get("classes", [])


def test_detector_from_real_config_builds_without_loading_model():
    cfg = load_config(apply_env=False)
    detector = Detector.from_config(cfg)
    assert detector.model_name_or_path
    assert detector.device in ("cpu", "cuda")
    # constructing a Detector must not trigger model loading / network calls
    assert detector._model is None
