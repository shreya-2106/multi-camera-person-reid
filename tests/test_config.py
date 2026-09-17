"""Unit tests for src.config."""

import os
import textwrap

import pytest

from src.config import (
    Config,
    ConfigError,
    load_yaml,
    load_config,
    load_reid_config,
    _deep_merge,
    _apply_env_overrides,
)


def write_yaml(path, content: str):
    path.write_text(textwrap.dedent(content))
    return path


def test_config_attribute_access():
    cfg = Config({"a": {"b": 1, "c": [1, 2, 3]}})
    assert cfg.a.b == 1
    assert cfg["a"]["b"] == 1
    assert cfg.a.c == [1, 2, 3]


def test_config_missing_attribute_raises():
    cfg = Config({"a": 1})
    with pytest.raises(AttributeError):
        _ = cfg.nonexistent


def test_config_set_attribute():
    cfg = Config()
    cfg.foo = "bar"
    assert cfg["foo"] == "bar"


def test_load_yaml_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(ConfigError):
        load_yaml(missing)


def test_load_yaml_empty_file(tmp_path):
    empty = write_yaml(tmp_path / "empty.yaml", "")
    assert load_yaml(empty) == {}


def test_load_yaml_non_mapping_raises(tmp_path):
    bad = write_yaml(tmp_path / "bad.yaml", "- 1\n- 2\n")
    with pytest.raises(ConfigError):
        load_yaml(bad)


def test_load_yaml_basic(tmp_path):
    f = write_yaml(
        tmp_path / "basic.yaml",
        """
        app:
          name: test-app
        """,
    )
    data = load_yaml(f)
    assert data["app"]["name"] == "test-app"


def test_deep_merge_overrides_nested_keys():
    base = {"a": {"b": 1, "c": 2}, "d": 4}
    override = {"a": {"b": 99}}
    merged = _deep_merge(base, override)
    assert merged == {"a": {"b": 99, "c": 2}, "d": 4}
    # base should be untouched
    assert base["a"]["b"] == 1


def test_deep_merge_adds_new_keys():
    base = {"a": 1}
    override = {"b": 2}
    merged = _deep_merge(base, override)
    assert merged == {"a": 1, "b": 2}


def test_apply_env_overrides(monkeypatch):
    base = {"logging": {"level": "INFO"}}
    monkeypatch.setenv("APP__LOGGING__LEVEL", "DEBUG")
    result = _apply_env_overrides(base)
    assert result["logging"]["level"] == "DEBUG"


def test_apply_env_overrides_creates_new_section(monkeypatch):
    base = {}
    monkeypatch.setenv("APP__NEWSECTION__KEY", "42")
    result = _apply_env_overrides(base)
    assert result["newsection"]["key"] == 42  # yaml.safe_load parses "42" as int


def test_load_config_merges_extra_and_applies_env(tmp_path, monkeypatch):
    main_cfg = write_yaml(
        tmp_path / "config.yaml",
        """
        app:
          name: main-app
        logging:
          level: INFO
        """,
    )
    extra_cfg = write_yaml(
        tmp_path / "reid.yaml",
        """
        reid:
          model_name: osnet_x1_0
        """,
    )
    monkeypatch.setenv("APP__LOGGING__LEVEL", "WARNING")

    cfg = load_config(config_path=main_cfg, extra_config_path=extra_cfg)

    assert cfg.app.name == "main-app"
    assert cfg.reid.model_name == "osnet_x1_0"
    assert cfg.logging.level == "WARNING"


def test_load_config_extra_config_optional(tmp_path):
    main_cfg = write_yaml(
        tmp_path / "config.yaml",
        """
        app:
          name: main-app
        """,
    )
    nonexistent_extra = tmp_path / "does_not_exist.yaml"
    cfg = load_config(config_path=main_cfg, extra_config_path=nonexistent_extra, apply_env=False)
    assert cfg.app.name == "main-app"


def test_get_path_relative_to_project_root(tmp_path):
    main_cfg = write_yaml(
        tmp_path / "config.yaml",
        """
        paths:
          videos_dir: data/videos
        """,
    )
    cfg = load_config(config_path=main_cfg, extra_config_path=None, apply_env=False)
    # nested access pattern used in main.py / Config.get_path:
    videos_path = cfg["paths"]["videos_dir"]
    assert videos_path == "data/videos"


def test_nested_get_path_resolves_relative_to_project_root(tmp_path):
    """Regression test: cfg.paths.get_path(...) must resolve relative to PROJECT_ROOT,
    mirroring the pattern used in main.py."""
    main_cfg = write_yaml(
        tmp_path / "config.yaml",
        """
        paths:
          videos_dir: data/videos
        """,
    )
    cfg = load_config(config_path=main_cfg, extra_config_path=None, apply_env=False)
    resolved = cfg.paths.get_path("videos_dir")
    assert resolved.parts[-2:] == ("data", "videos")
    assert resolved.is_absolute()


def test_load_reid_config(tmp_path):
    reid_cfg = write_yaml(
        tmp_path / "reid.yaml",
        """
        reid:
          embedding_dim: 512
        """,
    )
    cfg = load_reid_config(reid_cfg)
    assert cfg.reid.embedding_dim == 512


def test_real_project_configs_load():
    """Sanity check that the actual configs/config.yaml and configs/reid.yaml load."""
    cfg = load_config(apply_env=False)
    assert "app" in cfg
    assert "paths" in cfg
    assert "reid" in cfg  # merged in from configs/reid.yaml
