"""
Configuration loader.

Loads YAML configuration files and exposes them as attribute-accessible,
dict-like objects. Supports:
    - Loading a single YAML file
    - Merging a secondary config (e.g. reid.yaml) on top of / alongside a base config
    - Environment variable overrides of the form APP__SECTION__KEY=value
    - Simple path resolution relative to the project root

This module intentionally has no dependency on any of the (not-yet-implemented)
detection / tracking / reid modules -- it is pure infrastructure.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Root of the project (two levels up from this file: src/config.py -> project/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(Exception):
    """Raised when configuration loading or validation fails."""


class Config(dict):
    """
    A dict subclass that also allows attribute-style access.

    Example:
        cfg = Config({"a": {"b": 1}})
        cfg.a.b == 1
        cfg["a"]["b"] == 1
    """

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(
                f"No config key '{item}' found"
            ) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, key: str, default: Optional[str] = None) -> Path:
        """
        Fetch a config value that represents a path and resolve it relative
        to the project root if it is not already absolute.
        """
        value = self.get(key, default)
        if value is None:
            raise ConfigError(f"Path config key '{key}' not found and no default given")
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into `base`, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _apply_env_overrides(cfg: Dict[str, Any], prefix: str = "APP") -> Dict[str, Any]:
    """
    Apply environment variable overrides of the form:
        APP__SECTION__KEY=value
    to the given config dict. Values are parsed with yaml.safe_load so that
    ints/floats/bools/lists come through with the right type.
    """
    result = copy.deepcopy(cfg)
    env_prefix = f"{prefix}__"
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(env_prefix):
            continue
        parts = env_key[len(env_prefix):].split("__")
        if not parts or parts == [""]:
            continue
        node = result
        for part in parts[:-1]:
            key = part.lower()
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        try:
            parsed_value = yaml.safe_load(env_value)
        except yaml.YAMLError:
            parsed_value = env_value
        node[parts[-1].lower()] = parsed_value
    return result


def load_yaml(path: os.PathLike) -> Dict[str, Any]:
    """Load a single YAML file into a plain dict. Returns {} if file is empty."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a mapping at the top level")
    return data


def load_config(
    config_path: Optional[os.PathLike] = None,
    extra_config_path: Optional[os.PathLike] = None,
    apply_env: bool = True,
) -> Config:
    """
    Load the main application configuration, optionally merging an extra
    config file (e.g. reid.yaml) on top, then applying environment
    variable overrides.

    Args:
        config_path: path to the main config YAML (default: configs/config.yaml)
        extra_config_path: path to an additional config YAML to merge in
        apply_env: whether to apply APP__SECTION__KEY env var overrides

    Returns:
        Config object (attribute + dict accessible)
    """
    config_path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "config.yaml"
    merged = load_yaml(config_path)

    if extra_config_path is not None:
        extra_path = Path(extra_config_path)
        if extra_path.exists():
            extra = load_yaml(extra_path)
            merged = _deep_merge(merged, extra)

    if apply_env:
        merged = _apply_env_overrides(merged)

    return Config(merged)


def load_reid_config(reid_config_path: Optional[os.PathLike] = None) -> Config:
    """Load configs/reid.yaml on its own (useful once Phase 3 lands)."""
    reid_config_path = (
        Path(reid_config_path) if reid_config_path else PROJECT_ROOT / "configs" / "reid.yaml"
    )
    return Config(load_yaml(reid_config_path))
