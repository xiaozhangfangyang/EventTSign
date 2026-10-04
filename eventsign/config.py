from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from .paths import resolve_project_path


class Config(SimpleNamespace):
    def to_dict(self) -> dict[str, Any]:
        return {key: _to_dict(value) for key, value in vars(self).items()}


def _to_namespace(value: Any):
    if isinstance(value, dict):
        return Config(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _to_dict(value: Any):
    if isinstance(value, SimpleNamespace):
        return {key: _to_dict(item) for key, item in vars(value).items()}
    if isinstance(value, (list, tuple)):
        return [_to_dict(item) for item in value]
    return value


def config_from_dict(raw: dict[str, Any]) -> Config:
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a mapping.")
    for section in ("data", "model", "train"):
        if not isinstance(raw.get(section), dict):
            raise ValueError(f"Configuration requires a '{section}' mapping.")
    return _to_namespace(raw)


def load_config(path: str | Path) -> Config:
    with resolve_project_path(path).open("r", encoding="utf-8") as handle:
        return config_from_dict(yaml.safe_load(handle))
