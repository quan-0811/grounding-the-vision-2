"""
Config loading utilities.

This project intentionally uses separate small YAML files:

    configs/models/llava15_7b.yaml
    configs/datasets/coco_val2017.yaml
    configs/decoding/greedy.yaml
    configs/phg/default_llava15.yaml

instead of one giant config file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "utils.config requires PyYAML. Install with: pip install pyyaml"
    ) from exc


PathLike = Union[str, Path]


def load_yaml(path: PathLike) -> Dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a dict: {path}")

    return data

def deep_update(
    base: Dict[str, Any],
    override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Recursively update base dict with override dict.

    Example:
        base = {"a": {"b": 1, "c": 2}}
        override = {"a": {"b": 9}}
        result = {"a": {"b": 9, "c": 2}}
    """

    if override is None:
        return dict(base)

    result = dict(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value

    return result

def get_config_section(
    cfg: Dict[str, Any],
    key: str,
    default: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    value = cfg.get(key, default or {})

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise ValueError(f"Config section `{key}` must be a dict.")

    return value