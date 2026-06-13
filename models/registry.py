"""
Model registry.

Used by scripts so you can load models by short name:

    llava15_7b
    qwen2vl_7b
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_model_config(
    model_name: str,
    config_kwargs: Optional[Dict[str, Any]] = None,
):
    config_kwargs = config_kwargs or {}
    name = model_name.lower()

    if name in {"llava15", "llava15_7b", "llava_1_5_7b"}:
        from models.llava15 import Llava15Config

        config_kwargs.setdefault("model_id", "llava-hf/llava-1.5-7b-hf")
        return Llava15Config(**config_kwargs)

    if name in {"qwen2vl", "qwen2vl_7b", "qwen2_vl_7b"}:
        from models.qwen2vl import Qwen2VLConfig

        config_kwargs.setdefault("model_id", "Qwen/Qwen2-VL-7B-Instruct")
        return Qwen2VLConfig(**config_kwargs)

    raise ValueError(f"Unknown model name: {model_name}")


def build_model_wrapper(
    model_name: str,
    config_kwargs: Optional[Dict[str, Any]] = None,
):
    name = model_name.lower()
    config = build_model_config(
        model_name=name,
        config_kwargs=config_kwargs,
    )

    if name in {"llava15", "llava15_7b", "llava_1_5_7b"}:
        from models.llava15 import Llava15Wrapper

        return Llava15Wrapper(config)

    if name in {
        "qwen2vl",
        "qwen2vl_7b",
        "qwen2_vl_7b",
    }:
        from models.qwen2vl import Qwen2VLWrapper

        return Qwen2VLWrapper(config)

    raise ValueError(f"Unknown model name: {model_name}")

def load_model(
    model_name: str,
    config_kwargs: Optional[Dict[str, Any]] = None,
):
    return build_model_wrapper(
        model_name=model_name,
        config_kwargs=config_kwargs,
    ).load()