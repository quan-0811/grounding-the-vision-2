# data/registry.py

from __future__ import annotations

from typing import Any, Dict, Optional


SUPPORTED_DATASETS = {
    "coco_val2017",
    "coco",
    "coco2017",
    "amber",
}


def load_dataset(
    dataset_name: str,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
):
    dataset_kwargs = dataset_kwargs or {}
    name = dataset_name.lower()

    if name in {"coco", "coco_val2017", "coco2017"}:
        from data.coco import load_coco_val2017

        return load_coco_val2017(**dataset_kwargs)

    if name == "amber":
        from data.amber import load_amber

        return load_amber(**dataset_kwargs)

    raise ValueError(
        f"Unknown dataset name: {dataset_name}. "
        f"Supported datasets: {sorted(SUPPORTED_DATASETS)}"
    )