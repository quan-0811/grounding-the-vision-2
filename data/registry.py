# data/registry.py

from __future__ import annotations

from typing import Any, Dict, Optional


def load_dataset(
    dataset_name: str,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Dataset registry.

    Supported names:
        coco_val2017
        amber
        visual_genome
        vg
    """

    dataset_kwargs = dataset_kwargs or {}
    name = dataset_name.lower()

    if name in {"coco", "coco_val2017", "coco2017"}:
        from data.coco import load_coco_val2017

        return load_coco_val2017(**dataset_kwargs)

    if name in {"amber"}:
        from data.amber import load_amber

        return load_amber(**dataset_kwargs)

    if name in {"visual_genome", "vg"}:
        from data.visual_genome import load_visual_genome

        return load_visual_genome(**dataset_kwargs)

    raise ValueError(f"Unknown dataset name: {dataset_name}")