# data/coco.py

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from utils.io import load_json

PathLike = Union[str, Path]


DEFAULT_CAPTION_PROMPT = "Describe this image."

def _resolve_coco_image_path(
    image_dir: PathLike,
    file_name: str,
) -> str:
    return str(Path(image_dir) / file_name)


def _build_image_to_objects(
    annotations: Sequence[Dict[str, Any]],
    categories: Sequence[Dict[str, Any]],
) -> Dict[int, List[str]]:
    cat_id_to_name = {
        int(cat["id"]): str(cat["name"])
        for cat in categories
    }

    image_to_objects: Dict[int, set[str]] = {}

    for ann in annotations:
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])

        obj_name = cat_id_to_name.get(category_id)

        if obj_name is None:
            continue

        image_to_objects.setdefault(image_id, set()).add(obj_name)

    return {
        image_id: sorted(list(objects))
        for image_id, objects in image_to_objects.items()
    }


def load_coco_val2017(
    image_dir: PathLike,
    annotation_path: PathLike,
    max_samples: Optional[int] = None,
    prompt: str = DEFAULT_CAPTION_PROMPT,
    seed: int = 42,
    shuffle: bool = False,
    sample_ids: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Load COCO val2017 samples.

    Expected annotation file:
        instances_val2017.json

    Returned sample format:
        {
            "id": image_id,
            "image_id": image_id,
            "image_path": ".../000000xxxxxx.jpg",
            "file_name": "000000xxxxxx.jpg",
            "prompt": "Describe this image.",
            "objects": [...],
            "gt_objects": [...],
        }
    """

    image_dir = Path(image_dir)
    data = load_json(annotation_path)

    images = data["images"]
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    image_to_objects = _build_image_to_objects(
        annotations=annotations,
        categories=categories,
    )

    rows: List[Dict[str, Any]] = []

    allowed_ids = None

    if sample_ids is not None:
        allowed_ids = set(int(x) for x in sample_ids)

    for image_info in images:
        image_id = int(image_info["id"])

        if allowed_ids is not None and image_id not in allowed_ids:
            continue

        file_name = str(image_info["file_name"])
        image_path = _resolve_coco_image_path(
            image_dir=image_dir,
            file_name=file_name,
        )

        if not Path(image_path).exists():
            continue

        objects = image_to_objects.get(image_id, [])

        rows.append(
            {
                "id": image_id,
                "image_id": image_id,
                "image_path": image_path,
                "file_name": file_name,
                "prompt": prompt,
                "objects": objects,
                "gt_objects": objects,
            }
        )

    rows = sorted(rows, key=lambda x: int(x["id"]))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)

    if max_samples is not None:
        rows = rows[: int(max_samples)]

    return rows