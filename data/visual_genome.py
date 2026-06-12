# data/visual_genome.py

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


PathLike = Union[str, Path]


DEFAULT_VG_PROMPT = "Describe this image."


def _load_json(path: PathLike) -> Any:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_vg_image_path(
    vg_root: PathLike,
    image_id: int,
) -> str:
    """
    Resolve Visual Genome image path.

    Expected folders:
        data/visual_genome/images/VG_100K/
        data/visual_genome/images2/VG_100K_2/

    Also supports:
        data/visual_genome/VG_100K/
        data/visual_genome/VG_100K_2/
    """

    vg_root = Path(vg_root)
    image_id = int(image_id)

    candidates = [
        vg_root / "images" / "VG_100K" / f"{image_id}.jpg",
        vg_root / "images2" / "VG_100K_2" / f"{image_id}.jpg",
        vg_root / "VG_100K" / f"{image_id}.jpg",
        vg_root / "VG_100K_2" / f"{image_id}.jpg",
        vg_root / "images" / f"{image_id}.jpg",
        vg_root / "images2" / f"{image_id}.jpg",
        vg_root / f"{image_id}.jpg",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        f"Could not find Visual Genome image for image_id={image_id} "
        f"under root={vg_root}"
    )


def _extract_object_names(row: Dict[str, Any]) -> List[str]:
    """
    VG objects.json usually contains:

        {
            "image_id": ...,
            "objects": [
                {"names": ["man"], ...},
                {"name": "dog", ...}
            ]
        }
    """

    objects = row.get("objects", [])
    names = set()

    for obj in objects:
        if not isinstance(obj, dict):
            continue

        if "names" in obj and isinstance(obj["names"], list):
            for name in obj["names"]:
                if name:
                    names.add(str(name).lower().strip())

        elif "name" in obj:
            name = obj["name"]

            if name:
                names.add(str(name).lower().strip())

    return sorted(names)


def load_visual_genome(
    vg_root: PathLike,
    objects_path: Optional[PathLike] = None,
    max_samples: Optional[int] = None,
    prompt: str = DEFAULT_VG_PROMPT,
    seed: int = 42,
    shuffle: bool = False,
    sample_ids: Optional[Sequence[int]] = None,
    skip_missing_images: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load Visual Genome samples from objects.json.

    Default objects path:
        {vg_root}/VisualGenome_task/objects.json

    Returned sample format:
        {
            "id": image_id,
            "image_id": image_id,
            "image_path": ".../VG_100K/xxx.jpg",
            "prompt": "Describe this image.",
            "objects": [...],
            "gt_objects": [...],
        }
    """

    vg_root = Path(vg_root)

    if objects_path is None:
        objects_path = vg_root / "VisualGenome_task" / "objects.json"

    data = _load_json(objects_path)

    allowed_ids = None

    if sample_ids is not None:
        allowed_ids = set(int(x) for x in sample_ids)

    rows: List[Dict[str, Any]] = []

    for row in data:
        image_id = int(row["image_id"])

        if allowed_ids is not None and image_id not in allowed_ids:
            continue

        try:
            image_path = resolve_vg_image_path(
                vg_root=vg_root,
                image_id=image_id,
            )
        except FileNotFoundError:
            if skip_missing_images:
                continue
            raise

        objects = _extract_object_names(row)

        rows.append(
            {
                "id": image_id,
                "image_id": image_id,
                "image_path": image_path,
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