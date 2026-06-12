# data/amber.py

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


PathLike = Union[str, Path]


DEFAULT_AMBER_PROMPT = "Describe this image."


def _load_json(path: PathLike) -> Any:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _guess_image_path(
    image_dir: PathLike,
    row: Dict[str, Any],
) -> str:
    image_dir = Path(image_dir)

    candidates = []

    for key in [
        "image_path",
        "image",
        "image_name",
        "file_name",
        "filename",
    ]:
        if key in row and row[key] is not None:
            value = str(row[key])

            p = Path(value)

            if p.is_absolute() and p.exists():
                return str(p)

            candidates.append(image_dir / value)

    image_id = row.get("image_id", row.get("id"))

    if image_id is not None:
        image_id = str(image_id)

        candidates.extend(
            [
                image_dir / f"{image_id}.jpg",
                image_dir / f"{image_id}.png",
                image_dir / f"{int(image_id):06d}.jpg"
                if image_id.isdigit()
                else image_dir / f"{image_id}.jpg",
                image_dir / f"{int(image_id):012d}.jpg"
                if image_id.isdigit()
                else image_dir / f"{image_id}.jpg",
            ]
        )

    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        f"Could not resolve AMBER image path for row: {row}"
    )


def _get_prompt(row: Dict[str, Any], default_prompt: str) -> str:
    for key in [
        "prompt",
        "question",
        "query",
        "instruction",
    ]:
        if key in row and row[key]:
            return str(row[key])

    return default_prompt


def _get_id(row: Dict[str, Any], fallback_idx: int) -> int:
    for key in [
        "id",
        "image_id",
        "question_id",
        "sample_id",
    ]:
        if key in row and row[key] is not None:
            try:
                return int(row[key])
            except Exception:
                pass

    return int(fallback_idx)


def load_amber(
    image_dir: PathLike,
    annotation_path: Optional[PathLike] = None,
    max_samples: Optional[int] = None,
    prompt: str = DEFAULT_AMBER_PROMPT,
    seed: int = 42,
    shuffle: bool = False,
    sample_ids: Optional[Sequence[int]] = None,
    skip_missing_images: bool = True,
) -> List[Dict[str, Any]]:
    """
    Flexible AMBER loader.

    Supports annotation JSON formats like:
        [
            {"id": 1, "image": "xxx.jpg", "query": "..."},
            {"image_id": 1, "image_path": "...", "prompt": "..."}
        ]

    If annotation_path is None, this scans image_dir for jpg/png files.

    Returned sample format:
        {
            "id": ...,
            "image_id": ...,
            "image_path": "...",
            "prompt": "...",
            "objects": [],
            "gt_objects": [],
        }
    """

    image_dir = Path(image_dir)

    allowed_ids = None

    if sample_ids is not None:
        allowed_ids = set(int(x) for x in sample_ids)

    rows: List[Dict[str, Any]] = []

    if annotation_path is None:
        image_files = sorted(
            list(image_dir.glob("*.jpg"))
            + list(image_dir.glob("*.jpeg"))
            + list(image_dir.glob("*.png"))
        )

        for idx, image_path in enumerate(image_files):
            sample_id = idx

            if allowed_ids is not None and sample_id not in allowed_ids:
                continue

            rows.append(
                {
                    "id": sample_id,
                    "image_id": sample_id,
                    "image_path": str(image_path),
                    "prompt": prompt,
                    "objects": [],
                    "gt_objects": [],
                }
            )

    else:
        data = _load_json(annotation_path)

        if isinstance(data, dict):
            for key in ["data", "samples", "annotations", "questions"]:
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break

        if not isinstance(data, list):
            raise ValueError(
                f"AMBER annotation_path must contain a list or dict-with-list: {annotation_path}"
            )

        for idx, row in enumerate(data):
            if not isinstance(row, dict):
                continue

            sample_id = _get_id(row, fallback_idx=idx)

            if allowed_ids is not None and sample_id not in allowed_ids:
                continue

            try:
                image_path = _guess_image_path(
                    image_dir=image_dir,
                    row=row,
                )
            except FileNotFoundError:
                if skip_missing_images:
                    continue
                raise

            sample_prompt = _get_prompt(row, default_prompt=prompt)

            rows.append(
                {
                    "id": sample_id,
                    "image_id": int(row.get("image_id", sample_id))
                    if str(row.get("image_id", sample_id)).isdigit()
                    else sample_id,
                    "image_path": image_path,
                    "prompt": sample_prompt,
                    "objects": row.get("objects", row.get("gt_objects", [])),
                    "gt_objects": row.get("gt_objects", row.get("objects", [])),
                    "raw": row,
                }
            )

    rows = sorted(rows, key=lambda x: int(x["id"]))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)

    if max_samples is not None:
        rows = rows[: int(max_samples)]

    return rows