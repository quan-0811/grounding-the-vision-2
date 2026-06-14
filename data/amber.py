# data/amber.py

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from utils.io import load_json
from utils.dict_utils import first_existing


PathLike = Union[str, Path]

DEFAULT_AMBER_PROMPT = "Describe this image."

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def _natural_key(value: Any) -> list[Any]:
    text = str(value)

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None

def find_amber_images(
    root: PathLike = "data/amber",
    image_dir: Optional[PathLike] = None,
) -> List[Path]:
    """
    Recursively find AMBER images.

    Your download script extracts images under:
        data/amber

    So recursive search is safer than assuming one exact subfolder.
    """

    root = Path(root)

    search_root = Path(image_dir) if image_dir is not None else root

    if not search_root.exists():
        raise FileNotFoundError(
            f"AMBER image directory not found: {search_root}"
        )

    image_paths = [
        path
        for path in search_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(image_paths, key=_natural_key)


def _load_query_rows(
    query_path: Optional[PathLike],
) -> Optional[List[Dict[str, Any]]]:
    if query_path is None:
        return None

    query_path = Path(query_path)

    if not query_path.exists():
        return None

    data = load_json(query_path)

    if isinstance(data, list):
        return [dict(row) for row in data]

    if isinstance(data, dict):
        for key in [
            "data",
            "queries",
            "questions",
            "annotations",
            "samples",
            "items",
        ]:
            if key in data and isinstance(data[key], list):
                return [dict(row) for row in data[key]]

        rows = []

        for key, value in data.items():
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("id", key)
            else:
                row = {
                    "id": key,
                    "query": value,
                }

            rows.append(row)

        return rows

    raise ValueError(f"Unsupported AMBER query format: {query_path}")


def _build_image_index(
    image_paths: Sequence[Path],
) -> Dict[str, Path]:
    """
    Build flexible lookup index.

    Supports common AMBER naming patterns:
        1
        000001
        AMBER_1
        AMBER_1.jpg
        image filename
        image stem
    """

    index: Dict[str, Path] = {}

    for path in image_paths:
        keys = {
            path.name,
            path.stem,
            path.as_posix(),
            str(path),
        }

        stem_int = _safe_int(path.stem)

        if stem_int is not None:
            keys.update(
                {
                    str(stem_int),
                    f"{stem_int:04d}",
                    f"{stem_int:05d}",
                    f"{stem_int:06d}",
                    f"{stem_int:012d}",
                    f"AMBER_{stem_int}",
                    f"AMBER_{stem_int}.jpg",
                    f"AMBER_{stem_int}.jpeg",
                    f"AMBER_{stem_int}.png",
                    f"{stem_int}.jpg",
                    f"{stem_int}.jpeg",
                    f"{stem_int}.png",
                }
            )

        # Also support stems containing numbers, e.g. AMBER_123.
        match = re.search(r"(\d+)", path.stem)

        if match:
            number = int(match.group(1))

            keys.update(
                {
                    str(number),
                    f"{number:04d}",
                    f"{number:05d}",
                    f"{number:06d}",
                    f"{number:012d}",
                    f"AMBER_{number}",
                    f"AMBER_{number}.jpg",
                    f"AMBER_{number}.jpeg",
                    f"AMBER_{number}.png",
                    f"{number}.jpg",
                    f"{number}.jpeg",
                    f"{number}.png",
                }
            )

        for key in keys:
            index[str(key)] = path

    return index


def _extract_id(
    row: Dict[str, Any],
    fallback_index: int,
) -> Any:
    value = first_existing(
        row,
        [
            "id",
            "image_id",
            "img_id",
            "question_id",
            "idx",
        ],
    )

    if value is None:
        return fallback_index

    value_int = _safe_int(value)

    if value_int is not None:
        return value_int

    return str(value)


def _extract_prompt(
    row: Dict[str, Any],
    default_prompt: str,
) -> str:
    value = first_existing(
        row,
        [
            "query",
            "prompt",
            "question",
            "instruction",
            "text",
        ],
    )

    if value is None:
        return default_prompt

    value = str(value).strip()

    if value == "":
        return default_prompt

    return value


def _extract_image_refs(
    row: Dict[str, Any],
    sample_id: Any,
) -> List[str]:
    refs: List[str] = []

    value = first_existing(
        row,
        [
            "image_path",
            "img_path",
            "path",
            "image",
            "img",
            "file_name",
            "filename",
            "image_name",
            "name",
        ],
    )

    if value is not None:
        value = str(value)

        refs.extend(
            [
                value,
                Path(value).name,
                Path(value).stem,
            ]
        )

    sample_id_int = _safe_int(sample_id)

    if sample_id_int is not None:
        refs.extend(
            [
                str(sample_id_int),
                f"{sample_id_int:04d}",
                f"{sample_id_int:05d}",
                f"{sample_id_int:06d}",
                f"{sample_id_int:012d}",
                f"AMBER_{sample_id_int}",
                f"AMBER_{sample_id_int}.jpg",
                f"AMBER_{sample_id_int}.jpeg",
                f"AMBER_{sample_id_int}.png",
                f"{sample_id_int}.jpg",
                f"{sample_id_int}.jpeg",
                f"{sample_id_int}.png",
            ]
        )
    else:
        refs.append(str(sample_id))

    return refs


def _resolve_image_path(
    row: Dict[str, Any],
    sample_id: Any,
    image_index: Dict[str, Path],
    root: Path,
) -> Optional[Path]:
    for ref in _extract_image_refs(row=row, sample_id=sample_id):
        ref = str(ref)

        if ref in image_index:
            return image_index[ref]

        direct = Path(ref)

        if direct.exists():
            return direct

        joined = root / ref

        if joined.exists():
            return joined

    return None


def _normalize_objects(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return sorted(
            {
                str(x).strip()
                for x in value
                if str(x).strip()
            }
        )

    if isinstance(value, str):
        parts = re.split(r"[,;/|]", value)

        return sorted(
            {
                part.strip()
                for part in parts
                if part.strip()
            }
        )

    return []


def _extract_objects(row: Dict[str, Any]) -> List[str]:
    value = first_existing(
        row,
        [
            "objects",
            "gt_objects",
            "truth",
            "labels",
            "gt_labels",
            "ground_truth",
        ],
    )

    return _normalize_objects(value)


def _load_from_queries(
    query_rows: Sequence[Dict[str, Any]],
    root: Path,
    image_index: Dict[str, Path],
    prompt: str,
    sample_ids: Optional[Sequence[int]],
) -> List[Dict[str, Any]]:
    allowed_ids = None

    if sample_ids is not None:
        allowed_ids = set(int(x) for x in sample_ids)

    samples: List[Dict[str, Any]] = []

    for idx, row in enumerate(query_rows, start=1):
        sample_id = _extract_id(
            row=row,
            fallback_index=idx,
        )

        sample_id_int = _safe_int(sample_id)

        if allowed_ids is not None:
            if sample_id_int is None or sample_id_int not in allowed_ids:
                continue

        image_path = _resolve_image_path(
            row=row,
            sample_id=sample_id,
            image_index=image_index,
            root=root,
        )

        if image_path is None:
            continue

        sample_prompt = _extract_prompt(
            row=row,
            default_prompt=prompt,
        )

        objects = _extract_objects(row)

        sample = dict(row)
        sample.update(
            {
                "id": sample_id,
                "image_id": sample_id,
                "amber_id": sample_id,
                "image_path": str(image_path),
                "file_name": image_path.name,
                "prompt": sample_prompt,
                "query": sample_prompt,
                "objects": objects,
                "gt_objects": objects,
                "dataset": "amber",
            }
        )

        samples.append(sample)

    return samples


def _load_from_images_only(
    image_paths: Sequence[Path],
    prompt: str,
    sample_ids: Optional[Sequence[int]],
) -> List[Dict[str, Any]]:
    allowed_ids = None

    if sample_ids is not None:
        allowed_ids = set(int(x) for x in sample_ids)

    samples: List[Dict[str, Any]] = []

    for idx, image_path in enumerate(image_paths, start=1):
        stem_int = _safe_int(image_path.stem)

        if stem_int is not None:
            sample_id = stem_int
        else:
            match = re.search(r"(\d+)", image_path.stem)
            sample_id = int(match.group(1)) if match else idx

        if allowed_ids is not None and int(sample_id) not in allowed_ids:
            continue

        samples.append(
            {
                "id": sample_id,
                "image_id": sample_id,
                "amber_id": sample_id,
                "image_path": str(image_path),
                "file_name": image_path.name,
                "prompt": prompt,
                "query": prompt,
                "objects": [],
                "gt_objects": [],
                "dataset": "amber",
            }
        )

    return samples


def load_amber(
    root: PathLike = "data/amber",
    image_dir: Optional[PathLike] = None,
    query_path: Optional[PathLike] = "data/amber/query/query_generative.json",
    annotation_path: Optional[PathLike] = None,
    max_samples: Optional[int] = None,
    prompt: str = DEFAULT_AMBER_PROMPT,
    seed: int = 42,
    shuffle: bool = False,
    sample_ids: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Load AMBER generative-query samples.

    Main path:
        query_path="data/amber/query/query_generative.json"

    Returned sample:
        {
            "id": ...,
            "image_id": ...,
            "amber_id": ...,
            "image_path": "...",
            "file_name": "...",
            "prompt": "... AMBER query ...",
            "query": "... AMBER query ...",
            "objects": [],
            "gt_objects": [],
            "dataset": "amber",
        }

    `annotation_path` is accepted for compatibility. AMBER evaluation uses
    annotations.json separately, not during inference.
    """

    _ = annotation_path

    root = Path(root)

    image_paths = find_amber_images(
        root=root,
        image_dir=image_dir,
    )

    if len(image_paths) == 0:
        raise RuntimeError(f"No AMBER images found under: {root}")

    image_index = _build_image_index(image_paths)
    query_rows = _load_query_rows(query_path)

    if query_rows is not None:
        samples = _load_from_queries(
            query_rows=query_rows,
            root=root,
            image_index=image_index,
            prompt=prompt,
            sample_ids=sample_ids,
        )
    else:
        samples = _load_from_images_only(
            image_paths=image_paths,
            prompt=prompt,
            sample_ids=sample_ids,
        )

    samples = sorted(samples, key=lambda x: _natural_key(x["id"]))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(samples)

    if max_samples is not None:
        samples = samples[: int(max_samples)]

    return samples