"""
File I/O utilities.

Used by scripts for:
    - loading JSON / JSONL
    - saving JSON / JSONL
    - atomic saving
    - making output folders
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, List, Union, Dict


PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """
    Create directory if it does not exist.

    If path looks like a file path, use its parent.
    """

    path = Path(path)

    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)

    return path


def load_json(path: PathLike) -> Any:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(
    obj: Any,
    path: PathLike,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    path = Path(path)
    ensure_dir(path)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=indent,
            ensure_ascii=ensure_ascii,
        )


def save_json_atomic(
    obj: Any,
    path: PathLike,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """
    Atomic JSON save.

    Useful for long COCO/AMBER generation runs.
    Prevents corrupted output if the process is interrupted while writing.
    """

    path = Path(path)
    ensure_dir(path)

    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix=path.name + ".",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(
                obj,
                f,
                indent=indent,
                ensure_ascii=ensure_ascii,
            )

        os.replace(tmp_path, path)

    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_jsonl(path: PathLike) -> List[Any]:
    path = Path(path)

    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def save_jsonl(
    rows: Iterable[Any],
    path: PathLike,
    ensure_ascii: bool = False,
) -> None:
    path = Path(path)
    ensure_dir(path)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=ensure_ascii,
                )
                + "\n"
            )


def append_jsonl(
    row: Any,
    path: PathLike,
    ensure_ascii: bool = False,
) -> None:
    path = Path(path)
    ensure_dir(path)

    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                row,
                ensure_ascii=ensure_ascii,
            )
            + "\n"
        )


def file_exists(path: PathLike) -> bool:
    return Path(path).exists()


def count_json_rows(path: PathLike) -> int:
    """
    Count rows in a JSON list or JSONL file.

    Useful for resume logic.
    """

    path = Path(path)

    if not path.exists():
        return 0

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    data = load_json(path)

    if isinstance(data, list):
        return len(data)

    return 1

def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list in {path}")
        return data

    rows: List[Dict[str, Any]] = []

    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()

        if not line:
            continue

        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON on line {line_no} in {path}: {e}"
            ) from e

        if not isinstance(row, dict):
            raise ValueError(
                f"Expected JSON object on line {line_no} in {path}"
            )

        rows.append(row)

    return rows