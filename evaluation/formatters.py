# evaluation/formatters/format_predictions.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    # Normal JSON list
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list in {path}")
        return data

    # JSONL
    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e

        if not isinstance(row, dict):
            raise ValueError(f"Expected JSON object on line {line_no} in {path}")

        rows.append(row)

    return rows


def maybe_int(value: Any) -> Any:
    if isinstance(value, int):
        return value

    if isinstance(value, str):
        value = value.strip()
        if value.isdigit():
            return int(value)

    return value


def get_first_existing(row: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def get_caption(row: Dict[str, Any]) -> Optional[str]:
    value = get_first_existing(row, ["caption", "response", "prediction", "text"])

    if value is None:
        return None

    return str(value).strip()


def format_coco_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    image_id = get_first_existing(row, ["image_id", "id"])
    caption = get_caption(row)

    if image_id is None or caption is None or caption == "":
        return None

    return {
        "image_id": maybe_int(image_id),
        "caption": caption,
    }


def format_amber_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sample_id = get_first_existing(row, ["id", "amber_id", "image_id", "question_id"])
    response = get_caption(row)

    if sample_id is None or response is None or response == "":
        return None

    return {
        "id": maybe_int(sample_id),
        "response": response,
    }


def write_coco_jsonl(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json_array(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, help="Input generated JSON/JSONL file.")
    parser.add_argument("--output", required=True, help="Output formatted file.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["coco", "coco_val2017", "amber"],
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip rows with missing id/caption instead of raising an error.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    raw_rows = load_json_or_jsonl(input_path)

    formatted_rows: List[Dict[str, Any]] = []
    invalid_count = 0

    for idx, row in enumerate(raw_rows):
        if args.dataset in {"coco", "coco_val2017"}:
            formatted = format_coco_row(row)
        elif args.dataset == "amber":
            formatted = format_amber_row(row)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        if formatted is None:
            invalid_count += 1

            if args.skip_invalid:
                continue

            raise ValueError(
                f"Invalid row at index {idx}. "
                f"Could not find required id/caption fields. "
                f"Available keys: {list(row.keys())}"
            )

        formatted_rows.append(formatted)

    if args.dataset in {"coco", "coco_val2017"}:
        # COCO format requested: one JSON object per line.
        write_coco_jsonl(formatted_rows, output_path)

    elif args.dataset == "amber":
        # AMBER format requested: JSON array.
        write_json_array(formatted_rows, output_path)

    print("=" * 100)
    print("Formatted predictions")
    print("=" * 100)
    print("input:", input_path)
    print("output:", output_path)
    print("dataset:", args.dataset)
    print("input rows:", len(raw_rows))
    print("output rows:", len(formatted_rows))
    print("invalid rows:", invalid_count)


if __name__ == "__main__":
    main()