# evaluation/formatters/format_predictions.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from utils.io import load_json_or_jsonl, save_json, save_jsonl
from utils.dict_utils import first_existing, maybe_int


def get_caption(row: Dict[str, Any]) -> Optional[str]:
    value = first_existing(row, ["caption", "response", "prediction", "text"])

    if value is None:
        return None

    return str(value).strip()


def format_coco_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    image_id = first_existing(row, ["image_id", "id"])
    caption = get_caption(row)

    if image_id is None or caption is None or caption == "":
        return None

    return {
        "image_id": maybe_int(image_id),
        "caption": caption,
    }


def format_amber_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sample_id = first_existing(row, ["id", "amber_id", "image_id", "question_id"])
    response = get_caption(row)

    if sample_id is None or response is None or response == "":
        return None

    return {
        "id": maybe_int(sample_id),
        "response": response,
    }


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
        save_jsonl(formatted_rows, output_path)

    elif args.dataset == "amber":
        # AMBER format requested: JSON array.
        save_json(formatted_rows, output_path)

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