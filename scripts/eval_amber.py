# scripts/eval_amber.py

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


def maybe_int(value: Any) -> Any:
    if isinstance(value, int):
        return value

    if isinstance(value, str):
        value = value.strip()

        if value.isdigit():
            return int(value)

    return value


def first_existing(row: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for key in keys:
        value = row.get(key)

        if value is not None:
            return value

    return None


def format_amber_rows(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []

    for idx, row in enumerate(raw_rows):
        sample_id = first_existing(
            row,
            ["id", "amber_id", "question_id", "image_id", "idx"],
        )

        response = first_existing(
            row,
            ["response", "caption", "prediction", "text"],
        )

        if sample_id is None:
            raise ValueError(
                f"Row {idx} has no AMBER id. "
                f"Available keys: {list(row.keys())}"
            )

        if response is None or str(response).strip() == "":
            raise ValueError(
                f"Row {idx} has no response/caption. "
                f"Available keys: {list(row.keys())}"
            )

        formatted.append(
            {
                "id": maybe_int(sample_id),
                "response": str(response).strip(),
            }
        )

    return formatted


def save_json_array(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def default_formatted_path(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}_amber_format.json"


def default_metrics_path(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}_amber_metrics.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Raw generate.py output or already formatted AMBER JSON.",
    )

    parser.add_argument(
        "--formatted-output",
        default=None,
        help="Where to save formatted AMBER input JSON.",
    )

    parser.add_argument(
        "--metrics-output",
        default=None,
        help="Where to save printed AMBER metrics.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/eval/amber",
        help="Default directory for formatted file and metrics text.",
    )

    parser.add_argument(
        "--evaluation-type",
        default="g",
        choices=["a", "g", "d", "de", "da", "dr"],
        help=(
            "AMBER evaluation type. Use g for generative captions. "
            "Use a only if your file contains all AMBER tasks."
        ),
    )

    parser.add_argument(
        "--word-association",
        default="data/amber/relation.json",
    )

    parser.add_argument(
        "--safe-words",
        default="data/amber/safe_words.txt",
    )

    parser.add_argument(
        "--annotation",
        default="data/amber/annotations.json",
    )

    parser.add_argument(
        "--metrics",
        default="data/amber/metrics.txt",
    )

    parser.add_argument(
        "--similarity-score",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--skip-format",
        action="store_true",
        help="Use --input directly as AMBER evaluator input.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formatted_path = (
        Path(args.formatted_output)
        if args.formatted_output is not None
        else default_formatted_path(input_path, output_dir)
    )

    metrics_path = (
        Path(args.metrics_output)
        if args.metrics_output is not None
        else default_metrics_path(input_path, output_dir)
    )

    if args.skip_format:
        inference_data_path = input_path
    else:
        raw_rows = load_json_or_jsonl(input_path)
        formatted_rows = format_amber_rows(raw_rows)
        save_json_array(formatted_rows, formatted_path)
        inference_data_path = formatted_path

        print("Formatted AMBER input:", formatted_path)
        print("Rows:", len(formatted_rows))

    # Import here because amber_eval loads spaCy at import time.
    import evaluation.amber_eval as amber_eval

    eval_args = argparse.Namespace(
        word_association=args.word_association,
        safe_words=args.safe_words,
        inference_data=str(inference_data_path),
        annotation=args.annotation,
        metrics=args.metrics,
        similarity_score=args.similarity_score,
        evaluation_type=args.evaluation_type,
    )

    # evaluation.amber_eval.init() reads the module-level global `args`,
    # so set it explicitly before calling main().
    amber_eval.args = eval_args

    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        amber_eval.main(eval_args)

    metrics_text = buffer.getvalue()

    print(metrics_text)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(metrics_text, encoding="utf-8")

    print("Saved AMBER metrics:", metrics_path)


if __name__ == "__main__":
    main()