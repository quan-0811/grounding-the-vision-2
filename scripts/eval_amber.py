# scripts/eval_amber.py

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from utils.io import load_json_or_jsonl, save_json
from utils.dict_utils import first_existing, maybe_int
from evaluation.formatters import format_amber_row


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
        formatted_rows = [format_amber_row(row) for row in raw_rows]
        formatted_rows = [row for row in formatted_rows if row is not None]
        save_json(formatted_rows, formatted_path)
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