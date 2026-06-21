# scripts/eval_chair.py

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.io import load_json_or_jsonl, save_json, save_jsonl
from evaluation.formatters import format_coco_row

def default_formatted_path(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}_chair_format.jsonl"


def default_result_path(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}_chair_result.json"


def default_metrics_path(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}_chair_metrics.txt"


def resolve_default_cache() -> Path:
    # User said chair.pk is in evaluation/. Your chair.py default uses chair.pkl.
    # Support both names and prefer the existing one.
    candidates = [
        PROJECT_ROOT / "evaluation" / "chair.pk",
        PROJECT_ROOT / "evaluation" / "chair.pkl",
    ]

    for path in candidates:
        if path.exists():
            return path

    # If neither exists, create the standard pickle path.
    return PROJECT_ROOT / "evaluation" / "chair.pkl"

def load_or_build_chair_evaluator(
    cache_path: Path,
    coco_path: str,
):
    from evaluation.chair import CHAIR

    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                evaluator = pickle.load(f)

            print("Loaded CHAIR evaluator cache:", cache_path)
            return evaluator

        except AttributeError as e:
            print("[Warning] Failed to load CHAIR cache:", cache_path)
            print("[Warning] Reason:", repr(e))
            print("[Warning] This usually means the cache was created with CHAIR under __main__.")
            print("[Warning] Rebuilding CHAIR cache...")

        except Exception as e:
            print("[Warning] Failed to load CHAIR cache:", cache_path)
            print("[Warning] Reason:", repr(e))
            print("[Warning] Rebuilding CHAIR cache...")

    print("CHAIR cache not found or invalid. Building from scratch...")
    print("COCO annotation dir:", coco_path)

    evaluator = CHAIR(coco_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with cache_path.open("wb") as f:
        pickle.dump(evaluator, f)

    print("Saved CHAIR evaluator cache:", cache_path)

    return evaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Raw generate.py output or formatted COCO JSON/JSONL.",
    )

    parser.add_argument(
        "--formatted-output",
        default=None,
        help="Where to save formatted COCO JSONL.",
    )

    parser.add_argument(
        "--save-path",
        default=None,
        help="Where to save detailed CHAIR result JSON.",
    )

    parser.add_argument(
        "--metrics-output",
        default=None,
        help="Where to save short metric text.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/eval/chair",
        help="Default directory for formatted file, result JSON, and metrics text.",
    )

    parser.add_argument(
        "--cache",
        default=None,
        help="Path to CHAIR pickle cache. Defaults to evaluation/chair.pk or evaluation/chair.pkl.",
    )

    parser.add_argument(
        "--coco-path",
        default="data/coco2017/annotations",
        help=(
            "Directory containing captions_val2017.json, captions_train2017.json, "
            "instances_val2017.json, and instances_train2017.json. "
            "Only needed if cache must be rebuilt."
        ),
    )

    parser.add_argument(
        "--skip-format",
        action="store_true",
        help="Use --input directly as CHAIR cap_file.",
    )

    parser.add_argument(
        "--image-id-key",
        default="image_id",
    )

    parser.add_argument(
        "--caption-key",
        default="caption",
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

    save_path = (
        Path(args.save_path)
        if args.save_path is not None
        else default_result_path(input_path, output_dir)
    )

    metrics_path = (
        Path(args.metrics_output)
        if args.metrics_output is not None
        else default_metrics_path(input_path, output_dir)
    )

    cache_path = Path(args.cache) if args.cache is not None else resolve_default_cache()

    if args.skip_format:
        cap_file = input_path
    else:
        raw_rows = load_json_or_jsonl(input_path)
        formatted_rows = [format_coco_row(row) for row in raw_rows]
        formatted_rows = [row for row in formatted_rows if row is not None]
        save_jsonl(formatted_rows, formatted_path)
        cap_file = formatted_path

        print("Formatted COCO input:", formatted_path)
        print("Rows:", len(formatted_rows))

    evaluator = load_or_build_chair_evaluator(
        cache_path=cache_path,
        coco_path=args.coco_path,
    )

    result = evaluator.compute_chair(
        str(cap_file),
        args.image_id_key,
        args.caption_key,
    )

    save_json(result, save_path)

    metrics = result["overall_metrics"]
    lines = []

    for key, value in metrics.items():
        lines.append(f"{key}: {value * 100:.1f}")

    metrics_text = "\n".join(lines) + "\n"

    print(metrics_text)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(metrics_text, encoding="utf-8")

    print("Saved CHAIR result:", save_path)
    print("Saved CHAIR metrics:", metrics_path)


if __name__ == "__main__":
    main()