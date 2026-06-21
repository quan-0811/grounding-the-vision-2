# scripts/eval_coco_caption_quality.py

import argparse
import json
import os
import re
from typing import Any, Dict, List

from pycocotools.coco import COCO

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["results", "predictions", "data", "samples"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(f"Unsupported prediction format: {path}")


def extract_int_id(value: Any) -> int:
    """
    Supports:
    - 391895
    - "391895"
    - "COCO_val2017_000000391895.jpg"
    - "/path/to/000000391895.jpg"
    """
    if isinstance(value, int):
        return value

    value = str(value)
    base = os.path.splitext(os.path.basename(value))[0]

    if base.isdigit():
        return int(base)

    matches = re.findall(r"\d+", base)
    if matches:
        return int(matches[-1])

    raise ValueError(f"Cannot extract integer image id from value: {value}")


def get_image_id(record: Dict[str, Any]) -> int:
    for key in ["image_id", "coco_image_id", "id", "image", "image_path", "file_name"]:
        if key in record:
            return extract_int_id(record[key])

    raise KeyError(f"Cannot find image id field in record keys: {record.keys()}")


def get_caption(record: Dict[str, Any]) -> str:
    for key in ["caption", "response", "text", "generated_text", "prediction", "answer"]:
        if key in record and record[key] is not None:
            return str(record[key]).strip()

    raise KeyError(f"Cannot find caption field in record keys: {record.keys()}")


def convert_to_coco_caption_format(pred_path: str, out_path: str) -> List[Dict[str, Any]]:
    records = load_json_or_jsonl(pred_path)

    formatted = []
    seen = set()

    for record in records:
        image_id = get_image_id(record)
        caption = get_caption(record)

        # COCO caption evaluation expects one prediction per image.
        if image_id in seen:
            continue

        formatted.append(
            {
                "image_id": image_id,
                "caption": caption,
            }
        )
        seen.add(image_id)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, indent=2, ensure_ascii=False)

    return formatted


def evaluate_metrics(coco: COCO, coco_res: COCO) -> Dict[str, float]:
    img_ids = coco_res.getImgIds()

    gts = {}
    res = {}

    for img_id in img_ids:
        gts[img_id] = coco.imgToAnns[img_id]
        res[img_id] = coco_res.imgToAnns[img_id]

    print("tokenization...")
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
    ]

    results: Dict[str, float] = {}

    print("setting up scorers...")
    for scorer, method in scorers:
        print(f"computing {method} score...")

        score, _ = scorer.compute_score(gts, res)

        if isinstance(method, list):
            for metric_name, metric_score in zip(method, score):
                results[metric_name] = float(metric_score)
                print(f"{metric_name}: {metric_score:.4f}")
        else:
            results[method] = float(score)
            print(f"{method}: {score:.4f}")

        # METEOR starts a Java process in some versions.
        close_fn = getattr(scorer, "close", None)
        if callable(close_fn):
            close_fn()

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ann",
        default="data/coco2017/annotations/captions_val2017.json",
        help="Path to COCO captions_val2017.json",
    )
    parser.add_argument(
        "--pred",
        required=True,
        help="Raw prediction JSON/JSONL or already formatted COCO result JSON",
    )
    parser.add_argument(
        "--formatted-out",
        default=None,
        help="Where to save COCO-formatted predictions",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Where to save metric results JSON",
    )

    args = parser.parse_args()

    if args.formatted_out is None:
        base = os.path.splitext(args.out)[0]
        args.formatted_out = base + "_coco_format.json"

    preds = convert_to_coco_caption_format(args.pred, args.formatted_out)

    print(f"Saved COCO-formatted predictions to: {args.formatted_out}")
    print(f"Number of evaluated images: {len(preds)}")

    coco = COCO(args.ann)
    coco_res = coco.loadRes(args.formatted_out)

    results = evaluate_metrics(coco=coco, coco_res=coco_res)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nCaption quality metrics:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")

    print(f"\nSaved results to: {args.out}")


if __name__ == "__main__":
    main()