# scripts/eval_output_statistics.py

import argparse
import csv
import json
import os
import re
from collections import Counter
from statistics import mean, median
from typing import Any, Dict, List, Optional, Set, Tuple


COCO_ALIASES = {
    "person": ["person", "people", "man", "woman", "boy", "girl", "child", "children"],
    "bicycle": ["bicycle", "bike"],
    "car": ["car", "cars", "vehicle", "vehicles"],
    "motorcycle": ["motorcycle", "motorbike"],
    "airplane": ["airplane", "plane", "aircraft"],
    "bus": ["bus"],
    "train": ["train"],
    "truck": ["truck"],
    "boat": ["boat"],
    "traffic light": ["traffic light", "traffic lights"],
    "fire hydrant": ["fire hydrant"],
    "stop sign": ["stop sign"],
    "parking meter": ["parking meter"],
    "bench": ["bench"],
    "bird": ["bird"],
    "cat": ["cat"],
    "dog": ["dog"],
    "horse": ["horse"],
    "sheep": ["sheep"],
    "cow": ["cow"],
    "elephant": ["elephant"],
    "bear": ["bear"],
    "zebra": ["zebra"],
    "giraffe": ["giraffe"],
    "backpack": ["backpack", "bag"],
    "umbrella": ["umbrella"],
    "handbag": ["handbag", "purse"],
    "tie": ["tie"],
    "suitcase": ["suitcase", "luggage"],
    "frisbee": ["frisbee"],
    "skis": ["skis", "ski"],
    "snowboard": ["snowboard"],
    "sports ball": ["sports ball", "ball"],
    "kite": ["kite"],
    "baseball bat": ["baseball bat", "bat"],
    "baseball glove": ["baseball glove", "glove"],
    "skateboard": ["skateboard"],
    "surfboard": ["surfboard"],
    "tennis racket": ["tennis racket", "tennis racquet", "racket", "racquet"],
    "bottle": ["bottle"],
    "wine glass": ["wine glass", "glass"],
    "cup": ["cup"],
    "fork": ["fork"],
    "knife": ["knife"],
    "spoon": ["spoon"],
    "bowl": ["bowl"],
    "banana": ["banana"],
    "apple": ["apple"],
    "sandwich": ["sandwich"],
    "orange": ["orange"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot"],
    "hot dog": ["hot dog"],
    "pizza": ["pizza"],
    "donut": ["donut", "doughnut"],
    "cake": ["cake"],
    "chair": ["chair"],
    "couch": ["couch", "sofa"],
    "potted plant": ["potted plant", "plant"],
    "bed": ["bed"],
    "dining table": ["dining table", "table"],
    "toilet": ["toilet"],
    "tv": ["tv", "television"],
    "laptop": ["laptop", "computer"],
    "mouse": ["mouse"],
    "remote": ["remote"],
    "keyboard": ["keyboard"],
    "cell phone": ["cell phone", "phone", "mobile phone"],
    "microwave": ["microwave"],
    "oven": ["oven"],
    "toaster": ["toaster"],
    "sink": ["sink"],
    "refrigerator": ["refrigerator", "fridge"],
    "book": ["book"],
    "clock": ["clock"],
    "vase": ["vase"],
    "scissors": ["scissors"],
    "teddy bear": ["teddy bear"],
    "hair drier": ["hair drier", "hair dryer"],
    "toothbrush": ["toothbrush"],
}


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


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+", text.lower())


def count_sentences(text: str) -> int:
    parts = [p.strip() for p in re.split(r"[.!?]+", text) if p.strip()]
    return max(1, len(parts)) if text.strip() else 0


def build_alias_patterns() -> List[Tuple[str, str, re.Pattern]]:
    pairs = []
    for category, aliases in COCO_ALIASES.items():
        for alias in aliases:
            escaped = re.escape(alias.lower())
            pattern = re.compile(rf"(?<![a-z]){escaped}s?(?![a-z])")
            pairs.append((category, alias, pattern))

    # Match longer aliases first, so "potted plant" is preferred before "plant".
    pairs.sort(key=lambda x: len(x[1]), reverse=True)
    return pairs


ALIAS_PATTERNS = build_alias_patterns()


def overlaps_existing(start: int, end: int, spans: List[Tuple[int, int]]) -> bool:
    for old_start, old_end in spans:
        if start < old_end and end > old_start:
            return True
    return False


def extract_object_mentions(caption: str) -> List[str]:
    """
    Extract COCO-style object mentions using simple alias matching.
    Overlapping spans are suppressed to reduce double-counting.
    """
    caption_l = caption.lower()
    mentions = []
    used_spans: List[Tuple[int, int]] = []

    for category, _, pattern in ALIAS_PATTERNS:
        for match in pattern.finditer(caption_l):
            start, end = match.span()

            if overlaps_existing(start, end, used_spans):
                continue

            mentions.append(category)
            used_spans.append((start, end))

    return mentions


def load_coco_gt_categories(instances_path: str) -> Dict[int, Set[str]]:
    with open(instances_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cat_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}

    image_to_categories: Dict[int, Set[str]] = {}

    for ann in data["annotations"]:
        image_id = int(ann["image_id"])
        cat_name = cat_id_to_name[int(ann["category_id"])]
        image_to_categories.setdefault(image_id, set()).add(cat_name)

    return image_to_categories


def safe_avg(values: List[float]) -> float:
    return float(mean(values)) if values else 0.0


def safe_median(values: List[float]) -> float:
    return float(median(values)) if values else 0.0


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def f1_from_precision_recall(precision: float, recall: float) -> float:
    return safe_div(2.0 * precision * recall, precision + recall)


def evaluate_statistics(
    pred_path: str,
    instances_path: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    records = load_json_or_jsonl(pred_path)

    gt_by_image = None
    if instances_path:
        gt_by_image = load_coco_gt_categories(instances_path)

    per_image = []

    caption_lengths = []
    sentence_counts = []
    object_mention_counts = []
    unique_object_counts = []
    correct_counts = []
    hallucinated_counts = []

    total_object_counter = Counter()
    total_correct_counter = Counter()
    total_hallucinated_counter = Counter()

    # Micro unique-object classification counts.
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    # Optional macro per-image scores.
    macro_precision_scores = []
    macro_recall_scores = []
    macro_f1_scores = []

    seen = set()

    for record in records:
        image_id = get_image_id(record)
        if image_id in seen:
            continue
        seen.add(image_id)

        caption = get_caption(record)
        words = tokenize_words(caption)
        mentions = extract_object_mentions(caption)
        unique_mentions = sorted(set(mentions))

        total_object_counter.update(mentions)

        row = {
            "image_id": image_id,
            "caption_length_words": len(words),
            "sentence_count": count_sentences(caption),
            "object_mentions": len(mentions),
            "unique_object_mentions": len(unique_mentions),
            "objects": ";".join(unique_mentions),
            "caption": caption,
        }

        if gt_by_image is not None:
            gt_categories = gt_by_image.get(image_id, set())

            # Mention-level correctness.
            correct = [obj for obj in mentions if obj in gt_categories]
            hallucinated = [obj for obj in mentions if obj not in gt_categories]

            correct_counts.append(len(correct))
            hallucinated_counts.append(len(hallucinated))

            total_correct_counter.update(correct)
            total_hallucinated_counter.update(hallucinated)

            # Unique object-category precision/recall/F1.
            pred_unique = set(mentions)

            tp_set = pred_unique & gt_categories
            fp_set = pred_unique - gt_categories
            fn_set = gt_categories - pred_unique

            tp = len(tp_set)
            fp = len(fp_set)
            fn = len(fn_set)

            precision = safe_div(tp, tp + fp)
            recall = safe_div(tp, tp + fn)
            f1 = f1_from_precision_recall(precision, recall)

            macro_precision_scores.append(precision)
            macro_recall_scores.append(recall)
            macro_f1_scores.append(f1)

            micro_tp += tp
            micro_fp += fp
            micro_fn += fn

            row.update(
                {
                    "gt_objects": ";".join(sorted(gt_categories)),
                    "correct_object_mentions": len(correct),
                    "hallucinated_object_mentions": len(hallucinated),
                    "correct_objects": ";".join(sorted(set(correct))),
                    "hallucinated_objects": ";".join(sorted(set(hallucinated))),
                    "missed_gt_objects": ";".join(sorted(fn_set)),
                    "unique_object_precision": precision,
                    "unique_object_recall": recall,
                    "unique_object_f1": f1,
                }
            )

        per_image.append(row)

        caption_lengths.append(len(words))
        sentence_counts.append(row["sentence_count"])
        object_mention_counts.append(len(mentions))
        unique_object_counts.append(len(unique_mentions))

    total_mentions = sum(object_mention_counts)

    summary: Dict[str, Any] = {
        "num_images": len(per_image),
        "avg_caption_length_words": safe_avg(caption_lengths),
        "median_caption_length_words": safe_median(caption_lengths),
        "min_caption_length_words": min(caption_lengths) if caption_lengths else 0,
        "max_caption_length_words": max(caption_lengths) if caption_lengths else 0,
        "avg_sentence_count": safe_avg(sentence_counts),
        "avg_object_mentions": safe_avg(object_mention_counts),
        "avg_unique_object_mentions": safe_avg(unique_object_counts),
        "total_object_mentions": int(total_mentions),
        "captions_with_object_mentions": int(sum(1 for x in object_mention_counts if x > 0)),
        "top_object_mentions": total_object_counter.most_common(30),
    }

    if gt_by_image is not None:
        total_correct = sum(correct_counts)
        total_hallucinated = sum(hallucinated_counts)

        micro_precision = safe_div(micro_tp, micro_tp + micro_fp)
        micro_recall = safe_div(micro_tp, micro_tp + micro_fn)
        micro_f1 = f1_from_precision_recall(micro_precision, micro_recall)

        summary.update(
            {
                # Mention-level object statistics.
                "avg_correct_object_mentions": safe_avg(correct_counts),
                "avg_hallucinated_object_mentions": safe_avg(hallucinated_counts),
                "total_correct_object_mentions": int(total_correct),
                "total_hallucinated_object_mentions": int(total_hallucinated),
                "object_precision": safe_div(total_correct, total_mentions),
                "object_hallucination_rate": safe_div(total_hallucinated, total_mentions),

                # Unique object-category micro scores.
                "micro_unique_object_tp": int(micro_tp),
                "micro_unique_object_fp": int(micro_fp),
                "micro_unique_object_fn": int(micro_fn),
                "micro_unique_object_precision": micro_precision,
                "micro_unique_object_recall": micro_recall,
                "micro_unique_object_f1": micro_f1,

                # Unique object-category macro scores.
                "macro_unique_object_precision": safe_avg(macro_precision_scores),
                "macro_unique_object_recall": safe_avg(macro_recall_scores),
                "macro_unique_object_f1": safe_avg(macro_f1_scores),

                "captions_with_hallucinated_objects": int(sum(1 for x in hallucinated_counts if x > 0)),
                "top_correct_object_mentions": total_correct_counter.most_common(30),
                "top_hallucinated_object_mentions": total_hallucinated_counter.most_common(30),
            }
        )

    return summary, per_image


def save_json(path: str, data: Any) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred",
        required=True,
        help="Raw prediction JSON/JSONL.",
    )
    parser.add_argument(
        "--instances",
        default=None,
        help="Optional COCO instances JSON, e.g. data/coco2017/annotations/instances_val2017.json",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--per-image-out",
        default=None,
        help="Optional per-image CSV output path.",
    )

    args = parser.parse_args()

    summary, per_image = evaluate_statistics(
        pred_path=args.pred,
        instances_path=args.instances,
    )

    save_json(args.out, summary)

    if args.per_image_out:
        save_csv(args.per_image_out, per_image)

    print("\nOutput statistics:")
    print(f"Images: {summary['num_images']}")
    print(f"Avg. caption length: {summary['avg_caption_length_words']:.2f} words")
    print(f"Median caption length: {summary['median_caption_length_words']:.2f} words")
    print(f"Avg. sentence count: {summary['avg_sentence_count']:.2f}")
    print(f"Avg. object mentions: {summary['avg_object_mentions']:.2f}")
    print(f"Avg. unique object mentions: {summary['avg_unique_object_mentions']:.2f}")

    if "avg_correct_object_mentions" in summary:
        print(f"Avg. correct object mentions: {summary['avg_correct_object_mentions']:.2f}")
        print(f"Avg. hallucinated object mentions: {summary['avg_hallucinated_object_mentions']:.2f}")
        print(f"Object precision: {summary['object_precision']:.4f}")
        print(f"Object hallucination rate: {summary['object_hallucination_rate']:.4f}")
        print(f"Micro unique object precision: {summary['micro_unique_object_precision']:.4f}")
        print(f"Micro unique object recall: {summary['micro_unique_object_recall']:.4f}")
        print(f"Micro unique object F1: {summary['micro_unique_object_f1']:.4f}")

    print(f"\nSaved summary to: {args.out}")
    if args.per_image_out:
        print(f"Saved per-image statistics to: {args.per_image_out}")


if __name__ == "__main__":
    main()