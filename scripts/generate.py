# scripts/generate.py

from __future__ import annotations

import argparse
import hashlib
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from models.registry import build_model_wrapper
from decoding.registry import build_decoder_config, generate_samples_with_decoder
from phg import PHGConfig, generate_phg_samples

from utils.io import load_json, save_json_atomic, ensure_dir
from utils.seed import seed_everything


SUPPORTED_MODELS = {
    "llava15_7b",
    "qwen2vl_7b",
}

SUPPORTED_DECODINGS = {
    "greedy",
    "dola_low",
    "vcd",
    "greedy_phg",
    "dola_low_phg",
    "vcd_phg",
}

SUPPORTED_DATASETS = {
    "coco_val2017",
    "amber",
}


# ============================================================
# Small helpers
# ============================================================

def iter_batches(items: Sequence[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield list(items[i:i + batch_size])


def normalize_id(value: Any) -> str:
    return str(value)


def get_sample_id(sample: Dict[str, Any]) -> Any:
    for key in ["id", "image_id", "question_id", "idx", "amber_id"]:
        if key in sample:
            return sample[key]

    raise KeyError(
        f"Could not find sample id. Available keys: {list(sample.keys())}"
    )


def get_existing_ids(rows: Sequence[Dict[str, Any]]) -> set[str]:
    done = set()

    for row in rows:
        try:
            done.add(normalize_id(get_sample_id(row)))
        except Exception:
            pass

    return done


def is_phg_decoding(decoding: str) -> bool:
    return decoding.endswith("_phg")


def get_base_phg_mode(decoding: str) -> str:
    if decoding == "greedy_phg":
        return "greedy"

    if decoding == "dola_low_phg":
        return "dola"

    if decoding == "vcd_phg":
        return "vcd"

    raise ValueError(f"Unknown PHG decoding: {decoding}")


def model_uses_chat_template(model_name: str) -> bool:
    return model_name == "qwen2vl_7b"


def parse_selected_layers(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def stable_seed_from_batch(
    base_seed: int,
    batch: Sequence[Dict[str, Any]],
) -> int:
    """
    Build a deterministic per-batch seed.

    This matters for VCD / VCD+PHG because the noised image branch uses
    random diffusion noise. Using a stable seed makes resume/re-run behavior
    much more reproducible.
    """

    if len(batch) == 0:
        return int(base_seed)

    ids = "|".join(normalize_id(get_sample_id(sample)) for sample in batch)
    digest = hashlib.md5(ids.encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16)

    return int((int(base_seed) + offset) % (2**31 - 1))


# ============================================================
# Dataset loading
# ============================================================

def load_dataset(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.dataset == "coco_val2017":
        from data.coco import load_coco_val2017

        samples = load_coco_val2017(
            image_dir=args.coco_image_dir,
            annotation_path=args.coco_annotation_path,
            max_samples=args.max_samples,
            prompt=args.prompt,
        )

    elif args.dataset == "amber":
        from data.amber import load_amber

        samples = load_amber(
            root=args.amber_root,
            image_dir=args.amber_image_dir,
            query_path=args.amber_query_path,
            annotation_path=args.amber_annotation_path,
            max_samples=args.max_samples,
            prompt=args.prompt,
        )

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    normalized = []

    for sample in samples:
        sample = dict(sample)

        if "prompt" not in sample:
            sample["prompt"] = args.prompt

        normalized.append(sample)

    if args.max_samples is not None:
        normalized = normalized[: args.max_samples]

    return normalized


# ============================================================
# Model / decoding config
# ============================================================

def build_model_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "torch_dtype": args.dtype,
        "device_map": args.device_map,
    }

    if args.attn_implementation is not None and args.attn_implementation != "none":
        kwargs["attn_implementation"] = args.attn_implementation

    if args.model == "qwen2vl_7b":
        if args.qwen_min_pixels is not None:
            kwargs["min_pixels"] = args.qwen_min_pixels

        if args.qwen_max_pixels is not None:
            kwargs["max_pixels"] = args.qwen_max_pixels

    elif args.model == "llava15_7b":
        pass

    else:
        raise ValueError(f"Unknown model: {args.model}")

    return kwargs


def build_normal_decoder_config(
    model_name: str,
    decoding: str,
    args: argparse.Namespace,
):
    if decoding == "greedy":
        config_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "use_cache": True,
        }

        if args.do_sample:
            config_kwargs.update(
                {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                }
            )

    elif decoding == "dola_low":
        config_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "dola_layers": "low",
            "repetition_penalty": args.repetition_penalty,
            "do_sample": args.do_sample,
            "use_cache": True,
        }

        if args.do_sample:
            config_kwargs.update(
                {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                }
            )

    elif decoding == "vcd":
        if model_name not in {"llava15_7b", "qwen2vl_7b"}:
            raise ValueError(
                f"VCD is enabled only for llava15_7b and qwen2vl_7b. "
                f"Got {model_name}."
            )

        config_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "cd_alpha": args.cd_alpha,
            "cd_beta": args.cd_beta,
            "noise_step": args.noise_step,
            "do_sample": args.do_sample,
            "use_cache": True,
        }

        if args.do_sample:
            config_kwargs.update(
                {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                }
            )

    else:
        raise ValueError(f"Unknown decoding: {decoding}")

    return build_decoder_config(
        decoding_name=decoding,
        config_kwargs=config_kwargs,
    )


def build_phg_config(
    model_name: str,
    decoding: str,
    args: argparse.Namespace,
) -> PHGConfig:
    mode = get_base_phg_mode(decoding)

    if mode == "dola":
        dola_layers = "low"
    else:
        dola_layers = None

    # LLaVA-1.5 has fixed 24x24 image token grid.
    # Qwen2-VL uses dynamic image_grid_thw, so keep None.
    image_grid_shape = (24, 24) if model_name == "llava15_7b" else None

    return PHGConfig(
        decoding_mode=mode,

        max_rounds=args.phg_max_rounds,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.phg_min_new_tokens,

        top_k=args.phg_top_k,
        accumulate_prob=args.phg_accumulate_prob,

        iou_thresh=args.phg_iou_thresh,
        ads_thresh=args.phg_ads_thresh,
        ads_foreground_ratio=args.phg_ads_foreground_ratio,

        selected_layers=args.selected_layers,
        image_grid_shape=image_grid_shape,

        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,

        stop_at_sentence_end=args.stop_at_sentence_end,
        stop_if_sentence_end_in_candidates=args.stop_if_sentence_end_in_candidates,

        dola_layers=dola_layers,
        dola_relative_top=args.dola_relative_top,
        dola_select_strategy=args.dola_select_strategy,

        cd_alpha=args.cd_alpha,
        cd_beta=args.cd_beta,
        noise_step=args.noise_step,
        image_tensor_key="pixel_values",

        debug=args.phg_debug,
    )


# ============================================================
# Generation
# ============================================================

def add_row_metadata(
    row: Dict[str, Any],
    sample: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    out = dict(row)

    sample_id = get_sample_id(sample)

    out.setdefault("id", sample_id)

    if "image_id" in sample:
        out.setdefault("image_id", sample["image_id"])

    if "amber_id" in sample:
        out.setdefault("amber_id", sample["amber_id"])

    if "image_path" in sample:
        out.setdefault("image_path", sample["image_path"])

    if "file_name" in sample:
        out.setdefault("file_name", sample["file_name"])

    if "prompt" in sample:
        out.setdefault("prompt", sample["prompt"])

    out["model"] = args.model
    out["decoding"] = args.decoding
    out["dataset"] = args.dataset

    # AMBER evaluator alias.
    if args.dataset == "amber" and "caption" in out:
        out["response"] = out["caption"]

    return out


def generate_batch_rows(
    wrapper,
    batch: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    use_chat_template = model_uses_chat_template(args.model)

    if is_phg_decoding(args.decoding):
        config = build_phg_config(
            model_name=args.model,
            decoding=args.decoding,
            args=args,
        )

        rows = generate_phg_samples(
            wrapper=wrapper,
            samples=batch,
            config=config,
            image_key="image_path",
            prompt_key="prompt",
            id_key="id",
            caption_key="caption",
            use_chat_template=use_chat_template,
            include_prompt=True,
            include_trace=args.include_trace,
        )

    else:
        config = build_normal_decoder_config(
            model_name=args.model,
            decoding=args.decoding,
            args=args,
        )

        rows = generate_samples_with_decoder(
            decoding_name=args.decoding,
            wrapper=wrapper,
            samples=batch,
            config=config,
            image_key="image_path",
            prompt_key="prompt",
            id_key="id",
            caption_key="caption",
            use_chat_template=use_chat_template,
            include_prompt=True,
        )

    final_rows = []

    for sample, row in zip(batch, rows):
        final_rows.append(
            add_row_metadata(
                row=row,
                sample=sample,
                args=args,
            )
        )

    return final_rows


def validate_args(args: argparse.Namespace) -> None:
    if args.model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model: {args.model}. "
            f"Supported models: {sorted(SUPPORTED_MODELS)}"
        )

    if args.decoding not in SUPPORTED_DECODINGS:
        raise ValueError(
            f"Unknown decoding: {args.decoding}. "
            f"Supported decodings: {sorted(SUPPORTED_DECODINGS)}"
        )

    if args.dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset: {args.dataset}. "
            f"Supported datasets: {sorted(SUPPORTED_DATASETS)}"
        )

    if is_phg_decoding(args.decoding) and args.batch_size != 1:
        print("[Warning] PHG is stepwise. Forcing batch_size=1.")
        args.batch_size = 1

    if args.output is None:
        raise ValueError("--output is required")

    if args.do_sample and args.temperature <= 0:
        raise ValueError("--temperature must be > 0 when --do-sample is used.")


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--decoding", required=True, choices=sorted(SUPPORTED_DECODINGS))
    parser.add_argument("--dataset", required=True, choices=sorted(SUPPORTED_DATASETS))
    parser.add_argument("--output", required=True)

    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--max-new-tokens", type=int, default=64)

    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)

    # VCD
    parser.add_argument("--cd-alpha", type=float, default=1.0)
    parser.add_argument("--cd-beta", type=float, default=0.1)
    parser.add_argument("--noise-step", type=int, default=500)

    # DoLA
    parser.add_argument("--dola-relative-top", type=float, default=0.1)
    parser.add_argument("--dola-select-strategy", default="argmax")

    # PHG
    parser.add_argument("--phg-max-rounds", type=int, default=4)
    parser.add_argument("--phg-min-new-tokens", type=int, default=3)
    parser.add_argument("--phg-top-k", type=int, default=3)
    parser.add_argument("--phg-accumulate-prob", type=float, default=0.5)
    parser.add_argument("--phg-iou-thresh", type=float, default=0.5)
    parser.add_argument("--phg-ads-thresh", type=float, default=0.45)
    parser.add_argument("--phg-ads-foreground-ratio", type=float, default=0.10)
    parser.add_argument("--phg-debug", action="store_true")
    parser.add_argument("--include-trace", action="store_true")

    parser.add_argument(
        "--stop-at-sentence-end",
        action="store_true",
        help="Stop each PHG round at a sentence boundary.",
    )

    parser.add_argument(
        "--stop-if-sentence-end-in-candidates",
        action="store_true",
        help="Stop early if a sentence boundary appears among uncertain candidates.",
    )

    parser.add_argument(
        "--selected-layers",
        type=parse_selected_layers,
        default=[-8, -4, -1],
    )

    # COCO
    parser.add_argument(
        "--coco-image-dir",
        default="data/coco2017/val2017",
    )
    parser.add_argument(
        "--coco-annotation-path",
        default="data/coco2017/annotations/instances_val2017.json",
    )

    # AMBER
    parser.add_argument(
        "--amber-root",
        default="data/amber",
    )
    parser.add_argument(
        "--amber-image-dir",
        default=None,
    )
    parser.add_argument(
        "--amber-query-path",
        default="data/amber/query/query_generative.json",
    )
    parser.add_argument(
        "--amber-annotation-path",
        default="data/amber/annotations.json",
    )

    # Qwen dynamic-resolution controls
    parser.add_argument("--qwen-min-pixels", type=int, default=None)
    parser.add_argument("--qwen-max-pixels", type=int, default=None)

    return parser


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    validate_args(args)

    seed_everything(args.seed)

    output_path = Path(args.output)
    ensure_dir(output_path.parent)

    print("=" * 100)
    print("GENERATE")
    print("=" * 100)
    print("model:", args.model)
    print("decoding:", args.decoding)
    print("dataset:", args.dataset)
    print("dtype:", args.dtype)
    print("batch_size:", args.batch_size)
    print("max_samples:", args.max_samples)
    print("max_new_tokens:", args.max_new_tokens)
    print("phg_max_rounds:", args.phg_max_rounds)
    print("seed:", args.seed)
    print("output:", args.output)
    print("resume:", args.resume)

    samples = load_dataset(args)

    print("loaded samples:", len(samples))

    if len(samples) == 0:
        raise RuntimeError("No samples loaded.")

    existing_rows: List[Dict[str, Any]] = []

    if args.resume and output_path.exists():
        existing_rows = load_json(output_path)
        print("existing rows:", len(existing_rows))

    done_ids = get_existing_ids(existing_rows)

    remaining = [
        sample
        for sample in samples
        if normalize_id(get_sample_id(sample)) not in done_ids
    ]

    print("remaining samples:", len(remaining))

    if len(remaining) == 0:
        print("Nothing to do.")
        return

    print("\nLoading model...")
    wrapper = build_model_wrapper(
        args.model,
        config_kwargs=build_model_kwargs(args),
    ).load()

    all_rows = list(existing_rows)
    batches = list(iter_batches(remaining, args.batch_size))

    for batch_idx, batch in enumerate(tqdm(batches, desc="Generating")):
        try:
            # Stable per-batch seed. Important for VCD / VCD+PHG noised branch.
            batch_seed = stable_seed_from_batch(
                base_seed=args.seed,
                batch=batch,
            )
            seed_everything(batch_seed)

            rows = generate_batch_rows(
                wrapper=wrapper,
                batch=batch,
                args=args,
            )

            all_rows.extend(rows)
            save_json_atomic(all_rows, output_path)

            print("\nSaved:", output_path)
            print("total rows:", len(all_rows))

            for row in rows:
                print("-" * 100)
                print("id:", row.get("id"))
                print("caption repr:", repr(row.get("caption", "")))
                print("caption:")
                print(row.get("caption", ""))

        except Exception:
            print("\nFailed at batch:", batch_idx)
            print("Batch sample ids:", [get_sample_id(x) for x in batch])
            traceback.print_exc()

            save_json_atomic(all_rows, output_path)
            print("Partial output saved:", output_path)

            raise

    print("\nDone.")
    print("Final output:", output_path)
    print("Rows:", len(all_rows))


if __name__ == "__main__":
    main()