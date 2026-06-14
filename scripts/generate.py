# scripts/generate.py

from __future__ import annotations

import argparse
import hashlib
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from models.registry import build_model_wrapper
from decoding.registry import build_decoder_config, generate_samples_with_decoder
from phg import PHGConfig, generate_phg_samples

from utils.config import deep_update, get_config_section, load_yaml
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


CONFIG_SECTIONS = [
    "runtime",
    "model_kwargs",
    "dataset_kwargs",
    "generation",
    "decoding_kwargs",
    "phg_kwargs",
]


HARD_DEFAULTS: Dict[str, Any] = {
    "dtype": "float16",
    "device_map": "auto",
    "attn_implementation": "eager",
    "batch_size": 1,
    "max_samples": None,
    "resume": False,
    "seed": 42,

    "prompt": "Describe this image.",
    "max_new_tokens": 64,

    "do_sample": False,
    "temperature": 1.0,
    "top_p": None,
    "top_k": None,
    "repetition_penalty": 1.2,

    "cd_alpha": 1.0,
    "cd_beta": 0.1,
    "noise_step": 500,

    "dola_relative_top": 0.1,
    "dola_select_strategy": "argmax",

    "phg_max_rounds": 5,
    "phg_min_new_tokens": 3,
    "phg_top_k": 3,
    "phg_accumulate_prob": 0.5,
    "phg_iou_thresh": 0.5,
    "phg_ads_thresh": 0.45,
    "phg_ads_foreground_ratio": 0.10,
    "phg_debug": False,
    "include_trace": False,
    "stop_at_sentence_end": False,
    "stop_if_sentence_end_in_candidates": False,
    "selected_layers": None,

    "coco_image_dir": "data/coco2017/val2017",
    "coco_annotation_path": "data/coco2017/annotations/instances_val2017.json",

    "amber_root": "data/amber",
    "amber_image_dir": None,
    "amber_query_path": "data/amber/query/query_generative.json",
    "amber_annotation_path": "data/amber/annotations.json",

    "qwen_min_pixels": None,
    "qwen_max_pixels": None,
}


# ============================================================
# Config loading
# ============================================================

def config_path(
    config_root: Path,
    section: str,
    name: str,
) -> Path:
    return config_root / section / f"{name}.yaml"


def load_generation_config(args: argparse.Namespace) -> Dict[str, Any]:
    config_root = Path(args.config_root)

    model_path = (
        Path(args.model_config)
        if args.model_config is not None
        else config_path(config_root, "models", args.model)
    )

    dataset_path = (
        Path(args.dataset_config)
        if args.dataset_config is not None
        else config_path(config_root, "datasets", args.dataset)
    )

    decoding_path = (
        Path(args.decoding_config)
        if args.decoding_config is not None
        else config_path(config_root, "decoding", args.decoding)
    )

    merged: Dict[str, Any] = {}

    for path in [model_path, dataset_path, decoding_path]:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        merged = deep_update(merged, load_yaml(path))

    return merged


def fill_args_from_config(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
) -> argparse.Namespace:
    for section_name in CONFIG_SECTIONS:
        section = get_config_section(cfg, section_name)

        for key, value in section.items():
            if not hasattr(args, key):
                raise ValueError(
                    f"Unknown config key `{key}` in section `{section_name}`. "
                    f"Add it to argparse or remove it from YAML."
                )

            if getattr(args, key) is None:
                setattr(args, key, value)

    for key, value in HARD_DEFAULTS.items():
        if hasattr(args, key) and getattr(args, key) is None:
            setattr(args, key, value)

    return args


def print_effective_config(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
) -> None:
    if not args.print_config:
        return

    print("=" * 100)
    print("LOADED YAML CONFIG")
    print("=" * 100)

    for section_name in CONFIG_SECTIONS:
        section = get_config_section(cfg, section_name)
        if section:
            print(f"[{section_name}]")
            for key, value in section.items():
                print(f"{key}: {value}")
            print()

    print("=" * 100)
    print("EFFECTIVE ARGS")
    print("=" * 100)

    for key, value in sorted(vars(args).items()):
        print(f"{key}: {value}")


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


def parse_selected_layers(value: str) -> Optional[List[int]]:
    value = str(value).strip().lower()

    if value in {"none", "null", "auto", "all"}:
        return None

    return [int(x.strip()) for x in value.split(",") if x.strip()]


def stable_seed_from_batch(
    base_seed: int,
    batch: Sequence[Dict[str, Any]],
) -> int:
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

    dola_layers = "low" if mode == "dola" else None

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

    parser.add_argument("--config-root", default="configs")
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--decoding-config", default=None)
    parser.add_argument("--print-config", action="store_true")

    parser.add_argument("--dtype", default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)

    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)

    parser.add_argument("--cd-alpha", type=float, default=None)
    parser.add_argument("--cd-beta", type=float, default=None)
    parser.add_argument("--noise-step", type=int, default=None)

    parser.add_argument("--dola-relative-top", type=float, default=None)
    parser.add_argument("--dola-select-strategy", default=None)

    parser.add_argument("--phg-max-rounds", type=int, default=None)
    parser.add_argument("--phg-min-new-tokens", type=int, default=None)
    parser.add_argument("--phg-top-k", type=int, default=None)
    parser.add_argument("--phg-accumulate-prob", type=float, default=None)
    parser.add_argument("--phg-iou-thresh", type=float, default=None)
    parser.add_argument("--phg-ads-thresh", type=float, default=None)
    parser.add_argument("--phg-ads-foreground-ratio", type=float, default=None)
    parser.add_argument("--phg-debug", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include-trace", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument(
        "--stop-at-sentence-end",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument(
        "--stop-if-sentence-end-in-candidates",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument(
        "--selected-layers",
        type=parse_selected_layers,
        default=None,
    )

    parser.add_argument("--coco-image-dir", default=None)
    parser.add_argument("--coco-annotation-path", default=None)

    parser.add_argument("--amber-root", default=None)
    parser.add_argument("--amber-image-dir", default=None)
    parser.add_argument("--amber-query-path", default=None)
    parser.add_argument("--amber-annotation-path", default=None)

    parser.add_argument("--qwen-min-pixels", type=int, default=None)
    parser.add_argument("--qwen-max-pixels", type=int, default=None)

    return parser


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_generation_config(args)
    args = fill_args_from_config(args, cfg)

    validate_args(args)
    print_effective_config(args, cfg)

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