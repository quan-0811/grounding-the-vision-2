# scripts/test_all_llava_qwen_methods.py

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from models.registry import build_model_wrapper
from decoding.registry import build_decoder_config, generate_samples_with_decoder
from phg import PHGConfig, generate_phg_samples
from utils.seed import seed_everything


BASELINE_METHODS = [
    "greedy",
    "dola_low",
    "vcd",
]

PHG_METHODS = [
    "greedy_phg",
    "dola_low_phg",
    "vcd_phg",
]

ALL_METHODS = BASELINE_METHODS + PHG_METHODS

DEFAULT_MODELS = [
    "llava15_7b",
    "qwen2vl_7b",
]


def parse_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_selected_layers(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def load_samples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.dataset == "coco_val2017":
        from data.coco import load_coco_val2017

        return load_coco_val2017(
            image_dir=args.coco_image_dir,
            annotation_path=args.coco_annotation_path,
            max_samples=args.max_samples,
            prompt=args.prompt,
        )

    if args.dataset == "amber":
        from data.amber import load_amber

        return load_amber(
            root=args.amber_root,
            query_path=args.amber_query_path,
            max_samples=args.max_samples,
            prompt=args.prompt,
        )

    raise ValueError(f"Unknown dataset: {args.dataset}")


def is_qwen(model_name: str) -> bool:
    return model_name.lower().startswith("qwen")


def default_use_chat_template(model_name: str) -> bool:
    # Qwen2-VL requires its official chat template path.
    # LLaVA wrapper usually already handles its prompt/image formatting.
    return is_qwen(model_name)


def build_wrapper(model_name: str, args: argparse.Namespace):
    config_kwargs: Dict[str, Any] = {
        "torch_dtype": args.dtype,
        "device_map": args.device_map,
    }

    if args.attn_implementation is not None and args.attn_implementation != "none":
        config_kwargs["attn_implementation"] = args.attn_implementation

    return build_model_wrapper(
        model_name,
        config_kwargs=config_kwargs,
    ).load()


def build_baseline_config(
    method: str,
    args: argparse.Namespace,
):
    if method == "greedy":
        return build_decoder_config(
            "greedy",
            config_kwargs={
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "use_cache": True,
            },
        )

    if method == "dola_low":
        return build_decoder_config(
            "dola_low",
            config_kwargs={
                "max_new_tokens": args.max_new_tokens,
                "dola_layers": "low",
                "repetition_penalty": args.repetition_penalty,
                "do_sample": False,
                "use_cache": True,
            },
        )

    if method == "vcd":
        return build_decoder_config(
            "vcd",
            config_kwargs={
                "max_new_tokens": args.max_new_tokens,
                "cd_alpha": args.cd_alpha,
                "cd_beta": args.cd_beta,
                "noise_step": args.noise_step,
                "do_sample": False,
                "use_cache": True,
            },
        )

    raise ValueError(f"Unknown baseline method: {method}")


def build_phg_config(
    method: str,
    args: argparse.Namespace,
) -> PHGConfig:
    if method == "greedy_phg":
        decoding_mode = "greedy"
        dola_layers = None

    elif method == "dola_low_phg":
        decoding_mode = "dola"
        dola_layers = "low"

    elif method == "vcd_phg":
        decoding_mode = "vcd"
        dola_layers = None

    else:
        raise ValueError(f"Unknown PHG method: {method}")

    return PHGConfig(
        decoding_mode=decoding_mode,

        max_rounds=args.phg_max_rounds,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.phg_min_new_tokens,

        top_k=args.phg_top_k,
        accumulate_prob=args.phg_accumulate_prob,

        iou_thresh=args.phg_iou_thresh,
        ads_thresh=args.phg_ads_thresh,
        ads_foreground_ratio=args.phg_ads_foreground_ratio,

        selected_layers=args.selected_layers,
        image_grid_shape=None,

        do_sample=False,
        temperature=1.0,
        top_p=None,
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


def assert_caption_ok(caption: str) -> None:
    if not caption.strip():
        raise AssertionError("Empty caption.")

    bad_patterns = [
        "TheThe",
        "containss",
        "#<",
        "<|vision",
        "<|image",
        "<|im_start|>",
        "<|im_end|>",
    ]

    for pattern in bad_patterns:
        if pattern in caption:
            raise AssertionError(
                f"Malformed pattern {pattern!r} found in caption: {caption!r}"
            )


def add_common_metadata(
    rows: List[Dict[str, Any]],
    model_name: str,
    method: str,
    dataset: str,
) -> List[Dict[str, Any]]:
    out_rows = []

    for row in rows:
        row = dict(row)
        row["model"] = model_name
        row["method"] = method
        row["dataset"] = dataset

        # AMBER evaluator often expects response alias.
        if dataset == "amber" and "caption" in row:
            row["response"] = row["caption"]

        out_rows.append(row)

    return out_rows


def output_path_for(
    args: argparse.Namespace,
    model_name: str,
    method: str,
) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir / (
        f"{model_name}_{method}_{args.dataset}_"
        f"n{args.max_samples}_tok{args.max_new_tokens}_seed{args.seed}.json"
    )


def save_rows(
    rows: List[Dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("Saved:", path)


def run_baseline_method(
    wrapper,
    model_name: str,
    method: str,
    samples: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    seed_everything(args.seed)

    config = build_baseline_config(
        method=method,
        args=args,
    )

    rows = generate_samples_with_decoder(
        decoding_name=method,
        wrapper=wrapper,
        samples=samples,
        config=config,
        image_key="image_path",
        prompt_key="prompt",
        id_key="id",
        caption_key="caption",
        use_chat_template=default_use_chat_template(model_name),
        include_prompt=args.include_prompt,
    )

    return add_common_metadata(
        rows=rows,
        model_name=model_name,
        method=method,
        dataset=args.dataset,
    )


def run_phg_method(
    wrapper,
    model_name: str,
    method: str,
    samples: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    seed_everything(args.seed)

    config = build_phg_config(
        method=method,
        args=args,
    )

    rows = generate_phg_samples(
        wrapper=wrapper,
        samples=samples,
        config=config,
        image_key="image_path",
        prompt_key="prompt",
        id_key="id",
        caption_key="caption",
        use_chat_template=default_use_chat_template(model_name),
        include_prompt=args.include_prompt,
        include_trace=args.include_trace,
    )

    return add_common_metadata(
        rows=rows,
        model_name=model_name,
        method=method,
        dataset=args.dataset,
    )


def print_rows_summary(
    rows: List[Dict[str, Any]],
    model_name: str,
    method: str,
) -> None:
    print("=" * 100)
    print(f"SUMMARY: {model_name} / {method}")
    print("=" * 100)

    for row in rows:
        caption = row.get("caption", "")

        print("-" * 100)
        print("id:", row.get("id"))
        print("caption repr:", repr(caption))
        print("caption:")
        print(caption)

        assert_caption_ok(caption)

        if "objects" in row:
            print("objects:", row.get("objects", []))

        if "decision_trace" in row:
            print("num decision_trace:", len(row.get("decision_trace", [])))

        if "final_generated_ids" in row:
            print("num final_generated_ids:", len(row.get("final_generated_ids", [])))


def clear_gpu() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        type=parse_csv,
        default=DEFAULT_MODELS,
        help="Comma-separated models, e.g. llava15_7b,qwen2vl_7b",
    )

    parser.add_argument(
        "--methods",
        type=parse_csv,
        default=ALL_METHODS,
        help=(
            "Comma-separated methods. Options: "
            "greedy,dola_low,vcd,greedy_phg,dola_low_phg,vcd_phg"
        ),
    )

    parser.add_argument(
        "--dataset",
        default="coco_val2017",
        choices=["coco_val2017", "amber"],
    )

    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="eager")

    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output-dir", default="outputs/full_smoke")
    parser.add_argument("--include-prompt", action="store_true")
    parser.add_argument("--include-trace", action="store_true")

    # VCD
    parser.add_argument("--cd-alpha", type=float, default=1.0)
    parser.add_argument("--cd-beta", type=float, default=0.1)
    parser.add_argument("--noise-step", type=int, default=500)

    # DoLA
    parser.add_argument("--dola-relative-top", type=float, default=0.1)
    parser.add_argument("--dola-select-strategy", default="argmax")
    parser.add_argument("--repetition-penalty", type=float, default=1.2)

    # PHG
    parser.add_argument("--phg-max-rounds", type=int, default=4)
    parser.add_argument("--phg-min-new-tokens", type=int, default=3)
    parser.add_argument("--phg-top-k", type=int, default=3)
    parser.add_argument("--phg-accumulate-prob", type=float, default=0.5)
    parser.add_argument("--phg-iou-thresh", type=float, default=0.5)
    parser.add_argument("--phg-ads-thresh", type=float, default=0.45)
    parser.add_argument("--phg-ads-foreground-ratio", type=float, default=0.10)
    parser.add_argument("--phg-debug", action="store_true")

    parser.add_argument(
        "--selected-layers",
        type=parse_selected_layers,
        default=[-8, -4, -1],
    )

    parser.add_argument(
        "--stop-at-sentence-end",
        action="store_true",
    )

    parser.add_argument(
        "--stop-if-sentence-end-in-candidates",
        action="store_true",
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
        "--amber-query-path",
        default="data/amber/query/query_generative.json",
    )

    args = parser.parse_args()

    args.models = list(args.models)
    args.methods = list(args.methods)

    unknown_methods = sorted(set(args.methods) - set(ALL_METHODS))

    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}")

    print("=" * 100)
    print("FULL LLaVA/Qwen method smoke test")
    print("=" * 100)
    print("models:", args.models)
    print("methods:", args.methods)
    print("dataset:", args.dataset)
    print("max_samples:", args.max_samples)
    print("max_new_tokens:", args.max_new_tokens)
    print("seed:", args.seed)
    print("output_dir:", args.output_dir)

    seed_everything(args.seed)

    samples = load_samples(args)

    print("loaded samples:", len(samples))

    if len(samples) == 0:
        raise RuntimeError("No samples loaded.")

    manifest: List[Dict[str, Any]] = []

    for model_name in args.models:
        print("=" * 100)
        print("LOADING MODEL:", model_name)
        print("=" * 100)

        seed_everything(args.seed)

        wrapper = build_wrapper(
            model_name=model_name,
            args=args,
        )

        try:
            for method in args.methods:
                print("=" * 100)
                print("RUNNING:", model_name, "/", method)
                print("=" * 100)

                seed_everything(args.seed)

                if method in BASELINE_METHODS:
                    rows = run_baseline_method(
                        wrapper=wrapper,
                        model_name=model_name,
                        method=method,
                        samples=samples,
                        args=args,
                    )

                elif method in PHG_METHODS:
                    rows = run_phg_method(
                        wrapper=wrapper,
                        model_name=model_name,
                        method=method,
                        samples=samples,
                        args=args,
                    )

                else:
                    raise ValueError(f"Unknown method: {method}")

                print_rows_summary(
                    rows=rows,
                    model_name=model_name,
                    method=method,
                )

                path = output_path_for(
                    args=args,
                    model_name=model_name,
                    method=method,
                )

                save_rows(
                    rows=rows,
                    path=path,
                )

                manifest.append(
                    {
                        "model": model_name,
                        "method": method,
                        "dataset": args.dataset,
                        "num_rows": len(rows),
                        "path": str(path),
                    }
                )

        finally:
            del wrapper
            clear_gpu()

    manifest_path = Path(args.output_dir) / (
        f"manifest_{args.dataset}_n{args.max_samples}_"
        f"tok{args.max_new_tokens}_seed{args.seed}.json"
    )

    save_rows(
        rows=manifest,
        path=manifest_path,
    )

    print("=" * 100)
    print("OK: full LLaVA/Qwen smoke test completed")
    print("=" * 100)


if __name__ == "__main__":
    main()