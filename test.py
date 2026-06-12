# test_llava_all.py

import argparse
import math
import traceback
from typing import Any, Dict, List, Optional

import torch

from data.coco import load_coco_val2017
from models.registry import build_model_wrapper
from decoding.registry import build_decoder_config, generate_samples_with_decoder
from decoding.stepwise import StepwiseConfig, generate_stepwise_batch
from phg import PHGConfig, generate_phg_samples


MODEL_NAME = "llava15_7b"


def print_header(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def assert_good_caption(caption: str, test_name: str):
    print("caption repr:", repr(caption))
    print("caption:")
    print(caption)

    if not isinstance(caption, str):
        raise AssertionError(f"{test_name}: caption is not string")

    if caption.strip() == "":
        raise AssertionError(f"{test_name}: empty caption")

    if len(caption.strip().split()) < 3:
        raise AssertionError(f"{test_name}: suspiciously short caption: {repr(caption)}")


def build_llava_wrapper(dtype: str):
    return build_model_wrapper(
        MODEL_NAME,
        config_kwargs={
            "torch_dtype": dtype,
            "device_map": "auto",
            "attn_implementation": "eager",
        },
    ).load()


def load_one_sample(image_dir: str, annotation_path: str):
    samples = load_coco_val2017(
        image_dir=image_dir,
        annotation_path=annotation_path,
        max_samples=1,
    )

    sample = samples[0]

    print("Sample id:", sample["id"])
    print("Image path:", sample["image_path"])
    print("Prompt:", sample.get("prompt", "Describe this image."))

    return sample, samples


def test_direct_wrapper(wrapper, sample, max_new_tokens: int):
    print_header("1. DIRECT LLAVA WRAPPER TEST")

    inputs = wrapper.prepare_batch(
        image_paths=[sample["image_path"]],
        prompts=[sample.get("prompt", "Describe this image.")],
        use_chat_template=False,
    )

    print("Prepared input keys:")
    for key, value in inputs.items():
        if hasattr(value, "shape"):
            print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value)}")

    output = wrapper.generate_from_inputs(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    caption = output.captions[0]

    assert_good_caption(caption, "direct_wrapper")

    if output.input_ids is not None:
        print("input_ids shape:", tuple(output.input_ids.shape))

    if output.sequences is not None:
        print("sequences shape:", tuple(output.sequences.shape))

    print("OK: direct wrapper")


def test_normal_decoder(wrapper, samples, decoding_name: str, max_new_tokens: int):
    print_header(f"2. NORMAL DECODER TEST: {decoding_name}")

    if decoding_name == "greedy":
        config_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }

    elif decoding_name == "dola_low":
        config_kwargs = {
            "max_new_tokens": max_new_tokens,
            "dola_layers": "low",
            "repetition_penalty": 1.2,
            "do_sample": False,
        }

    elif decoding_name == "vcd":
        config_kwargs = {
            "max_new_tokens": max_new_tokens,
            "cd_alpha": 1.0,
            "cd_beta": 0.1,
            "noise_step": 500,
            "top_p": None,
            "do_sample": False,
        }

    else:
        raise ValueError(f"Unknown decoding: {decoding_name}")

    decoder_config = build_decoder_config(
        decoding_name=decoding_name,
        config_kwargs=config_kwargs,
    )

    rows = generate_samples_with_decoder(
        decoding_name=decoding_name,
        wrapper=wrapper,
        samples=samples,
        config=decoder_config,
        use_chat_template=False,
    )

    for row in rows:
        print("id:", row["id"])
        caption = row["caption"]
        assert_good_caption(caption, f"normal_decoder_{decoding_name}")

    print(f"OK: normal decoder {decoding_name}")


def build_stepwise_config(mode: str, max_new_tokens: int):
    kwargs = {
        "decoding_mode": mode,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": 0,
        "output_attentions": True,
        "selected_layers": [-8, -4, -1],
        "image_grid_shape": (24, 24),
        "keep_attn_on_cpu": True,
        "do_sample": False,
    }

    if mode == "dola":
        kwargs.update(
            {
                "dola_layers": "low",
                "dola_relative_top": 0.1,
                "repetition_penalty": 1.2,
            }
        )

    elif mode == "vcd":
        kwargs.update(
            {
                "cd_alpha": 1.0,
                "cd_beta": 0.1,
                "noise_step": 500,
                "top_p": None,
            }
        )

    return StepwiseConfig(**kwargs)


def test_stepwise(wrapper, sample, mode: str, max_new_tokens: int):
    print_header(f"3. STEPWISE TEST: {mode}")

    config = build_stepwise_config(
        mode=mode,
        max_new_tokens=max_new_tokens,
    )

    output = generate_stepwise_batch(
        wrapper=wrapper,
        image_paths=[sample["image_path"]],
        prompts=[sample.get("prompt", "Describe this image.")],
        config=config,
        use_chat_template=False,
    )

    assert_good_caption(output.caption, f"stepwise_{mode}")

    print("\nTOKENS")
    print("-" * 100)

    for i, step in enumerate(output.steps):
        print(
            i,
            "token_id=", step.token_id,
            "token_text=", repr(step.token_text),
            "token_prob=", step.token_prob,
            "max_prob=", step.max_prob,
            "has_attn=", step.image_attn_by_layer is not None,
            "dola_layer=", step.dola_premature_layer,
        )

    has_any_attention = any(
        step.image_attn_by_layer is not None
        for step in output.steps
    )

    print("\nhas_any_attention:", has_any_attention)

    if not has_any_attention:
        raise AssertionError(f"stepwise_{mode}: no attention maps found")

    for step in output.steps:
        if step.image_attn_by_layer is not None:
            print("\nFIRST ATTENTION STEP")
            print("token:", repr(step.token_text))

            for layer_id, attn in step.image_attn_by_layer.items():
                print("layer:", layer_id, "attn shape:", tuple(attn.shape))

            break

    print(f"OK: stepwise {mode}")

    return output


def normalize_attn_to_grid(attn: torch.Tensor, grid_shape=(24, 24)) -> torch.Tensor:
    """
    Robust attention-map normalization for smoke testing.

    Expected possibilities:
        [576]
        [24, 24]
        [heads, 576]
        [heads, 24, 24]
    """

    if not torch.is_tensor(attn):
        attn = torch.tensor(attn)

    attn = attn.detach().float().cpu()

    if attn.ndim == 1:
        if attn.numel() == grid_shape[0] * grid_shape[1]:
            return attn.reshape(grid_shape)

    if attn.ndim == 2:
        if tuple(attn.shape) == tuple(grid_shape):
            return attn

        if attn.shape[-1] == grid_shape[0] * grid_shape[1]:
            return attn.mean(dim=0).reshape(grid_shape)

    if attn.ndim == 3:
        if tuple(attn.shape[-2:]) == tuple(grid_shape):
            return attn.mean(dim=0)

    raise ValueError(f"Cannot convert attention shape to grid: {tuple(attn.shape)}")


def simple_ads(attn_grid: torch.Tensor, foreground_ratio: float = 0.10) -> float:
    """
    Simple ADS-like smoke metric.

    This is not your final paper metric. It only checks that attention can be
    converted into a spatial map and produces a finite diffuse score.
    """

    x = attn_grid.detach().float().cpu()
    x = torch.clamp(x, min=0)

    total = x.sum()

    if total <= 0:
        return float("nan")

    p = x / total

    flat = p.flatten()
    k = max(1, int(math.ceil(flat.numel() * foreground_ratio)))

    topk_values, _ = torch.topk(flat, k=k)

    foreground_mass = topk_values.sum().item()
    background_mass = max(0.0, 1.0 - foreground_mass)

    eps = 1e-12
    entropy = -(flat * torch.log(flat + eps)).sum().item()
    max_entropy = math.log(float(flat.numel()))

    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    ads = background_mass * normalized_entropy

    return float(ads)


def test_grounding_ads(stepwise_output):
    print_header("4. GROUNDING / ATTENTION / ADS SMOKE TEST")

    chosen_step = None

    for step in stepwise_output.steps:
        if step.image_attn_by_layer is not None:
            chosen_step = step
            break

    if chosen_step is None:
        raise AssertionError("No step with image attention found")

    print("chosen token:", repr(chosen_step.token_text))

    ads_values = {}

    for layer_id, attn in chosen_step.image_attn_by_layer.items():
        attn_grid = normalize_attn_to_grid(attn, grid_shape=(24, 24))
        ads_value = simple_ads(attn_grid)

        print("layer:", layer_id)
        print("attn_grid shape:", tuple(attn_grid.shape))
        print("attn sum:", float(attn_grid.sum()))
        print("ADS smoke value:", ads_value)

        if not math.isfinite(ads_value):
            raise AssertionError(f"ADS is not finite for layer {layer_id}")

        ads_values[layer_id] = ads_value

    print("OK: grounding ADS smoke")

    return ads_values


def build_phg_config(mode: str, max_new_tokens: int, max_rounds: int):
    kwargs = {
        "decoding_mode": mode,
        "max_rounds": max_rounds,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": 3,
        "top_k": 3,
        "accumulate_prob": 0.5,
        "iou_thresh": 0.5,
        "ads_thresh": 0.45,
        "ads_foreground_ratio": 0.10,
        "selected_layers": [-8, -4, -1],
        "image_grid_shape": (24, 24),
        "debug": True,
    }

    if mode == "dola":
        kwargs.update(
            {
                "dola_layers": "low",
                "dola_relative_top": 0.1,
                "repetition_penalty": 1.2,
            }
        )

    elif mode == "vcd":
        kwargs.update(
            {
                "cd_alpha": 1.0,
                "cd_beta": 0.1,
                "noise_step": 500,
                "top_p": None,
            }
        )

    return PHGConfig(**kwargs)


def test_phg(wrapper, samples, mode: str, max_new_tokens: int, max_rounds: int):
    print_header(f"5. PHG SMOKE TEST: {mode}")

    config = build_phg_config(
        mode=mode,
        max_new_tokens=max_new_tokens,
        max_rounds=max_rounds,
    )

    rows = generate_phg_samples(
        wrapper=wrapper,
        samples=samples,
        config=config,
        use_chat_template=False,
        include_trace=True,
    )

    row = rows[0]

    caption = row["caption"]
    assert_good_caption(caption, f"phg_{mode}")

    print("\nOBJECTS")
    print("-" * 100)
    print(row.get("objects"))

    print("\nDECISION TRACE")
    print("-" * 100)

    for decision in row.get("decision_trace", []):
        print("round:", decision.get("round"))
        print("path:", decision.get("path"))
        print("stop_reason:", decision.get("stop_reason"))

        if "selected_candidate" in decision:
            print("selected_candidate:", decision["selected_candidate"])

        print("-" * 100)

    print(f"OK: PHG {mode}")


def run_test(name: str, fn):
    print_header(f"RUNNING: {name}")

    try:
        result = fn()
        print_header(f"PASSED: {name}")
        return result

    except Exception:
        print_header(f"FAILED: {name}")
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--stepwise-new-tokens", type=int, default=16)
    parser.add_argument("--phg-new-tokens", type=int, default=24)
    parser.add_argument("--phg-rounds", type=int, default=2)

    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "wrapper",
            "normal",
            "stepwise",
            "grounding",
            "phg",
        ],
        default="all",
    )

    parser.add_argument(
        "--image-dir",
        default="data/coco2017/val2017",
    )
    parser.add_argument(
        "--annotation-path",
        default="data/coco2017/annotations/instances_val2017.json",
    )

    args = parser.parse_args()

    print_header("LLAVA FULL TEST SUITE")
    print("model:", MODEL_NAME)
    print("dtype:", args.dtype)
    print("stage:", args.stage)

    sample, samples = load_one_sample(
        image_dir=args.image_dir,
        annotation_path=args.annotation_path,
    )

    wrapper = build_llava_wrapper(dtype=args.dtype)

    if args.stage in {"all", "wrapper"}:
        run_test(
            "direct_wrapper",
            lambda: test_direct_wrapper(
                wrapper=wrapper,
                sample=sample,
                max_new_tokens=args.max_new_tokens,
            ),
        )

    if args.stage in {"all", "normal"}:
        for decoding_name in ["greedy", "dola_low", "vcd"]:
            run_test(
                f"normal_{decoding_name}",
                lambda decoding_name=decoding_name: test_normal_decoder(
                    wrapper=wrapper,
                    samples=samples,
                    decoding_name=decoding_name,
                    max_new_tokens=args.max_new_tokens,
                ),
            )

    stepwise_greedy_output = None

    if args.stage in {"all", "stepwise", "grounding"}:
        for mode in ["greedy", "dola", "vcd"]:
            output = run_test(
                f"stepwise_{mode}",
                lambda mode=mode: test_stepwise(
                    wrapper=wrapper,
                    sample=sample,
                    mode=mode,
                    max_new_tokens=args.stepwise_new_tokens,
                ),
            )

            if mode == "greedy":
                stepwise_greedy_output = output

    if args.stage in {"all", "grounding"}:
        if stepwise_greedy_output is None:
            stepwise_greedy_output = run_test(
                "stepwise_greedy_for_grounding",
                lambda: test_stepwise(
                    wrapper=wrapper,
                    sample=sample,
                    mode="greedy",
                    max_new_tokens=args.stepwise_new_tokens,
                ),
            )

        run_test(
            "grounding_ads",
            lambda: test_grounding_ads(stepwise_greedy_output),
        )

    if args.stage in {"all", "phg"}:
        for mode in ["greedy", "dola", "vcd"]:
            run_test(
                f"phg_{mode}",
                lambda mode=mode: test_phg(
                    wrapper=wrapper,
                    samples=samples,
                    mode=mode,
                    max_new_tokens=args.phg_new_tokens,
                    max_rounds=args.phg_rounds,
                ),
            )

    print_header("ALL REQUESTED LLAVA TESTS PASSED")


if __name__ == "__main__":
    main()