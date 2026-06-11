# scripts/test_next_models.py

from __future__ import annotations

import argparse
import traceback
from typing import Dict, List

from data.coco import load_coco_val2017
from models.registry import build_model_wrapper
from decoding.registry import build_decoder_config, generate_samples_with_decoder
from decoding.stepwise import StepwiseConfig, generate_stepwise_batch
from phg import PHGConfig, generate_phg_samples


ALL_MODELS = [
    "llava15_7b",
    "qwen2vl_7b",
    "internvl2_8b",
]


DEFAULT_DTYPE = {
    "llava15_7b": "float16",
    "qwen2vl_7b": "bfloat16",
    "internvl2_8b": "bfloat16",
}


def qwen_model(model_name: str) -> bool:
    return model_name in {"qwen2vl_7b"}


def internvl_model(model_name: str) -> bool:
    return model_name == "internvl2_8b"


def llava_model(model_name: str) -> bool:
    return model_name == "llava15_7b"


def use_chat_template_for_model(model_name: str) -> bool:
    return qwen_model(model_name)


def build_wrapper_for_test(
    model_name: str,
    dtype: str | None,
    need_attention: bool = False,
):
    config_kwargs = {
        "device_map": "auto",
        "torch_dtype": dtype or DEFAULT_DTYPE[model_name],
    }

    if llava_model(model_name):
        config_kwargs["attn_implementation"] = "eager" if need_attention else None

    if qwen_model(model_name):
        config_kwargs["attn_implementation"] = "eager" if need_attention else None

        # Optional memory control. Uncomment if Qwen image token count is too high.
        # config_kwargs["min_pixels"] = 256 * 28 * 28
        # config_kwargs["max_pixels"] = 1280 * 28 * 28

    # InternVL2 does not use attn_implementation here.

    wrapper = build_model_wrapper(
        model_name=model_name,
        config_kwargs=config_kwargs,
    ).load()

    return wrapper


def load_one_sample():
    samples = load_coco_val2017(
        image_dir="data/coco2017/val2017",
        annotation_path="data/coco2017/annotations/instances_val2017.json",
        max_samples=1,
    )

    assert len(samples) == 1
    return samples


def print_inputs(inputs):
    print("Input keys:", list(inputs.keys()))

    for key, value in inputs.items():
        if hasattr(value, "shape"):
            print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value)} = {value}")


def test_direct_generation(model_name: str, dtype: str | None, max_new_tokens: int):
    print(f"\n========== DIRECT GENERATION: {model_name} ==========")

    samples = load_one_sample()
    wrapper = build_wrapper_for_test(
        model_name=model_name,
        dtype=dtype,
        need_attention=False,
    )

    use_chat_template = use_chat_template_for_model(model_name)

    inputs = wrapper.prepare_batch(
        image_paths=[samples[0]["image_path"]],
        prompts=[samples[0].get("prompt", "Describe this image.")],
        use_chat_template=use_chat_template,
    )

    print_inputs(inputs)

    output = wrapper.generate_from_inputs(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    print("Caption:", repr(output.captions[0]))
    print(
        "Sequences shape:",
        None if output.sequences is None else tuple(output.sequences.shape),
    )

    assert len(output.captions) == 1
    assert isinstance(output.captions[0], str)

    # For InternVL2, blank caption means the generated-only output was sliced wrongly.
    assert output.captions[0].strip() != "", (
        "Blank caption. For InternVL2, check that generate_from_inputs() "
        "does NOT slice output_ids by prompt_len when output_ids is generated-only."
    )

    print("DIRECT OK.")


def test_registry_greedy(model_name: str, dtype: str | None, max_new_tokens: int):
    print(f"\n========== REGISTRY GREEDY: {model_name} ==========")

    samples = load_one_sample()
    wrapper = build_wrapper_for_test(
        model_name=model_name,
        dtype=dtype,
        need_attention=False,
    )

    config = build_decoder_config(
        decoding_name="greedy",
        config_kwargs={
            "max_new_tokens": max_new_tokens,
        },
    )

    rows = generate_samples_with_decoder(
        decoding_name="greedy",
        wrapper=wrapper,
        samples=samples,
        config=config,
        use_chat_template=use_chat_template_for_model(model_name),
    )

    print(rows[0])

    assert "caption" in rows[0]
    assert rows[0]["caption"].strip() != ""

    print("REGISTRY GREEDY OK.")


def test_registry_vcd(model_name: str, dtype: str | None, max_new_tokens: int):
    print(f"\n========== REGISTRY VCD: {model_name} ==========")

    if internvl_model(model_name):
        print("SKIP: InternVL2 + VCD is not supported yet in this codebase.")
        return

    samples = load_one_sample()
    wrapper = build_wrapper_for_test(
        model_name=model_name,
        dtype=dtype,
        need_attention=False,
    )

    vcd_kwargs = {
        "max_new_tokens": max_new_tokens,
    
        # Weaker VCD first.
        "cd_alpha": 0.5,
        "cd_beta": 0.2,
    
        # Less destructive than 500 for Qwen.
        "noise_step": 300,
    
        # Deterministic VCD.
        "noise_seed": 42,
    
        # For greedy VCD, do not use top_p.
        "top_p": None,
        "top_k": None,
        "do_sample": False,
    }
    
    if qwen_model(model_name):
        vcd_kwargs["suppress_token_ids"] = wrapper.get_suppress_token_ids()
    
    config = build_decoder_config(
        decoding_name="vcd",
        config_kwargs=vcd_kwargs,
    )

    rows = generate_samples_with_decoder(
        decoding_name="vcd",
        wrapper=wrapper,
        samples=samples,
        config=config,
        use_chat_template=use_chat_template_for_model(model_name),
    )

    print(rows[0])

    assert "caption" in rows[0]
    assert rows[0]["caption"].strip() != ""

    print("REGISTRY VCD OK.")


def test_stepwise(model_name: str, dtype: str | None, max_new_tokens: int):
    print(f"\n========== STEPWISE: {model_name} ==========")

    if internvl_model(model_name):
        print("SKIP: InternVL2 + stepwise is not supported yet.")
        return

    samples = load_one_sample()

    # LLaVA attention extraction is supported.
    # Qwen basic stepwise decoding is tested without attention first.
    need_attention = llava_model(model_name)

    wrapper = build_wrapper_for_test(
        model_name=model_name,
        dtype=dtype,
        need_attention=need_attention,
    )

    if llava_model(model_name):
        config = StepwiseConfig(
            decoding_mode="greedy",
            max_new_tokens=max_new_tokens,
            output_attentions=True,
            selected_layers=[-8, -4, -1],
            image_grid_shape=(24, 24),
            stop_on_eos=True,
            stop_on_sentence_end=False,
        )

    else:
        # Qwen stepwise first sanity test: no attention extraction yet.
        config = StepwiseConfig(
            decoding_mode="greedy",
            max_new_tokens=max_new_tokens,
            output_attentions=False,
            selected_layers=None,
            image_grid_shape=None,
            stop_on_eos=True,
            stop_on_sentence_end=False,
        )

    output = generate_stepwise_batch(
        wrapper=wrapper,
        image_paths=[samples[0]["image_path"]],
        prompts=[samples[0].get("prompt", "Describe this image.")],
        config=config,
        use_chat_template=use_chat_template_for_model(model_name),
    )

    print("Caption:", repr(output.caption))
    print("First token ids:", output.token_ids[:10])
    print("First token texts:", [repr(x) for x in output.token_texts[:10]])
    print("First token probs:", output.token_probs[:10])

    assert output.caption.strip() != ""
    assert len(output.steps) > 0

    first_step = output.steps[0]
    print("First step has attention:", first_step.image_attn_by_layer is not None)

    if llava_model(model_name):
        assert first_step.image_attn_by_layer is not None

    print("STEPWISE OK.")


def test_grounding_llava(dtype: str | None, max_new_tokens: int):
    print("\n========== GROUNDING: llava15_7b ==========")

    samples = load_one_sample()

    wrapper = build_wrapper_for_test(
        model_name="llava15_7b",
        dtype=dtype,
        need_attention=True,
    )

    from grounding import (
        compute_ads_from_step,
        get_kept_lh_from_step,
        get_object_mask_from_step,
    )

    output = generate_stepwise_batch(
        wrapper=wrapper,
        image_paths=[samples[0]["image_path"]],
        prompts=[samples[0].get("prompt", "Describe this image.")],
        config=StepwiseConfig(
            decoding_mode="greedy",
            max_new_tokens=max_new_tokens,
            output_attentions=True,
            selected_layers=[-8, -4, -1],
            image_grid_shape=(24, 24),
        ),
        use_chat_template=False,
    )

    step_record = None

    for step in output.steps:
        if step.image_attn_by_layer is not None:
            step_record = step
            break

    assert step_record is not None

    step = {
        "step": step_record.step,
        "token_id": step_record.token_id,
        "token_text": step_record.token_text,
        "image_attn_by_layer": step_record.image_attn_by_layer,
    }

    kept = get_kept_lh_from_step(
        step,
        image_grid_shape=(24, 24),
        attn_sum_threshold=0.49,
    )

    ads = compute_ads_from_step(
        step,
        image_grid_shape=(24, 24),
        foreground_ratio=0.10,
        top_n_heads=3,
        attn_sum_threshold=0.49,
    )

    mask = get_object_mask_from_step(
        step,
        image_grid_shape=(24, 24),
        top_n_heads=5,
        attn_sum_threshold=0.49,
    )

    print("Caption prefix:", repr(output.caption))
    print("Selected heads:", kept[:3])
    print("ADS:", ads)
    print("Mask:", None if mask is None else {"shape": mask.shape, "area": int(mask.astype(bool).sum())})

    assert ads is not None

    print("GROUNDING OK.")


def test_phg_llava(dtype: str | None, max_new_tokens: int):
    print("\n========== PHG: llava15_7b ==========")

    samples = load_one_sample()

    wrapper = build_wrapper_for_test(
        model_name="llava15_7b",
        dtype=dtype,
        need_attention=True,
    )

    rows = generate_phg_samples(
        wrapper=wrapper,
        samples=samples,
        config=PHGConfig(
            decoding_mode="greedy",
            max_rounds=2,
            max_new_tokens=max_new_tokens,
            min_new_tokens=3,

            top_k=3,
            accumulate_prob=0.5,

            iou_thresh=0.5,
            ads_thresh=0.45,
            ads_foreground_ratio=0.10,

            selected_layers=[-8, -4, -1],
            image_grid_shape=(24, 24),

            debug=False,
        ),
        use_chat_template=False,
        include_trace=True,
    )

    row = rows[0]

    print("Caption:", repr(row["caption"]))
    print("Objects:", row.get("objects"))
    print("Num decisions:", len(row.get("decision_trace", [])))

    for decision in row.get("decision_trace", []):
        print(
            {
                "round": decision.get("round"),
                "path": decision.get("path"),
                "stop_reason": decision.get("stop_reason"),
                "selected_candidate": decision.get("selected_candidate"),
            }
        )

    assert row["caption"].strip() != ""

    print("PHG OK.")


def run_one(
    model_name: str,
    suite: str,
    dtype: str | None,
    max_new_tokens: int,
):
    if suite == "direct":
        test_direct_generation(model_name, dtype, max_new_tokens)

    elif suite == "greedy":
        test_registry_greedy(model_name, dtype, max_new_tokens)

    elif suite == "vcd":
        test_registry_vcd(model_name, dtype, max_new_tokens)

    elif suite == "stepwise":
        test_stepwise(model_name, dtype, max_new_tokens)

    elif suite == "grounding":
        if not llava_model(model_name):
            print(f"SKIP: grounding currently stable only for llava15_7b, got {model_name}")
            return
        test_grounding_llava(dtype, max_new_tokens)

    elif suite == "phg":
        if not llava_model(model_name):
            print(f"SKIP: PHG currently stable only for llava15_7b, got {model_name}")
            return
        test_phg_llava(dtype, max_new_tokens)

    elif suite == "all_safe":
        test_direct_generation(model_name, dtype, max_new_tokens)
        test_registry_greedy(model_name, dtype, max_new_tokens)

        if not internvl_model(model_name):
            test_registry_vcd(model_name, dtype, max_new_tokens)

        if not internvl_model(model_name):
            test_stepwise(model_name, dtype, min(max_new_tokens, 16))

        if llava_model(model_name):
            test_grounding_llava(dtype, min(max_new_tokens, 8))
            test_phg_llava(dtype, min(max_new_tokens, 24))

    else:
        raise ValueError(f"Unknown suite: {suite}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=ALL_MODELS + ["all"],
        default="all",
    )

    parser.add_argument(
        "--suite",
        choices=[
            "direct",
            "greedy",
            "vcd",
            "stepwise",
            "grounding",
            "phg",
            "all_safe",
        ],
        default="direct",
    )

    parser.add_argument("--dtype", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--keep-going", action="store_true")

    args = parser.parse_args()

    models = ALL_MODELS if args.model == "all" else [args.model]

    failures: Dict[str, str] = {}

    for model_name in models:
        try:
            run_one(
                model_name=model_name,
                suite=args.suite,
                dtype=args.dtype,
                max_new_tokens=args.max_new_tokens,
            )

        except Exception as exc:
            failures[model_name] = repr(exc)

            print(f"\nFAILED: {model_name}")
            traceback.print_exc()

            if not args.keep_going:
                raise

    if failures:
        print("\n========== FAILURES ==========")
        for model_name, error in failures.items():
            print(model_name, "->", error)
    else:
        print("\nALL REQUESTED TESTS PASSED.")


if __name__ == "__main__":
    main()