# decoding/vcd.py

"""
Visual Contrastive Decoding for normal LVLM generation.

This file is the generic/manual VCD implementation.

Intended use:
    - LLaVA-style models where manual token-by-token decoding with
      past_key_values is stable.

Not intended use:
    - Qwen2-VL.
      Qwen2-VL requires its own VCD path in decoding/qwen_vcd.py because
      it depends on Qwen's official generate path, dynamic visual tokens,
      attention masks, and prepare_inputs_for_generation() / M-ROPE handling.

Design:
    - Tensor-level pixel_values noise.
    - Manual clean/CD forward passes.
    - Standard VCD scoring:
          score = (1 + alpha) log p_clean - alpha log p_cd
    - Standard VCD plausibility mask controlled by cd_beta:
          p_clean(token) >= beta * max_token p_clean(token)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from models.base import BaseLVLM, GenerationOutput, PathLike, TensorDict
from decoding.logits import (
    prepare_logits_for_selection,
    sample_or_argmax,
)
from utils.image_noise import add_diffusion_noise_to_tensor
from decoding.utils import get_model, get_processor, get_tokenizer, is_qwen_wrapper, move_inputs_to_model, get_eos_token_ids, get_fallback_token_id


@dataclass
class VCDConfig:
    max_new_tokens: int = 256
    use_cache: bool = True

    cd_alpha: float = 1.0
    cd_beta: float = 0.1
    noise_step: int = 500

    do_sample: bool = False
    temperature: float = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None

    image_tensor_key: str = "pixel_values"

    extra_forward_kwargs: Dict[str, Any] = field(default_factory=dict)

def apply_vcd_logits(
    clean_logits: torch.Tensor,
    cd_logits: torch.Tensor,
    cd_alpha: float = 1.0,
    cd_beta: float = 0.1,
) -> torch.Tensor:
    """
    Standard VCD in log-probability space.

    score = (1 + alpha) log p_clean - alpha log p_cd

    Plausibility mask:
        p_clean(token) >= beta * max_token p_clean(token)

    No model-specific token filtering.
    """

    cd_alpha = float(cd_alpha)
    cd_beta = float(max(cd_beta, 1e-12))

    if cd_alpha <= 0:
        return clean_logits.float()

    clean_log_probs = F.log_softmax(clean_logits.float(), dim=-1)
    cd_log_probs = F.log_softmax(cd_logits.float(), dim=-1)

    vcd_scores = (
        (1.0 + cd_alpha) * clean_log_probs
        - cd_alpha * cd_log_probs
    )

    max_clean_log_prob = clean_log_probs.max(
        dim=-1,
        keepdim=True,
    ).values

    cutoff = max_clean_log_prob + torch.log(
        torch.tensor(
            cd_beta,
            device=clean_logits.device,
            dtype=clean_log_probs.dtype,
        )
    )

    plausible = clean_log_probs >= cutoff

    vcd_scores = vcd_scores.masked_fill(
        ~plausible,
        -float("inf"),
    )

    all_inf = torch.isinf(vcd_scores).all(
        dim=-1,
        keepdim=True,
    )

    if all_inf.any():
        fallback = clean_logits.float()
        vcd_scores = torch.where(all_inf, fallback, vcd_scores)

    return vcd_scores

def _make_tensor_noised_inputs(
    inputs: TensorDict,
    image_tensor_key: str,
    noise_step: int,
) -> TensorDict:
    if image_tensor_key not in inputs:
        raise KeyError(
            f"Expected `{image_tensor_key}` in inputs. "
            f"Available keys: {list(inputs.keys())}"
        )

    noised: Dict[str, Any] = {}

    for key, value in inputs.items():
        if str(key).startswith("_"):
            continue

        if torch.is_tensor(value):
            noised[key] = value.clone()
        else:
            noised[key] = value

    noised[image_tensor_key] = add_diffusion_noise_to_tensor(
        noised[image_tensor_key],
        noise_step=noise_step,
    )

    return noised


def _make_contrastive_inputs(
    wrapper: BaseLVLM,
    clean_inputs_cpu: TensorDict,
    image_tensor_key: str,
    noise_step: int,
) -> TensorDict:
    """
    Build contrastive branch for generic/LLaVA VCD.

    Qwen2-VL must not use this path.
    Qwen2-VL is dispatched to decoding/qwen_vcd.py by decoding/registry.py.
    """

    if is_qwen_wrapper(wrapper):
        raise RuntimeError(
            "Qwen2-VL must not use decoding.vcd.VCDDecoder. "
            "Use decoding.qwen_vcd through decoding.registry instead."
        )

    return _make_tensor_noised_inputs(
        clean_inputs_cpu,
        image_tensor_key=image_tensor_key,
        noise_step=noise_step,
    )

def _decode_new_tokens(
    wrapper: BaseLVLM,
    processor: Any,
    new_token_ids: torch.Tensor,
) -> List[str]:
    if hasattr(wrapper, "batch_decode"):
        return [
            text.strip()
            for text in wrapper.batch_decode(new_token_ids)
        ]

    if hasattr(processor, "batch_decode"):
        captions = processor.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return [text.strip() for text in captions]

    captions = processor.tokenizer.batch_decode(
        new_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return [text.strip() for text in captions]


def _build_generation_output(
    captions: List[str],
    sequences: torch.Tensor,
    input_ids: torch.Tensor,
) -> GenerationOutput:
    attempts = [
        {
            "captions": captions,
            "sequences": sequences,
            "input_ids": input_ids,
            "raw_outputs": None,
        },
        {
            "captions": captions,
            "sequences": sequences,
            "input_ids": input_ids,
        },
        {
            "captions": captions,
            "sequences": sequences,
        },
        {
            "captions": captions,
        },
    ]

    for kwargs in attempts:
        try:
            return GenerationOutput(**kwargs)
        except TypeError:
            pass

    return SimpleNamespace(
        captions=captions,
        sequences=sequences,
        input_ids=input_ids,
        raw_outputs=None,
    )  # type: ignore[return-value]


class VCDDecoder:
    def __init__(self, config: Optional[VCDConfig] = None) -> None:
        self.config = config or VCDConfig()

    def build_forward_kwargs(
        self,
        **override_kwargs: Any,
    ) -> Dict[str, Any]:
        cfg = self.config

        kwargs: Dict[str, Any] = {
            "use_cache": cfg.use_cache,
            "return_dict": True,
        }

        kwargs.update(cfg.extra_forward_kwargs)
        kwargs.update(override_kwargs)

        return kwargs

    @torch.inference_mode()
    def generate_from_inputs(
        self,
        wrapper: BaseLVLM,
        inputs: TensorDict,
        **override_forward_kwargs: Any,
    ) -> GenerationOutput:
        cfg = self.config

        model = get_model(wrapper)
        processor = get_processor(wrapper)
        tokenizer = get_tokenizer(wrapper, processor)

        model.eval()

        clean_inputs_cpu = inputs

        cd_inputs_cpu = _make_contrastive_inputs(
            wrapper=wrapper,
            clean_inputs_cpu=clean_inputs_cpu,
            image_tensor_key=cfg.image_tensor_key,
            noise_step=cfg.noise_step,
        )

        clean_inputs = move_inputs_to_model(clean_inputs_cpu, model)
        cd_inputs = move_inputs_to_model(cd_inputs_cpu, model)

        input_ids = clean_inputs["input_ids"]

        if "attention_mask" in clean_inputs and clean_inputs["attention_mask"] is not None:
            attention_mask = clean_inputs["attention_mask"]
        else:
            attention_mask = torch.ones_like(input_ids)
            clean_inputs["attention_mask"] = attention_mask
            cd_inputs["attention_mask"] = attention_mask.clone()

        batch_size = input_ids.shape[0]
        prompt_len = input_ids.shape[1]

        generated_ids = input_ids.clone()

        past_key_values = None
        past_key_values_cd = None

        next_input_ids = input_ids
        next_input_ids_cd = cd_inputs["input_ids"]

        eos_token_ids = get_eos_token_ids(
            model=model,
            tokenizer=tokenizer,
        )

        fallback_token_id = get_fallback_token_id(
            processor=processor,
            eos_token_ids=eos_token_ids,
        )

        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=input_ids.device,
        )

        forward_kwargs = self.build_forward_kwargs(
            **override_forward_kwargs,
        )

        for step_idx in range(cfg.max_new_tokens):
            if step_idx == 0:
                clean_step_inputs = dict(clean_inputs)
                cd_step_inputs = dict(cd_inputs)
            else:
                clean_step_inputs = {
                    "input_ids": next_input_ids,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                }

                cd_step_inputs = {
                    "input_ids": next_input_ids_cd,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values_cd,
                }

            outputs = model(
                **clean_step_inputs,
                **forward_kwargs,
            )

            outputs_cd = model(
                **cd_step_inputs,
                **forward_kwargs,
            )

            clean_logits = outputs.logits[:, -1, :]
            cd_logits = outputs_cd.logits[:, -1, :]

            past_key_values = outputs.past_key_values
            past_key_values_cd = outputs_cd.past_key_values

            vcd_logits = apply_vcd_logits(
                clean_logits=clean_logits,
                cd_logits=cd_logits,
                cd_alpha=cfg.cd_alpha,
                cd_beta=cfg.cd_beta,
            )

            selection_logits = prepare_logits_for_selection(
                logits=vcd_logits,
                generated_token_ids=[],
                repetition_penalty=None,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
            )

            selection_logits[finished, :] = -float("inf")
            selection_logits[finished, fallback_token_id] = 0.0

            next_token = sample_or_argmax(
                selection_logits,
                do_sample=cfg.do_sample,
            )

            next_token = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_token, fallback_token_id),
                next_token,
            )

            generated_ids = torch.cat(
                [generated_ids, next_token],
                dim=-1,
            )

            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (batch_size, 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=-1,
            )

            next_input_ids = next_token
            next_input_ids_cd = next_token

            if len(eos_token_ids) > 0:
                newly_finished = torch.zeros_like(finished)

                for eos_id in eos_token_ids:
                    newly_finished |= next_token.squeeze(-1).eq(int(eos_id))

                finished |= newly_finished

                if finished.all():
                    break

        new_token_ids = generated_ids[:, prompt_len:]

        captions = _decode_new_tokens(
            wrapper=wrapper,
            processor=processor,
            new_token_ids=new_token_ids,
        )

        return _build_generation_output(
            captions=captions,
            sequences=generated_ids,
            input_ids=input_ids,
        )

    def generate_batch(
        self,
        wrapper: BaseLVLM,
        image_paths: Optional[Sequence[PathLike]] = None,
        images: Optional[Sequence[Any]] = None,
        prompts: Union[str, Sequence[str]] = "Describe this image.",
        use_chat_template: Optional[bool] = None,
        **prepare_kwargs: Any,
    ) -> GenerationOutput:
        inputs = wrapper.prepare_batch(
            image_paths=image_paths,
            images=images,
            prompts=prompts,
            use_chat_template=use_chat_template,
            **prepare_kwargs,
        )

        return self.generate_from_inputs(
            wrapper=wrapper,
            inputs=inputs,
        )

    def generate_samples(
        self,
        wrapper: BaseLVLM,
        samples: Sequence[Dict[str, Any]],
        image_key: str = "image_path",
        prompt_key: str = "prompt",
        id_key: str = "id",
        caption_key: str = "caption",
        use_chat_template: Optional[bool] = False,
        include_prompt: bool = False,
        **prepare_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        if len(samples) == 0:
            return []

        image_paths = [sample[image_key] for sample in samples]

        prompts = [
            sample.get(prompt_key, "Describe this image.")
            for sample in samples
        ]

        output = self.generate_batch(
            wrapper=wrapper,
            image_paths=image_paths,
            prompts=prompts,
            use_chat_template=use_chat_template,
            **prepare_kwargs,
        )

        rows: List[Dict[str, Any]] = []

        for sample, caption in zip(samples, output.captions):
            sample_id = sample[id_key]

            try:
                sample_id = int(sample_id)
            except Exception:
                pass

            row: Dict[str, Any] = {
                id_key: sample_id,
                caption_key: caption,
            }

            if include_prompt:
                row[prompt_key] = sample.get(
                    prompt_key,
                    "Describe this image.",
                )

            rows.append(row)

        return rows


def generate_vcd_from_inputs(
    wrapper: BaseLVLM,
    inputs: TensorDict,
    config: Optional[VCDConfig] = None,
    **override_forward_kwargs: Any,
) -> GenerationOutput:
    return VCDDecoder(config).generate_from_inputs(
        wrapper=wrapper,
        inputs=inputs,
        **override_forward_kwargs,
    )


def generate_vcd_batch(
    wrapper: BaseLVLM,
    image_paths: Optional[Sequence[PathLike]] = None,
    images: Optional[Sequence[Any]] = None,
    prompts: Union[str, Sequence[str]] = "Describe this image.",
    config: Optional[VCDConfig] = None,
    use_chat_template: Optional[bool] = None,
    **prepare_kwargs: Any,
) -> GenerationOutput:
    return VCDDecoder(config).generate_batch(
        wrapper=wrapper,
        image_paths=image_paths,
        images=images,
        prompts=prompts,
        use_chat_template=use_chat_template,
        **prepare_kwargs,
    )


def generate_vcd_samples(
    wrapper: BaseLVLM,
    samples: Sequence[Dict[str, Any]],
    config: Optional[VCDConfig] = None,
    image_key: str = "image_path",
    prompt_key: str = "prompt",
    id_key: str = "id",
    caption_key: str = "caption",
    use_chat_template: Optional[bool] = False,
    include_prompt: bool = False,
    **prepare_kwargs: Any,
) -> List[Dict[str, Any]]:
    return VCDDecoder(config).generate_samples(
        wrapper=wrapper,
        samples=samples,
        image_key=image_key,
        prompt_key=prompt_key,
        id_key=id_key,
        caption_key=caption_key,
        use_chat_template=use_chat_template,
        include_prompt=include_prompt,
        **prepare_kwargs,
    )