"""
Visual Contrastive Decoding for normal LVLM generation.

Revised design:
    - If wrapper has prepare_vcd_inputs(), use it.
      This is required for Qwen2-VL.
    - Otherwise, fallback to tensor-level pixel_values noise.
      This works for LLaVA.
    - No min_new_tokens.
    - No EOS forcing.
    - Conservative clean-guided VCD.
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


@dataclass
class VCDConfig:
    max_new_tokens: int = 256
    use_cache: bool = True

    # Conservative defaults.
    cd_alpha: float = 0.2
    cd_beta: float = 0.5
    noise_step: int = 500

    do_sample: bool = False
    temperature: float = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None

    image_tensor_key: str = "pixel_values"

    # Clean branch safety.
    clean_confidence_gate: Optional[float] = 0.35
    fallback_to_clean: bool = True
    fallback_clean_prob_ratio: float = 0.30

    extra_forward_kwargs: Dict[str, Any] = field(default_factory=dict)


def add_diffusion_noise(
    image_tensor: torch.Tensor,
    noise_step: int = 500,
) -> torch.Tensor:
    """
    Backward-compatible export for stepwise.py.
    """

    return add_diffusion_noise_to_tensor(
        image_tensor,
        noise_step=noise_step,
    )


def apply_vcd_logits(
    clean_logits: torch.Tensor,
    cd_logits: torch.Tensor,
    cd_alpha: float = 0.2,
    cd_beta: float = 0.5,
) -> torch.Tensor:
    """
    Conservative VCD in log-probability space.

    score = (1 + alpha) log p_clean - alpha log p_cd

    Then keep only tokens plausible under clean branch:
        p_clean(token) >= beta * max_clean_prob
    """

    cd_alpha = float(cd_alpha)

    if cd_alpha <= 0:
        return clean_logits.float()

    cd_beta = float(max(cd_beta, 1e-8))

    clean_log_probs = F.log_softmax(clean_logits.float(), dim=-1)
    cd_log_probs = F.log_softmax(cd_logits.float(), dim=-1)

    vcd_scores = (1.0 + cd_alpha) * clean_log_probs - cd_alpha * cd_log_probs

    max_clean_log_prob = clean_log_probs.max(dim=-1, keepdim=True).values

    cutoff = max_clean_log_prob + torch.log(
        torch.tensor(
            cd_beta,
            device=clean_logits.device,
            dtype=clean_log_probs.dtype,
        )
    )

    plausible = clean_log_probs >= cutoff

    masked_scores = vcd_scores.masked_fill(~plausible, -float("inf"))

    all_inf = torch.isinf(masked_scores).all(dim=-1, keepdim=True)

    return torch.where(all_inf, clean_logits.float(), masked_scores)


def apply_clean_confidence_gate(
    clean_logits: torch.Tensor,
    vcd_logits: torch.Tensor,
    gate: Optional[float],
) -> torch.Tensor:
    """
    If clean branch is already confident, use clean logits.

    This prevents VCD from damaging fluent high-confidence tokens.
    """

    if gate is None:
        return vcd_logits

    clean_probs = F.softmax(clean_logits.float(), dim=-1)
    clean_max = clean_probs.max(dim=-1, keepdim=True).values

    use_clean = clean_max >= float(gate)

    return torch.where(use_clean, clean_logits.float(), vcd_logits)


def maybe_fallback_to_clean(
    selected_token: torch.Tensor,
    clean_logits: torch.Tensor,
    fallback_to_clean: bool,
    fallback_clean_prob_ratio: float,
    do_sample: bool,
) -> torch.Tensor:
    """
    If selected VCD token is too weak under the clean branch,
    replace with clean argmax.
    """

    if not fallback_to_clean:
        return selected_token

    if do_sample:
        return selected_token

    clean_probs = F.softmax(clean_logits.float(), dim=-1)

    selected_clean_prob = clean_probs.gather(
        dim=-1,
        index=selected_token,
    )

    max_clean_prob = clean_probs.max(dim=-1, keepdim=True).values

    too_weak = selected_clean_prob < (
        float(fallback_clean_prob_ratio) * max_clean_prob
    )

    clean_argmax = torch.argmax(
        clean_logits,
        dim=-1,
        keepdim=True,
    )

    return torch.where(
        too_weak,
        clean_argmax,
        selected_token,
    )


def _get_model(wrapper: BaseLVLM) -> Any:
    if hasattr(wrapper, "model"):
        return wrapper.model

    if hasattr(wrapper, "get_model"):
        return wrapper.get_model()

    raise AttributeError("Wrapper must expose `.model` or `.get_model()`.")


def _get_processor(wrapper: BaseLVLM) -> Any:
    if hasattr(wrapper, "processor"):
        return wrapper.processor

    if hasattr(wrapper, "get_processor"):
        return wrapper.get_processor()

    raise AttributeError("Wrapper must expose `.processor` or `.get_processor()`.")


def _get_model_device_and_dtype(model: Any) -> tuple[torch.device, torch.dtype]:
    param = next(model.parameters())
    return param.device, param.dtype


def _strip_private_inputs(inputs: TensorDict) -> TensorDict:
    """
    Remove private metadata and unsupported non-tensor fields.
    """

    allowed_non_tensor_keys = set()

    allowed_tensor_keys = {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
        "position_ids",
        "token_type_ids",
    }

    out: Dict[str, Any] = {}

    for key, value in inputs.items():
        if key.startswith("_"):
            continue

        if key in allowed_tensor_keys:
            out[key] = value
            continue

        if key in allowed_non_tensor_keys:
            out[key] = value
            continue

    return out


def _move_inputs_to_model(
    inputs: TensorDict,
    model: Any,
) -> TensorDict:
    device, dtype = _get_model_device_and_dtype(model)

    inputs = _strip_private_inputs(inputs)

    moved: Dict[str, Any] = {}

    for key, value in inputs.items():
        if torch.is_tensor(value):
            if value.is_floating_point():
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value

    return moved


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
        if key.startswith("_"):
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
    Build contrastive branch.

    Priority:
        1. wrapper.prepare_vcd_inputs()
        2. tensor-level pixel_values noise
    """

    if hasattr(wrapper, "prepare_vcd_inputs"):
        return wrapper.prepare_vcd_inputs(
            clean_inputs_cpu,
            noise_step=noise_step,
        )

    return _make_tensor_noised_inputs(
        clean_inputs_cpu,
        image_tensor_key=image_tensor_key,
        noise_step=noise_step,
    )


def _get_eos_token_ids(
    model: Any,
    processor: Any,
) -> List[int]:
    eos_token_id = getattr(model.generation_config, "eos_token_id", None)

    if eos_token_id is None and hasattr(processor, "tokenizer"):
        eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)

    if eos_token_id is None:
        return []

    if isinstance(eos_token_id, int):
        return [int(eos_token_id)]

    return [int(x) for x in eos_token_id]


def _get_fallback_token_id(
    processor: Any,
    eos_token_ids: Sequence[int],
) -> int:
    if hasattr(processor, "tokenizer"):
        pad_id = getattr(processor.tokenizer, "pad_token_id", None)
        eos_id = getattr(processor.tokenizer, "eos_token_id", None)

        if pad_id is not None:
            return int(pad_id)

        if eos_id is not None:
            return int(eos_id)

    if len(eos_token_ids) > 0:
        return int(eos_token_ids[0])

    return 0


def _decode_new_tokens(
    wrapper: BaseLVLM,
    processor: Any,
    new_token_ids: torch.Tensor,
) -> List[str]:
    if hasattr(wrapper, "batch_decode"):
        return [x.strip() for x in wrapper.batch_decode(new_token_ids)]

    if hasattr(processor, "batch_decode"):
        captions = processor.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return [x.strip() for x in captions]

    captions = processor.tokenizer.batch_decode(
        new_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return [x.strip() for x in captions]


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

        model = _get_model(wrapper)
        processor = _get_processor(wrapper)

        model.eval()

        # Keep CPU/private version for wrapper.prepare_vcd_inputs().
        clean_inputs_cpu = inputs

        cd_inputs_cpu = _make_contrastive_inputs(
            wrapper=wrapper,
            clean_inputs_cpu=clean_inputs_cpu,
            image_tensor_key=cfg.image_tensor_key,
            noise_step=cfg.noise_step,
        )

        clean_inputs = _move_inputs_to_model(clean_inputs_cpu, model)
        cd_inputs = _move_inputs_to_model(cd_inputs_cpu, model)

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

        eos_token_ids = _get_eos_token_ids(model, processor)

        fallback_token_id = _get_fallback_token_id(
            processor=processor,
            eos_token_ids=eos_token_ids,
        )

        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=input_ids.device,
        )

        forward_kwargs = self.build_forward_kwargs(**override_forward_kwargs)

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
                    "input_ids": next_input_ids,
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

            vcd_logits = apply_clean_confidence_gate(
                clean_logits=clean_logits,
                vcd_logits=vcd_logits,
                gate=cfg.clean_confidence_gate,
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

            next_token = maybe_fallback_to_clean(
                selected_token=next_token,
                clean_logits=clean_logits,
                fallback_to_clean=cfg.fallback_to_clean,
                fallback_clean_prob_ratio=cfg.fallback_clean_prob_ratio,
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
            row: Dict[str, Any] = {
                id_key: int(sample[id_key]),
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