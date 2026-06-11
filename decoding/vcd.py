"""
Visual Contrastive Decoding for normal LVLM generation.

VCD = clean visual branch - noised visual branch.

For PHG-compatible VCD, use decoding/stepwise.py instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from models.base import BaseLVLM, GenerationOutput, PathLike, TensorDict
from decoding.logits import (
    distribution_stats,
    prepare_logits_for_selection,
    sample_or_argmax,
    top_k_top_p_filtering,
)


@dataclass
class VCDConfig:
    max_new_tokens: int = 256
    use_cache: bool = True

    cd_alpha: float = 1.0
    cd_beta: float = 0.1
    noise_step: int = 500

    # Add this
    noise_seed: Optional[int] = 42

    do_sample: bool = False
    temperature: float = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None

    # Add this
    suppress_token_ids: Optional[List[int]] = None

    image_tensor_key: str = "pixel_values"

    extra_forward_kwargs: Dict[str, Any] = field(default_factory=dict)


def add_diffusion_noise(
    image_tensor: torch.Tensor,
    noise_step: int = 500,
    noise_seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Add deterministic VCD-style diffusion noise to preprocessed visual tensor.

    If noise_seed is None, noise is random.
    If noise_seed is int, repeated runs are deterministic.
    """

    noise_step = int(max(0, min(999, noise_step)))

    device = image_tensor.device
    orig_dtype = image_tensor.dtype

    betas = torch.linspace(-6, 6, 1000, device=device, dtype=torch.float32)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    alphas = 1.0 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)

    sqrt_alpha_prod = torch.sqrt(alphas_prod[noise_step]).to(dtype=orig_dtype)
    sqrt_one_minus_alpha_prod = torch.sqrt(
        1.0 - alphas_prod[noise_step]
    ).to(dtype=orig_dtype)

    if noise_seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(noise_seed))

        noise = torch.randn(
            image_tensor.shape,
            device=device,
            dtype=orig_dtype,
            generator=generator,
        )
    else:
        noise = torch.randn_like(image_tensor)

    return sqrt_alpha_prod * image_tensor + sqrt_one_minus_alpha_prod * noise


def apply_vcd_logits(
    clean_logits: torch.Tensor,
    cd_logits: torch.Tensor,
    cd_alpha: float = 1.0,
    cd_beta: float = 0.1,
) -> torch.Tensor:
    """
    VCD logits:

        (1 + alpha) * clean_logits - alpha * cd_logits

    with adaptive plausibility constraint.
    """

    cd_alpha = float(cd_alpha)
    cd_beta = float(max(cd_beta, 1e-8))

    vcd_logits = (1.0 + cd_alpha) * clean_logits - cd_alpha * cd_logits

    cutoff = (
        torch.log(
            torch.tensor(
                cd_beta,
                device=clean_logits.device,
                dtype=clean_logits.dtype,
            )
        )
        + clean_logits.max(dim=-1, keepdim=True).values
    )

    masked = vcd_logits.masked_fill(clean_logits < cutoff, -float("inf"))

    all_inf = torch.isinf(masked).all(dim=-1, keepdim=True)

    return torch.where(all_inf, vcd_logits, masked)


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


def _move_inputs_to_model(inputs: TensorDict, model: Any) -> TensorDict:
    device, dtype = _get_model_device_and_dtype(model)

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


def _make_noised_inputs(
    inputs: TensorDict,
    image_tensor_key: str,
    noise_step: int,
    noise_seed: Optional[int] = None,
) -> TensorDict:
    if image_tensor_key not in inputs:
        raise KeyError(
            f"Expected `{image_tensor_key}` in inputs. "
            f"Available keys: {list(inputs.keys())}"
        )

    noised: Dict[str, Any] = {}

    for key, value in inputs.items():
        if torch.is_tensor(value):
            noised[key] = value.clone()
        else:
            noised[key] = value

    noised[image_tensor_key] = add_diffusion_noise(
        noised[image_tensor_key],
        noise_step=noise_step,
        noise_seed=noise_seed,
    )

    return noised


def _get_eos_token_ids(model: Any, processor: Any) -> List[int]:
    eos_token_id = getattr(model.generation_config, "eos_token_id", None)

    if eos_token_id is None and hasattr(processor, "tokenizer"):
        eos_token_id = getattr(processor.tokenizer, "eos_token_id", None)

    if eos_token_id is None:
        return []

    if isinstance(eos_token_id, int):
        return [int(eos_token_id)]

    return [int(x) for x in eos_token_id]


def _get_fallback_token_id(model: Any, processor: Any, eos_token_ids: Sequence[int]) -> int:
    if len(eos_token_ids) > 0:
        return int(eos_token_ids[0])

    if hasattr(processor, "tokenizer"):
        pad_id = getattr(processor.tokenizer, "pad_token_id", None)
        eos_id = getattr(processor.tokenizer, "eos_token_id", None)

        if pad_id is not None:
            return int(pad_id)

        if eos_id is not None:
            return int(eos_id)

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

    def build_forward_kwargs(self, **override_kwargs: Any) -> Dict[str, Any]:
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

        clean_inputs = _move_inputs_to_model(inputs, model)
        cd_inputs = _make_noised_inputs(
            clean_inputs,
            image_tensor_key=cfg.image_tensor_key,
            noise_step=cfg.noise_step,
            noise_seed=cfg.noise_seed,
        )

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
        next_input_ids_cd = input_ids

        eos_token_ids = _get_eos_token_ids(model, processor)
        fallback_token_id = _get_fallback_token_id(model, processor, eos_token_ids)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

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
                    "input_ids": next_input_ids_cd,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values_cd,
                }

            outputs = model(**clean_step_inputs, **forward_kwargs)
            outputs_cd = model(**cd_step_inputs, **forward_kwargs)

            clean_logits = outputs.logits[:, -1, :]
            cd_logits = outputs_cd.logits[:, -1, :]

            past_key_values = outputs.past_key_values
            past_key_values_cd = outputs_cd.past_key_values

            selection_logits = apply_vcd_logits(
                clean_logits=clean_logits,
                cd_logits=cd_logits,
                cd_alpha=cfg.cd_alpha,
                cd_beta=cfg.cd_beta,
            )

            if cfg.suppress_token_ids is not None:
                for token_id in cfg.suppress_token_ids:
                    token_id = int(token_id)
                    if 0 <= token_id < selection_logits.shape[-1]:
                        selection_logits[:, token_id] = -float("inf")
            
            selection_logits = prepare_logits_for_selection(
                logits=selection_logits,
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

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

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

        return self.generate_from_inputs(wrapper, inputs)

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
        prompts = [sample.get(prompt_key, "Describe this image.") for sample in samples]

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
                row[prompt_key] = sample.get(prompt_key, "Describe this image.")

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