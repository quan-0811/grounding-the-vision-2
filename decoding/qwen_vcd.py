# decoding/qwen_vcd.py

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList

from models.base import BaseLVLM, GenerationOutput, PathLike, TensorDict


@dataclass
class QwenVCDConfig:
    max_new_tokens: int = 256

    cd_alpha: float = 1.0
    cd_beta: float = 0.1
    noise_step: int = 500

    do_sample: bool = False
    temperature: float = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None

    use_cache: bool = True


def _strip_private_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in inputs.items()
        if not str(key).startswith("_")
    }


def _move_inputs_to_model(
    inputs: Dict[str, Any],
    model: Any,
) -> Dict[str, Any]:
    param = next(model.parameters())
    device = param.device
    dtype = param.dtype

    moved = {}

    for key, value in _strip_private_inputs(inputs).items():
        if torch.is_tensor(value):
            if value.is_floating_point():
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value

    return moved


def _extend_attention_mask(
    base_attention_mask: torch.Tensor,
    current_input_ids: torch.Tensor,
) -> torch.Tensor:
    batch_size, current_len = current_input_ids.shape
    base_len = base_attention_mask.shape[1]

    if current_len == base_len:
        return base_attention_mask

    if current_len < base_len:
        return base_attention_mask[:, :current_len]

    extra_len = current_len - base_len

    extra = torch.ones(
        (batch_size, extra_len),
        dtype=base_attention_mask.dtype,
        device=base_attention_mask.device,
    )

    return torch.cat([base_attention_mask, extra], dim=1)


def _decode_generated(
    wrapper: BaseLVLM,
    output_ids: torch.Tensor,
    input_ids: torch.Tensor,
) -> List[str]:
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(input_ids, output_ids)
    ]

    pad_id = getattr(wrapper.tokenizer, "pad_token_id", 0)

    new_token_ids = torch.nn.utils.rnn.pad_sequence(
        generated_ids_trimmed,
        batch_first=True,
        padding_value=pad_id,
    )

    return wrapper.batch_decode(new_token_ids)


def _pure_vcd_scores(
    clean_logits: torch.Tensor,
    cd_logits: torch.Tensor,
    cd_alpha: float,
    cd_beta: float,
) -> torch.Tensor:
    """
    Pure VCD score.

    VCD:
        score = (1 + alpha) * log p_clean - alpha * log p_noised

    Plausibility mask:
        keep tokens whose clean probability >= beta * max_clean_probability

    No Qwen-specific output filtering.
    """

    clean_log_probs = F.log_softmax(clean_logits.float(), dim=-1)
    cd_log_probs = F.log_softmax(cd_logits.float(), dim=-1)

    scores = (
        (1.0 + float(cd_alpha)) * clean_log_probs
        - float(cd_alpha) * cd_log_probs
    )

    max_clean_log_prob = clean_log_probs.max(dim=-1, keepdim=True).values

    cutoff = max_clean_log_prob + torch.log(
        torch.tensor(
            max(float(cd_beta), 1e-12),
            device=clean_logits.device,
            dtype=clean_log_probs.dtype,
        )
    )

    plausible = clean_log_probs >= cutoff
    scores = scores.masked_fill(~plausible, -float("inf"))

    all_inf = torch.isinf(scores).all(dim=-1, keepdim=True)

    if all_inf.any():
        clean_top1 = torch.argmax(clean_logits, dim=-1, keepdim=True)

        fallback = torch.full_like(scores, -float("inf"))
        fallback.scatter_(dim=-1, index=clean_top1, value=0.0)

        scores = torch.where(all_inf, fallback, scores)

    return scores


class QwenVCDLogitsProcessor(LogitsProcessor):
    """
    Pure VCD LogitsProcessor for Qwen2-VL.

    Clean branch:
        model.generate() handles it.

    Contrastive branch:
        noised image + same generated prefix.

    Critical Qwen fix:
        use model.prepare_inputs_for_generation() so Qwen computes M-ROPE
        position_ids / rope_deltas correctly for the noised branch.
    """

    def __init__(
        self,
        model: Any,
        cd_inputs: Dict[str, Any],
        config: QwenVCDConfig,
    ) -> None:
        self.model = model
        self.cd_inputs = cd_inputs
        self.config = config

        if "attention_mask" not in cd_inputs:
            raise ValueError("Qwen VCD requires attention_mask in cd_inputs.")

        self.base_attention_mask = cd_inputs["attention_mask"]

    @torch.inference_mode()
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        attention_mask = _extend_attention_mask(
            base_attention_mask=self.base_attention_mask,
            current_input_ids=input_ids,
        )

        cd_model_inputs = dict(self.cd_inputs)

        cache_position = torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
            dtype=torch.long,
        )

        prepared = self.model.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=None,
            attention_mask=attention_mask,
            cache_position=cache_position,
            use_cache=False,
            pixel_values=cd_model_inputs.get("pixel_values", None),
            pixel_values_videos=cd_model_inputs.get("pixel_values_videos", None),
            image_grid_thw=cd_model_inputs.get("image_grid_thw", None),
            video_grid_thw=cd_model_inputs.get("video_grid_thw", None),
        )
        
        # prepare_inputs_for_generation may already include use_cache / return_dict.
        # Set them inside the dict instead of passing duplicate kwargs.
        prepared["use_cache"] = False
        prepared["return_dict"] = True
        
        outputs_cd = self.model(**prepared)

        cd_logits = outputs_cd.logits[:, -1, :]

        return _pure_vcd_scores(
            clean_logits=scores,
            cd_logits=cd_logits,
            cd_alpha=self.config.cd_alpha,
            cd_beta=self.config.cd_beta,
        )


class QwenVCDDecoder:
    def __init__(self, config: Optional[QwenVCDConfig] = None) -> None:
        self.config = config or QwenVCDConfig()

    @torch.inference_mode()
    def generate_from_inputs(
        self,
        wrapper: BaseLVLM,
        inputs: TensorDict,
    ) -> GenerationOutput:
        cfg = self.config

        model = wrapper.model
        tokenizer = wrapper.tokenizer

        model.eval()

        if not hasattr(wrapper, "prepare_vcd_inputs"):
            raise AttributeError(
                "Qwen VCD requires wrapper.prepare_vcd_inputs()."
            )

        clean_inputs = _move_inputs_to_model(inputs, model)

        cd_inputs_raw = wrapper.prepare_vcd_inputs(
            inputs,
            noise_step=cfg.noise_step,
        )

        cd_inputs = _move_inputs_to_model(cd_inputs_raw, model)

        logits_processor = LogitsProcessorList(
            [
                QwenVCDLogitsProcessor(
                    model=model,
                    cd_inputs=cd_inputs,
                    config=cfg,
                )
            ]
        )

        generation_config = copy.deepcopy(model.generation_config)
        generation_config.do_sample = bool(cfg.do_sample)
        generation_config.pad_token_id = tokenizer.pad_token_id
        generation_config.eos_token_id = tokenizer.eos_token_id

        if cfg.do_sample:
            generation_config.temperature = cfg.temperature

            if cfg.top_p is not None:
                generation_config.top_p = cfg.top_p

            if cfg.top_k is not None:
                generation_config.top_k = cfg.top_k
        else:
            # Neutralize Qwen generation_config sampling defaults.
            generation_config.temperature = 1.0
            generation_config.top_p = 1.0
            generation_config.top_k = 50

        output_ids = model.generate(
            **clean_inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            max_new_tokens=cfg.max_new_tokens,
            use_cache=cfg.use_cache,
        )

        captions = _decode_generated(
            wrapper=wrapper,
            output_ids=output_ids,
            input_ids=clean_inputs["input_ids"],
        )

        return GenerationOutput(
            captions=captions,
            sequences=output_ids,
            input_ids=clean_inputs["input_ids"],
            raw_outputs=output_ids,
        )

    def generate_batch(
        self,
        wrapper: BaseLVLM,
        image_paths: Optional[Sequence[PathLike]] = None,
        images: Optional[Sequence[Any]] = None,
        prompts: Union[str, Sequence[str]] = "Describe this image.",
        use_chat_template: Optional[bool] = True,
        **prepare_kwargs: Any,
    ) -> GenerationOutput:
        inputs = wrapper.prepare_batch(
            image_paths=image_paths,
            images=images,
            prompts=prompts,
            use_chat_template=True,
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
        use_chat_template: Optional[bool] = True,
        include_prompt: bool = False,
        **prepare_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        image_paths = [sample[image_key] for sample in samples]

        prompts = [
            sample.get(prompt_key, "Describe this image.")
            for sample in samples
        ]

        output = self.generate_batch(
            wrapper=wrapper,
            image_paths=image_paths,
            prompts=prompts,
            use_chat_template=True,
            **prepare_kwargs,
        )

        rows = []

        for sample, caption in zip(samples, output.captions):
            sample_id = sample[id_key]

            try:
                sample_id = int(sample_id)
            except Exception:
                pass

            row = {
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


def _coerce_config(config: Any) -> QwenVCDConfig:
    if config is None:
        return QwenVCDConfig()

    if isinstance(config, QwenVCDConfig):
        return config

    return QwenVCDConfig(
        max_new_tokens=getattr(config, "max_new_tokens", 256),
        cd_alpha=getattr(config, "cd_alpha", 1.0),
        cd_beta=getattr(config, "cd_beta", 0.1),
        noise_step=getattr(config, "noise_step", 500),
        do_sample=getattr(config, "do_sample", False),
        temperature=getattr(config, "temperature", 1.0),
        top_p=getattr(config, "top_p", None),
        top_k=getattr(config, "top_k", None),
        use_cache=getattr(config, "use_cache", True),
    )


def generate_qwen_vcd_from_inputs(
    wrapper: BaseLVLM,
    inputs: TensorDict,
    config: Optional[Any] = None,
) -> GenerationOutput:
    return QwenVCDDecoder(
        _coerce_config(config)
    ).generate_from_inputs(
        wrapper=wrapper,
        inputs=inputs,
    )


def generate_qwen_vcd_batch(
    wrapper: BaseLVLM,
    image_paths: Optional[Sequence[PathLike]] = None,
    images: Optional[Sequence[Any]] = None,
    prompts: Union[str, Sequence[str]] = "Describe this image.",
    config: Optional[Any] = None,
    use_chat_template: Optional[bool] = True,
    **prepare_kwargs: Any,
) -> GenerationOutput:
    return QwenVCDDecoder(
        _coerce_config(config)
    ).generate_batch(
        wrapper=wrapper,
        image_paths=image_paths,
        images=images,
        prompts=prompts,
        use_chat_template=True,
        **prepare_kwargs,
    )


def generate_qwen_vcd_samples(
    wrapper: BaseLVLM,
    samples: Sequence[Dict[str, Any]],
    config: Optional[Any] = None,
    image_key: str = "image_path",
    prompt_key: str = "prompt",
    id_key: str = "id",
    caption_key: str = "caption",
    use_chat_template: Optional[bool] = True,
    include_prompt: bool = False,
    **prepare_kwargs: Any,
) -> List[Dict[str, Any]]:
    return QwenVCDDecoder(
        _coerce_config(config)
    ).generate_samples(
        wrapper=wrapper,
        samples=samples,
        config=config,
        image_key=image_key,
        prompt_key=prompt_key,
        id_key=id_key,
        caption_key=caption_key,
        use_chat_template=True,
        include_prompt=include_prompt,
        **prepare_kwargs,
    )