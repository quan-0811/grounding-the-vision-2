"""
Stepwise decoding for PHG.

This is the PHG-compatible decoding backend for:

    greedy + PHG
    DoLA + PHG
    VCD + PHG

Normal batched generation should use greedy.py, dola.py, or vcd.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from models.base import BaseLVLM, PathLike, TensorDict

from decoding.logits import (
    apply_relative_top_filter,
    distribution_stats,
    prepare_logits_for_selection,
    sample_or_argmax,
)
from decoding.vcd import apply_vcd_logits
from grounding.attention import extract_image_attn_by_layer
from decoding.utils import (
    decode_token,
    get_eos_token_ids,
    get_model,
    get_processor,
    get_tokenizer,
    make_noised_inputs,
    move_inputs_to_model,
)

StepwiseMode = Literal["greedy", "dola", "vcd"]

@dataclass
class StepRecord:
    step: int
    token_id: int
    token_text: str

    token_prob: float
    max_prob: float
    entropy: float

    decoding_mode: str

    image_attn_by_layer: Optional[Dict[int, torch.Tensor]] = None

    dola_premature_layer: Optional[int] = None

    clean_token_prob: Optional[float] = None
    cd_token_prob: Optional[float] = None


@dataclass
class StepwiseOutput:
    caption: str
    token_ids: List[int]
    token_texts: List[str]
    token_probs: List[float]
    steps: List[StepRecord]

    sequences: Optional[torch.Tensor] = None
    input_ids: Optional[torch.Tensor] = None
    raw_outputs: Optional[Any] = None


@dataclass
class StepwiseConfig:
    decoding_mode: StepwiseMode = "greedy"

    max_new_tokens: int = 256
    min_new_tokens: int = 0
    use_cache: bool = True

    output_attentions: bool = True
    selected_layers: Optional[Sequence[int]] = None
    keep_attn_on_cpu: bool = True
    image_grid_shape: Optional[Tuple[int, int]] = None

    do_sample: bool = False
    temperature: float = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None

    stop_on_eos: bool = True
    stop_on_sentence_end: bool = False
    sentence_end_texts: Tuple[str, ...] = (".", "!", "?", "\n")

    forced_token_ids: Optional[Sequence[int]] = None

    dola_layers: Union[str, Sequence[int]] = "low"
    dola_relative_top: Optional[float] = 0.1
    dola_select_strategy: Literal["js", "first", "last"] = "js"

    cd_alpha: float = 1.0
    cd_beta: float = 0.1
    noise_step: int = 500
    image_tensor_key: str = "pixel_values"

    extra_forward_kwargs: Dict[str, Any] = field(default_factory=dict)

def _decode_caption(
    wrapper: BaseLVLM,
    tokenizer: Any,
    token_ids: Sequence[int],
) -> str:
    tensor_ids = torch.tensor([list(token_ids)], dtype=torch.long)

    if hasattr(wrapper, "batch_decode"):
        return wrapper.batch_decode(tensor_ids)[0].strip()

    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()


def _resolve_num_layers(model: Any, hidden_states: Sequence[torch.Tensor]) -> int:
    if hidden_states is not None and len(hidden_states) > 1:
        return len(hidden_states) - 1

    for attr_path in [
        ("language_model", "config", "num_hidden_layers"),
        ("config", "text_config", "num_hidden_layers"),
        ("config", "num_hidden_layers"),
    ]:
        obj = model
        ok = True

        for attr in attr_path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)

        if ok and obj is not None:
            return int(obj)

    raise ValueError("Could not resolve number of layers.")


def _resolve_dola_candidate_layers(
    dola_layers: Union[str, Sequence[int]],
    num_layers: int,
) -> List[int]:
    if isinstance(dola_layers, str):
        name = dola_layers.lower()

        if name == "low":
            return list(range(0, max(1, num_layers // 2)))

        if name == "high":
            return list(range(max(0, num_layers // 2), num_layers))

        if name == "all":
            return list(range(num_layers))

        raise ValueError(f"Unsupported dola_layers: {dola_layers}")

    resolved = []

    for layer_id in dola_layers:
        layer_id = int(layer_id)

        if layer_id < 0:
            layer_id = num_layers + layer_id

        if layer_id < 0 or layer_id >= num_layers:
            raise ValueError(
                f"DoLA layer {layer_id} is out of range for {num_layers} layers."
            )

        resolved.append(layer_id)

    if len(resolved) == 0:
        raise ValueError("DoLA candidate layer list is empty.")

    return resolved


def _get_lm_head(model: Any) -> Any:
    if hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
        if head is not None:
            return head

    if hasattr(model, "language_model") and hasattr(model.language_model, "lm_head"):
        return model.language_model.lm_head

    if hasattr(model, "lm_head"):
        return model.lm_head

    raise AttributeError("Could not resolve LM head.")


def _project_hidden_to_logits(
    model: Any,
    hidden_state: torch.Tensor,
) -> torch.Tensor:
    if hidden_state.dim() == 3:
        hidden_state = hidden_state[:, -1, :]

    return _get_lm_head(model)(hidden_state)


def _js_divergence(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
) -> torch.Tensor:
    p = F.softmax(p_logits.float(), dim=-1)
    q = F.softmax(q_logits.float(), dim=-1)

    m = 0.5 * (p + q)

    kl_pm = torch.sum(p * (torch.log(p + 1e-12) - torch.log(m + 1e-12)), dim=-1)
    kl_qm = torch.sum(q * (torch.log(q + 1e-12) - torch.log(m + 1e-12)), dim=-1)

    return 0.5 * (kl_pm + kl_qm)


def _compute_dola_logits(
    model: Any,
    outputs: Any,
    cfg: StepwiseConfig,
) -> tuple[torch.Tensor, Optional[int]]:
    if outputs.hidden_states is None:
        raise ValueError("Stepwise DoLA requires output_hidden_states=True.")

    hidden_states = outputs.hidden_states
    num_layers = _resolve_num_layers(model, hidden_states)

    candidate_layers = _resolve_dola_candidate_layers(
        cfg.dola_layers,
        num_layers=num_layers,
    )

    mature_logits = outputs.logits[:, -1, :]

    candidate_logits: List[tuple[int, torch.Tensor]] = []

    for layer_id in candidate_layers:
        h = hidden_states[layer_id + 1]
        premature_logits = _project_hidden_to_logits(model, h[:, -1:, :])
        candidate_logits.append((layer_id, premature_logits))

    if cfg.dola_select_strategy == "first":
        selected_layer, premature_logits = candidate_logits[0]

    elif cfg.dola_select_strategy == "last":
        selected_layer, premature_logits = candidate_logits[-1]

    else:
        divergences = [
            _js_divergence(mature_logits, prem_logits)
            for _, prem_logits in candidate_logits
        ]

        scores = torch.stack(divergences, dim=0)[:, 0]
        best_idx = int(torch.argmax(scores).item())

        selected_layer, premature_logits = candidate_logits[best_idx]

    contrast_logits = mature_logits - premature_logits

    contrast_logits = apply_relative_top_filter(
        contrast_logits=contrast_logits,
        mature_logits=mature_logits,
        relative_top=cfg.dola_relative_top,
    )

    return contrast_logits, int(selected_layer)


def _get_clean_and_cd_token_probs(
    clean_logits: torch.Tensor,
    cd_logits: torch.Tensor,
    selected_token: torch.Tensor,
) -> tuple[float, float]:
    clean_probs = F.softmax(clean_logits.float(), dim=-1)
    cd_probs = F.softmax(cd_logits.float(), dim=-1)

    clean_prob = float(clean_probs.gather(-1, selected_token)[0, 0].item())
    cd_prob = float(cd_probs.gather(-1, selected_token)[0, 0].item())

    return clean_prob, cd_prob


class StepwiseDecoder:
    def __init__(self, config: Optional[StepwiseConfig] = None) -> None:
        self.config = config or StepwiseConfig()

    def build_forward_kwargs(self, **override_kwargs: Any) -> Dict[str, Any]:
        cfg = self.config

        kwargs: Dict[str, Any] = {
            "use_cache": cfg.use_cache,
            "return_dict": True,
        }

        kwargs["output_hidden_states"] = cfg.decoding_mode == "dola"

        kwargs.update(cfg.extra_forward_kwargs)
        kwargs.update(override_kwargs)

        return kwargs

    def _select_next_token(
        self,
        model: Any,
        outputs: Any,
        generated_token_ids: Sequence[int],
        forced_token_id: Optional[int] = None,
        cd_outputs: Optional[Any] = None,
    ):
        cfg = self.config

        dola_premature_layer = None
        clean_token_prob = None
        cd_token_prob = None

        if cfg.decoding_mode == "greedy":
            selection_logits = outputs.logits[:, -1, :]

        elif cfg.decoding_mode == "dola":
            selection_logits, dola_premature_layer = _compute_dola_logits(
                model=model,
                outputs=outputs,
                cfg=cfg,
            )

        elif cfg.decoding_mode == "vcd":
            if cd_outputs is None:
                raise ValueError("VCD stepwise decoding requires cd_outputs.")

            clean_logits = outputs.logits[:, -1, :]
            cd_logits = cd_outputs.logits[:, -1, :]

            selection_logits = apply_vcd_logits(
                clean_logits=clean_logits,
                cd_logits=cd_logits,
                cd_alpha=cfg.cd_alpha,
                cd_beta=cfg.cd_beta,
            )

        else:
            raise ValueError(f"Unsupported decoding mode: {cfg.decoding_mode}")

        selection_logits = prepare_logits_for_selection(
            logits=selection_logits,
            generated_token_ids=generated_token_ids,
            repetition_penalty=cfg.repetition_penalty,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
        )

        if forced_token_id is not None:
            next_token = torch.tensor(
                [[int(forced_token_id)]],
                dtype=torch.long,
                device=selection_logits.device,
            )
        else:
            next_token = sample_or_argmax(
                selection_logits,
                do_sample=cfg.do_sample,
            )

        token_prob, max_prob, entropy = distribution_stats(
            selection_logits,
            selected_token=next_token,
        )

        if cfg.decoding_mode == "vcd":
            clean_token_prob, cd_token_prob = _get_clean_and_cd_token_probs(
                clean_logits=outputs.logits[:, -1, :],
                cd_logits=cd_outputs.logits[:, -1, :],
                selected_token=next_token,
            )

        return (
            next_token,
            selection_logits,
            token_prob,
            max_prob,
            entropy,
            dola_premature_layer,
            clean_token_prob,
            cd_token_prob,
        )

    @torch.inference_mode()
    def generate_from_inputs(
        self,
        wrapper: BaseLVLM,
        inputs: TensorDict,
        **override_forward_kwargs: Any,
    ) -> StepwiseOutput:
        cfg = self.config

        model = get_model(wrapper)
        processor = get_processor(wrapper)
        tokenizer = get_tokenizer(wrapper, processor)

        model.eval()

        inputs = move_inputs_to_model(inputs, model)

        if inputs["input_ids"].shape[0] != 1:
            raise ValueError("Stepwise decoding currently expects batch size 1.")

        input_ids = inputs["input_ids"]

        if "attention_mask" in inputs and inputs["attention_mask"] is not None:
            attention_mask = inputs["attention_mask"]
        else:
            attention_mask = torch.ones_like(input_ids)
            inputs["attention_mask"] = attention_mask

        forward_kwargs = self.build_forward_kwargs(**override_forward_kwargs)

        cd_inputs = None

        if cfg.decoding_mode == "vcd":
            cd_inputs = make_noised_inputs(
                inputs,
                image_tensor_key=cfg.image_tensor_key,
                noise_step=cfg.noise_step,
            )

        prefill_outputs = model(
            **inputs,
            output_attentions=False,
            **forward_kwargs,
        )

        cd_prefill_outputs = None

        if cfg.decoding_mode == "vcd":
            cd_prefill_outputs = model(
                **cd_inputs,
                output_attentions=False,
                **forward_kwargs,
            )

        past_key_values = prefill_outputs.past_key_values
        past_key_values_cd = (
            cd_prefill_outputs.past_key_values
            if cd_prefill_outputs is not None
            else None
        )

        outputs_for_next = prefill_outputs
        cd_outputs_for_next = cd_prefill_outputs

        eos_token_ids = get_eos_token_ids(model, tokenizer)

        generated_token_ids: List[int] = []
        generated_token_texts: List[str] = []
        generated_token_probs: List[float] = []
        steps: List[StepRecord] = []

        sequence_ids = input_ids.clone()
        image_token_indices = None

        forced_token_ids = (
            list(cfg.forced_token_ids)
            if cfg.forced_token_ids is not None
            else []
        )

        for step_idx in range(cfg.max_new_tokens):
            forced_token_id = None

            if step_idx < len(forced_token_ids):
                forced_token_id = int(forced_token_ids[step_idx])

            (
                next_token,
                _selection_logits,
                token_prob,
                max_prob,
                entropy,
                dola_premature_layer,
                clean_token_prob,
                cd_token_prob,
            ) = self._select_next_token(
                model=model,
                outputs=outputs_for_next,
                cd_outputs=cd_outputs_for_next,
                generated_token_ids=generated_token_ids,
                forced_token_id=forced_token_id,
            )

            token_id = int(next_token[0, 0].item())
            token_text = decode_token(tokenizer, token_id)

            generated_token_ids.append(token_id)
            generated_token_texts.append(token_text)
            generated_token_probs.append(token_prob)

            sequence_ids = torch.cat([sequence_ids, next_token], dim=-1)

            is_eos = token_id in eos_token_ids

            if is_eos and cfg.stop_on_eos and step_idx + 1 >= cfg.min_new_tokens:
                steps.append(
                    StepRecord(
                        step=step_idx,
                        token_id=token_id,
                        token_text=token_text,
                        token_prob=token_prob,
                        max_prob=max_prob,
                        entropy=entropy,
                        decoding_mode=cfg.decoding_mode,
                        image_attn_by_layer=None,
                        dola_premature_layer=dola_premature_layer,
                        clean_token_prob=clean_token_prob,
                        cd_token_prob=cd_token_prob,
                    )
                )
                break

            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (1, 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=-1,
            )

            step_outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_attentions=cfg.output_attentions,
                **forward_kwargs,
            )

            step_cd_outputs = None

            if cfg.decoding_mode == "vcd":
                step_cd_outputs = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values_cd,
                    output_attentions=False,
                    **forward_kwargs,
                )

            past_key_values = step_outputs.past_key_values

            if step_cd_outputs is not None:
                past_key_values_cd = step_cd_outputs.past_key_values

            image_attn_by_layer = None

            if cfg.output_attentions and step_outputs.attentions is not None:
                image_attn_by_layer, image_token_indices = extract_image_attn_by_layer(
                    attentions=step_outputs.attentions,
                    input_ids=input_ids,
                    current_step=step_idx,
                    model=model,
                    tokenizer=tokenizer,
                    image_token_indices=image_token_indices,
                    selected_layers=cfg.selected_layers,
                    keep_attn_on_cpu=cfg.keep_attn_on_cpu,
                )

            steps.append(
                StepRecord(
                    step=step_idx,
                    token_id=token_id,
                    token_text=token_text,
                    token_prob=token_prob,
                    max_prob=max_prob,
                    entropy=entropy,
                    decoding_mode=cfg.decoding_mode,
                    image_attn_by_layer=image_attn_by_layer,
                    dola_premature_layer=dola_premature_layer,
                    clean_token_prob=clean_token_prob,
                    cd_token_prob=cd_token_prob,
                )
            )

            outputs_for_next = step_outputs
            cd_outputs_for_next = step_cd_outputs

            if (
                cfg.stop_on_sentence_end
                and step_idx + 1 >= cfg.min_new_tokens
                and any(end_text in token_text for end_text in cfg.sentence_end_texts)
            ):
                break

        caption = _decode_caption(
            wrapper=wrapper,
            tokenizer=tokenizer,
            token_ids=generated_token_ids,
        )

        return StepwiseOutput(
            caption=caption,
            token_ids=generated_token_ids,
            token_texts=generated_token_texts,
            token_probs=generated_token_probs,
            steps=steps,
            sequences=sequence_ids,
            input_ids=input_ids,
            raw_outputs=outputs_for_next,
        )

    def generate_batch(
        self,
        wrapper: BaseLVLM,
        image_paths: Optional[Sequence[PathLike]] = None,
        images: Optional[Sequence[Any]] = None,
        prompts: Union[str, Sequence[str]] = "Describe this image.",
        use_chat_template: Optional[bool] = None,
        **prepare_kwargs: Any,
    ) -> StepwiseOutput:
        inputs = wrapper.prepare_batch(
            image_paths=image_paths,
            images=images,
            prompts=prompts,
            use_chat_template=use_chat_template,
            **prepare_kwargs,
        )

        return self.generate_from_inputs(wrapper, inputs)


