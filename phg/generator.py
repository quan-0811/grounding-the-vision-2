"""
Main PHG generator.

This file should stay as orchestration only.

Algorithm:
    1. Decode one sentence/segment.
    2. If all tokens are confident, score segment and continue.
    3. If uncertainty appears, checkpoint before first uncertain token.
    4. Score prefix before checkpoint.
    5. Decode candidate continuations.
    6. Rerank candidates by hallucination score.
    7. Commit object memory at sentence boundary.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from models.base import BaseLVLM, PathLike, TensorDict

from decoding.stepwise import (
    StepwiseConfig,
    StepwiseDecoder,
    _decode_token,
    _get_eos_token_ids,
    _get_model,
    _get_processor,
    _get_tokenizer,
    _make_noised_inputs,
    _move_inputs_to_model,
)

from phg.candidates import (
    candidates_to_trace,
    first_boundary_candidate,
    select_candidate_tokens,
)
from phg.checkpoint import (
    append_generated_ids_to_inputs,
    build_checkpoint,
    extract_output_segment_by_abs_range,
    normalize_prefix_ids,
)
from phg.memory import PHGMemory
from phg.scoring import score_segment
from phg.types import (
    CandidateRecord,
    CandidateBranch,
    CheckpointState,
    PHGConfig,
    PHGOutput,
)


def _is_sentence_boundary_stop(output: Dict[str, Any]) -> bool:
    return output.get("stop_reason") in {
        "sentence_end_generated",
        "sentence_end_or_eos_in_candidates",
        "eos_generated",
    }


def _is_eos_stop(output: Dict[str, Any]) -> bool:
    return output.get("stop_reason") == "eos_generated"


def _serialize_step_for_trace(step: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "step": step.get("step"),
        "token_id": step.get("token_id"),
        "token_text": step.get("token_text"),
        "max_prob": step.get("max_prob"),
        "selected_token_prob": step.get("selected_token_prob"),
        "is_low_confidence": step.get("is_low_confidence"),
        "decoding_mode": step.get("decoding_mode"),
        "dola_premature_layer": step.get("dola_premature_layer"),
        "clean_token_prob": step.get("clean_token_prob"),
        "cd_token_prob": step.get("cd_token_prob"),
        "has_image_attn": step.get("image_attn_by_layer") is not None,
    }


class PHGGenerator:
    """
    PHG generator.

    Example:
        generator = PHGGenerator(PHGConfig(decoding_mode="greedy"))
        output = generator.generate_from_inputs(wrapper, inputs)
    """

    def __init__(self, config: Optional[PHGConfig] = None) -> None:
        self.config = config or PHGConfig()

    def _build_stepwise_config(
        self,
        forced_token_ids: Optional[Sequence[int]] = None,
        enable_sentence_stop: Optional[bool] = None,
    ) -> StepwiseConfig:
        cfg = self.config

        return StepwiseConfig(
            decoding_mode=cfg.decoding_mode,
            max_new_tokens=cfg.max_new_tokens,
            min_new_tokens=cfg.min_new_tokens,
            use_cache=True,

            output_attentions=True,
            selected_layers=cfg.selected_layers,
            keep_attn_on_cpu=cfg.keep_attn_on_cpu,
            image_grid_shape=cfg.image_grid_shape,

            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=None,
            repetition_penalty=cfg.repetition_penalty,

            stop_on_eos=True,
            stop_on_sentence_end=(
                cfg.stop_at_sentence_end
                if enable_sentence_stop is None
                else bool(enable_sentence_stop)
            ),

            forced_token_ids=forced_token_ids,

            dola_layers=cfg.dola_layers,
            dola_relative_top=cfg.dola_relative_top,
            dola_select_strategy=cfg.dola_select_strategy,

            cd_alpha=cfg.cd_alpha,
            cd_beta=cfg.cd_beta,
            noise_step=cfg.noise_step,
            image_tensor_key=cfg.image_tensor_key,
        )

    @torch.inference_mode()
    def _decode_segment_with_checkpoint(
        self,
        wrapper: BaseLVLM,
        base_inputs: TensorDict,
        prefix_ids: Sequence[int],
        force_first_token_id: Optional[int] = None,
        enable_uncertainty_check: bool = True,
    ) -> Dict[str, Any]:
        """
        Decode one sentence/segment.

        This records:
            - token steps
            - image attention
            - first low-confidence checkpoint
            - candidates at checkpoint
        """

        cfg = self.config

        model = _get_model(wrapper)
        processor = _get_processor(wrapper)
        tokenizer = _get_tokenizer(wrapper, processor)

        model.eval()

        stepwise_config = self._build_stepwise_config(
            forced_token_ids=(
                [int(force_first_token_id)]
                if force_first_token_id is not None
                else None
            ),
            enable_sentence_stop=cfg.stop_at_sentence_end,
        )

        step_decoder = StepwiseDecoder(stepwise_config)
        forward_kwargs = step_decoder.build_forward_kwargs()

        working_inputs = append_generated_ids_to_inputs(
            base_inputs,
            prefix_ids,
        )

        if working_inputs["input_ids"].shape[0] != 1:
            raise ValueError("PHG currently expects batch size 1.")

        input_ids = working_inputs["input_ids"]

        if "attention_mask" in working_inputs and working_inputs["attention_mask"] is not None:
            attention_mask = working_inputs["attention_mask"]
        else:
            attention_mask = torch.ones_like(input_ids)
            working_inputs["attention_mask"] = attention_mask

        cd_inputs = None

        if cfg.decoding_mode == "vcd":
            cd_inputs = _make_noised_inputs(
                working_inputs,
                image_tensor_key=cfg.image_tensor_key,
                noise_step=cfg.noise_step,
            )

        prefill_outputs = model(
            **working_inputs,
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

        eos_token_ids = _get_eos_token_ids(model, tokenizer)

        prefix_ids = normalize_prefix_ids(prefix_ids)

        generated_ids: List[int] = []
        steps: List[Dict[str, Any]] = []

        checkpoint: Optional[CheckpointState] = None
        candidates: Optional[List[CandidateRecord]] = None
        candidate_records: List[Dict[str, Any]] = []

        has_uncertainty = False
        first_uncertain_step = None
        uncertain_steps: List[int] = []

        stop_reason = None

        image_token_indices = None

        for step_idx in range(cfg.max_new_tokens):
            forced_first_step = (
                force_first_token_id is not None
                and step_idx == 0
            )

            (
                next_token,
                selection_logits,
                selected_token_prob,
                max_prob,
                entropy,
                dola_premature_layer,
                clean_token_prob,
                cd_token_prob,
            ) = step_decoder._select_next_token(
                model=model,
                outputs=outputs_for_next,
                cd_outputs=cd_outputs_for_next,
                generated_token_ids=prefix_ids + generated_ids,
                forced_token_id=(
                    int(force_first_token_id)
                    if forced_first_step
                    else None
                ),
            )

            is_low_confidence = bool(
                enable_uncertainty_check
                and not forced_first_step
                and max_prob < cfg.accumulate_prob
            )

            if is_low_confidence:
                has_uncertainty = True
                uncertain_steps.append(step_idx)

                if first_uncertain_step is None:
                    first_uncertain_step = step_idx

            forced_boundary_token = None

            should_store_uncertainty = (
                is_low_confidence
                and (
                    not cfg.checkpoint_once
                    or checkpoint is None
                )
            )

            if should_store_uncertainty:
                current_candidates, candidate_max_prob = select_candidate_tokens(
                    logits=selection_logits,
                    tokenizer=tokenizer,
                    top_k=cfg.top_k,
                    accumulate_prob=cfg.accumulate_prob,
                )

                candidate_records.append(
                    {
                        "step": int(step_idx),
                        "max_prob": float(candidate_max_prob),
                        "threshold": float(cfg.accumulate_prob),
                        "candidates": candidates_to_trace(current_candidates),
                        "dola_premature_layer": dola_premature_layer,
                    }
                )

                if candidates is None:
                    candidates = current_candidates

                if checkpoint is None:
                    checkpoint = build_checkpoint(
                        base_inputs=base_inputs,
                        prefix_ids=prefix_ids,
                        generated_ids=generated_ids,
                        tokenizer=tokenizer,
                    )

                if cfg.stop_if_sentence_end_in_candidates:
                    boundary_candidate = first_boundary_candidate(
                        current_candidates,
                        tokenizer=tokenizer,
                        eos_token_ids=eos_token_ids,
                    )

                    if boundary_candidate is not None:
                        forced_boundary_token = int(boundary_candidate.token_id)
                        stop_reason = "sentence_end_or_eos_in_candidates"

                if cfg.debug:
                    print(
                        f"[PHG uncertainty] step={step_idx}, "
                        f"max_prob={candidate_max_prob:.4f}, "
                        f"candidates={candidates_to_trace(current_candidates)}"
                    )

            if forced_boundary_token is not None:
                next_token = torch.tensor(
                    [[int(forced_boundary_token)]],
                    dtype=torch.long,
                    device=input_ids.device,
                )

            token_id = int(next_token[0, 0].item())
            token_text = _decode_token(tokenizer, token_id)

            generated_ids.append(token_id)

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
                output_attentions=True,
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

            if step_outputs.attentions is not None:
                from grounding.attention import extract_image_attn_by_layer

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

            step_dict = {
                "step": int(step_idx),
                "token_id": int(token_id),
                "token_text": token_text,
                "max_prob": float(max_prob),
                "selected_token_prob": float(selected_token_prob),
                "entropy": float(entropy),
                "is_low_confidence": bool(is_low_confidence),
                "decoding_mode": cfg.decoding_mode,
                "dola_premature_layer": dola_premature_layer,
                "clean_token_prob": clean_token_prob,
                "cd_token_prob": cd_token_prob,
                "image_attn_by_layer": image_attn_by_layer,
            }

            steps.append(step_dict)

            outputs_for_next = step_outputs
            cd_outputs_for_next = step_cd_outputs

            if stop_reason is not None:
                break

            if token_id in eos_token_ids:
                stop_reason = "eos_generated"
                break

            decoded_text_raw = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            if (
                cfg.stop_at_sentence_end
                and len(generated_ids) >= cfg.min_new_tokens
                and re.search(r"([.!?。！？]\s*|\n+|\r+)$", decoded_text_raw)
            ):
                stop_reason = "sentence_end_generated"
                break

        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        full_generated_ids = prefix_ids + generated_ids

        full_generated_text = tokenizer.decode(
            full_generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        if stop_reason is None:
            stop_reason = "max_new_tokens_reached"

        return {
            "generated_text": generated_text,
            "generated_ids": generated_ids,

            "prefix_ids": prefix_ids,
            "full_generated_ids": full_generated_ids,
            "full_generated_text": full_generated_text,

            "confidence_path": "uncertainty" if has_uncertainty else "certainty",
            "has_uncertainty": has_uncertainty,
            "first_uncertain_step": first_uncertain_step,
            "uncertain_steps": uncertain_steps,

            "stop_reason": stop_reason,
            "steps": steps,

            "checkpoint": checkpoint,
            "candidates": candidates,
            "candidate_records": candidate_records,

            "step_summaries": [
                _serialize_step_for_trace(step)
                for step in steps
            ],
        }

    @torch.inference_mode()
    def generate_from_inputs(
        self,
        wrapper: BaseLVLM,
        inputs: TensorDict,
    ) -> PHGOutput:
        """
        Full PHG generation from prepared model inputs.
        """

        cfg = self.config

        model = _get_model(wrapper)
        processor = _get_processor(wrapper)
        tokenizer = _get_tokenizer(wrapper, processor)

        base_inputs = _move_inputs_to_model(inputs, model)

        accepted_generated_ids = normalize_prefix_ids(cfg.prefix_ids)

        memory = PHGMemory(
            processed_prefix_len=len(accepted_generated_ids)
        )

        round_outputs: List[Dict[str, Any]] = []
        decision_trace: List[Dict[str, Any]] = []

        for round_idx in range(cfg.max_rounds):
            if cfg.debug:
                print(f"\n========== PHG ROUND {round_idx} ==========")
                print(
                    "[prefix]",
                    tokenizer.decode(
                        accepted_generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    ),
                )
                print("[memory]", memory.to_trace())

            current_output = self._decode_segment_with_checkpoint(
                wrapper=wrapper,
                base_inputs=base_inputs,
                prefix_ids=accepted_generated_ids,
                force_first_token_id=None,
                enable_uncertainty_check=True,
            )

            round_outputs.append(current_output)

            if cfg.debug:
                print("[generated]", repr(current_output["generated_text"]))
                print("[full]", repr(current_output["full_generated_text"]))
                print("[path]", current_output["confidence_path"])
                print("[stop_reason]", current_output["stop_reason"])

            # ======================================================
            # Certainty path.
            # ======================================================
            if current_output["confidence_path"] == "certainty":
                segment_score = score_segment(
                    tokenizer=tokenizer,
                    segment_token_ids=current_output["generated_ids"],
                    segment_steps=current_output["steps"],
                    memory=memory,
                    config=cfg,
                    inputs=base_inputs,
                )

                memory.add_score_to_sentence(segment_score)

                accepted_generated_ids = current_output["full_generated_ids"]
                memory.set_processed_prefix_len(len(accepted_generated_ids))

                decision_trace.append(
                    {
                        "round": int(round_idx),
                        "path": "certainty",
                        "stop_reason": current_output["stop_reason"],
                        "segment_score": segment_score.to_trace(),
                        "memory": memory.to_trace(),
                    }
                )

                if _is_sentence_boundary_stop(current_output):
                    memory.commit_sentence()

                    if _is_eos_stop(current_output):
                        break

                continue

            # ======================================================
            # Uncertainty path.
            # ======================================================
            checkpoint = current_output.get("checkpoint")

            if checkpoint is None:
                checkpoint_pos = len(accepted_generated_ids)
                checkpoint_generated_ids = list(accepted_generated_ids)
            else:
                checkpoint_pos = int(checkpoint.position)
                checkpoint_generated_ids = list(checkpoint.generated_ids)

            # Process certain prefix before checkpoint:
            # [memory.processed_prefix_len, checkpoint_pos)
            prefix_segment_ids, prefix_segment_steps = extract_output_segment_by_abs_range(
                output=current_output,
                start_abs=memory.processed_prefix_len,
                end_abs=checkpoint_pos,
            )

            prefix_score = score_segment(
                tokenizer=tokenizer,
                segment_token_ids=prefix_segment_ids,
                segment_steps=prefix_segment_steps,
                memory=memory,
                config=cfg,
                inputs=base_inputs,
            )

            memory.add_score_to_sentence(prefix_score)
            memory.set_processed_prefix_len(checkpoint_pos)

            if cfg.debug:
                print("[checkpoint_pos]", checkpoint_pos)
                print("[prefix_score]", prefix_score.to_trace())

            # If boundary among candidates was selected, accept shortened output.
            if current_output["stop_reason"] == "sentence_end_or_eos_in_candidates":
                accepted_generated_ids = current_output["full_generated_ids"]
                memory.set_processed_prefix_len(len(accepted_generated_ids))
                memory.commit_sentence()

                decision_trace.append(
                    {
                        "round": int(round_idx),
                        "path": "uncertainty_boundary",
                        "stop_reason": current_output["stop_reason"],
                        "prefix_score": prefix_score.to_trace(),
                        "memory": memory.to_trace(),
                    }
                )

                continue

            candidates = current_output.get("candidates")

            if not candidates:
                accepted_generated_ids = current_output["full_generated_ids"]
                memory.set_processed_prefix_len(len(accepted_generated_ids))

                decision_trace.append(
                    {
                        "round": int(round_idx),
                        "path": "uncertainty_no_candidates_fallback",
                        "stop_reason": current_output["stop_reason"],
                        "prefix_score": prefix_score.to_trace(),
                        "memory": memory.to_trace(),
                    }
                )

                if _is_sentence_boundary_stop(current_output):
                    memory.commit_sentence()

                    if _is_eos_stop(current_output):
                        break

                continue

            branches: List[CandidateBranch] = []

            for cand in candidates:
                if cfg.debug:
                    print(
                        f"[candidate {cand.rank}] "
                        f"{cand.token_id} {repr(cand.token_text)}"
                    )

                branch_output = self._decode_segment_with_checkpoint(
                    wrapper=wrapper,
                    base_inputs=base_inputs,
                    prefix_ids=checkpoint_generated_ids,
                    force_first_token_id=cand.token_id,
                    enable_uncertainty_check=True,
                )

                branch_score = score_segment(
                    tokenizer=tokenizer,
                    segment_token_ids=branch_output["generated_ids"],
                    segment_steps=branch_output["steps"],
                    memory=memory,
                    config=cfg,
                    inputs=base_inputs,
                )

                branches.append(
                    CandidateBranch(
                        candidate=cand,
                        output=branch_output,
                        score=branch_score,
                    )
                )

                if cfg.debug:
                    print("[branch]", branch_score.to_trace())

            selected_branch = sorted(
                branches,
                key=lambda b: b.ranking_tuple,
            )[0]

            selected_output = selected_branch.output
            selected_score = selected_branch.score

            accepted_generated_ids = selected_output["full_generated_ids"]
            memory.set_processed_prefix_len(len(accepted_generated_ids))
            memory.add_score_to_sentence(selected_score)

            decision_trace.append(
                {
                    "round": int(round_idx),
                    "path": "uncertainty_candidate_selection",
                    "stop_reason": selected_output["stop_reason"],
                    "prefix_score": prefix_score.to_trace(),
                    "branches": [branch.to_trace() for branch in branches],
                    "selected_candidate": selected_branch.candidate.to_dict(),
                    "selected_score": selected_score.to_trace(),
                    "memory": memory.to_trace(),
                }
            )

            if _is_sentence_boundary_stop(selected_output):
                memory.commit_sentence()

                if _is_eos_stop(selected_output):
                    break

            continue

        final_text = tokenizer.decode(
            accepted_generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        return PHGOutput(
            final_text=final_text,
            final_generated_ids=accepted_generated_ids,

            objects=memory.global_objects,
            masks=memory.global_masks,

            global_objects=memory.global_objects,
            global_masks=memory.global_masks,

            sentence_objects=memory.sentence_objects,
            sentence_masks=memory.sentence_masks,

            processed_prefix_len=memory.processed_prefix_len,

            round_outputs=round_outputs,
            decision_trace=decision_trace,
        )

    def generate_batch(
        self,
        wrapper: BaseLVLM,
        image_paths: Optional[Sequence[PathLike]] = None,
        images: Optional[Sequence[Any]] = None,
        prompts: Union[str, Sequence[str]] = "Describe this image.",
        use_chat_template: Optional[bool] = None,
        **prepare_kwargs: Any,
    ) -> PHGOutput:
        """
        Prepare one sample and run PHG.

        PHG is intentionally batch-size-1 because candidate branches and memory
        are per sample.
        """

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
        include_trace: bool = False,
        **prepare_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Generate PHG captions from normalized dataset samples.

        Default output:
            [{"id": ..., "caption": "..."}]
        """

        rows: List[Dict[str, Any]] = []

        for sample in samples:
            output = self.generate_batch(
                wrapper=wrapper,
                image_paths=[sample[image_key]],
                prompts=[sample.get(prompt_key, "Describe this image.")],
                use_chat_template=use_chat_template,
                **prepare_kwargs,
            )

            row: Dict[str, Any] = {
                id_key: int(sample[id_key]),
                caption_key: output.final_text,
            }

            if include_prompt:
                row[prompt_key] = sample.get(prompt_key, "Describe this image.")

            if include_trace:
                row["objects"] = output.objects
                row["decision_trace"] = output.decision_trace
                row["final_generated_ids"] = output.final_generated_ids

            rows.append(row)

        return rows


def generate_phg_from_inputs(
    wrapper: BaseLVLM,
    inputs: TensorDict,
    config: Optional[PHGConfig] = None,
) -> PHGOutput:
    generator = PHGGenerator(config)
    return generator.generate_from_inputs(
        wrapper=wrapper,
        inputs=inputs,
    )


def generate_phg_batch(
    wrapper: BaseLVLM,
    image_paths: Optional[Sequence[PathLike]] = None,
    images: Optional[Sequence[Any]] = None,
    prompts: Union[str, Sequence[str]] = "Describe this image.",
    config: Optional[PHGConfig] = None,
    use_chat_template: Optional[bool] = None,
    **prepare_kwargs: Any,
) -> PHGOutput:
    generator = PHGGenerator(config)
    return generator.generate_batch(
        wrapper=wrapper,
        image_paths=image_paths,
        images=images,
        prompts=prompts,
        use_chat_template=use_chat_template,
        **prepare_kwargs,
    )


def generate_phg_samples(
    wrapper: BaseLVLM,
    samples: Sequence[Dict[str, Any]],
    config: Optional[PHGConfig] = None,
    image_key: str = "image_path",
    prompt_key: str = "prompt",
    id_key: str = "id",
    caption_key: str = "caption",
    use_chat_template: Optional[bool] = False,
    include_prompt: bool = False,
    include_trace: bool = False,
    **prepare_kwargs: Any,
) -> List[Dict[str, Any]]:
    generator = PHGGenerator(config)
    return generator.generate_samples(
        wrapper=wrapper,
        samples=samples,
        image_key=image_key,
        prompt_key=prompt_key,
        id_key=id_key,
        caption_key=caption_key,
        use_chat_template=use_chat_template,
        include_prompt=include_prompt,
        include_trace=include_trace,
        **prepare_kwargs,
    )