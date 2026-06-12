"""
Checkpoint utilities for PHG.

A checkpoint is stored immediately before the first low-confidence token.
Candidate continuations start from this checkpoint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from models.base import TensorDict
from phg.types import CheckpointState


def normalize_prefix_ids(
    prefix_ids: Optional[Union[Sequence[int], torch.Tensor]],
) -> List[int]:
    """
    Convert prefix ids into a Python list[int].
    """

    if prefix_ids is None:
        return []

    if torch.is_tensor(prefix_ids):
        prefix_ids = prefix_ids.detach().cpu()

        if prefix_ids.dim() == 2:
            prefix_ids = prefix_ids[0]

        return [int(x) for x in prefix_ids.tolist()]

    return [int(x) for x in prefix_ids]


def append_generated_ids_to_inputs(
    inputs: TensorDict,
    generated_ids: Optional[Sequence[int]],
) -> TensorDict:
    """
    Append answer-side generated token ids to prepared multimodal inputs.

    Image tensors remain unchanged.
    """

    generated_ids = normalize_prefix_ids(generated_ids)

    out: Dict[str, Any] = {}

    for key, value in inputs.items():
        if torch.is_tensor(value):
            out[key] = value.clone()
        else:
            out[key] = value

    if len(generated_ids) == 0:
        return out

    if "input_ids" not in out:
        raise KeyError("Expected `input_ids` in inputs.")

    device = out["input_ids"].device
    dtype = out["input_ids"].dtype

    gen_tensor = torch.tensor(
        [generated_ids],
        dtype=dtype,
        device=device,
    )

    out["input_ids"] = torch.cat(
        [out["input_ids"], gen_tensor],
        dim=-1,
    )

    if "attention_mask" in out and out["attention_mask"] is not None:
        gen_mask = torch.ones(
            (out["attention_mask"].shape[0], len(generated_ids)),
            dtype=out["attention_mask"].dtype,
            device=out["attention_mask"].device,
        )

        out["attention_mask"] = torch.cat(
            [out["attention_mask"], gen_mask],
            dim=-1,
        )

    return out


def build_checkpoint(
    base_inputs: TensorDict,
    prefix_ids: Sequence[int],
    generated_ids: Sequence[int],
    tokenizer: Any,
) -> CheckpointState:
    """
    Build checkpoint immediately before an uncertain token.

    checkpoint_generated_ids = prefix_ids + generated_ids
    """

    checkpoint_generated_ids = normalize_prefix_ids(prefix_ids) + normalize_prefix_ids(
        generated_ids
    )

    checkpoint_inputs = append_generated_ids_to_inputs(
        base_inputs,
        checkpoint_generated_ids,
    )

    checkpoint_text = tokenizer.decode(
        checkpoint_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    return CheckpointState(
        generated_ids=checkpoint_generated_ids,
        input_ids=checkpoint_inputs["input_ids"].detach().cpu(),
        text=checkpoint_text,
        position=len(checkpoint_generated_ids),
    )


def extract_output_segment_by_abs_range(
    output: Dict[str, Any],
    start_abs: int,
    end_abs: int,
) -> tuple[List[int], List[Dict[str, Any]]]:
    """
    Extract generated token ids and step records from an absolute answer-token
    range.

    output["full_generated_ids"]:
        prefix_ids + generated_ids

    output["steps"]:
        only corresponds to output["generated_ids"].
    """

    prefix_len = len(output.get("prefix_ids", []))

    local_start = max(0, int(start_abs) - prefix_len)
    local_end = max(0, int(end_abs) - prefix_len)

    generated_ids = output.get("generated_ids", [])
    steps = output.get("steps", [])

    return (
        generated_ids[local_start:local_end],
        steps[local_start:local_end],
    )