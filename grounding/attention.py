"""
Attention extraction and layer-head selection utilities.

This file converts raw generated-token attentions into PHG image-attention maps.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from grounding.ads import spatial_entropy
from grounding.grid import image_attn_to_grid


def get_image_token_id(model: Any, tokenizer: Any) -> int:
    """
    Get image token id.

    For LLaVA:
        model.config.image_token_index

    Fallback:
        tokenizer.convert_tokens_to_ids("<image>")
    """

    image_token_id = getattr(model.config, "image_token_index", None)

    if image_token_id is None:
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    if image_token_id is None:
        raise ValueError("Could not resolve image token id.")

    return int(image_token_id)


def resolve_image_token_indices(
    input_ids: torch.Tensor,
    token_attn_key_len: int,
    current_step: int,
    model: Any,
    tokenizer: Any,
) -> torch.Tensor:
    """
    Resolve image-token positions in the attention key dimension.

    Handles two cases:

    Case 1:
        input_ids already contains many expanded image tokens.

    Case 2:
        input_ids contains one <image> placeholder and LLaVA internally
        expands it into visual patch tokens.

    Assumption:
        batch size = 1, single image.
    """

    image_token_id = get_image_token_id(model, tokenizer)

    raw_img_positions = (
        input_ids[0] == image_token_id
    ).nonzero(as_tuple=False).flatten()

    if len(raw_img_positions) == 0:
        return torch.empty(0, dtype=torch.long)

    raw_prompt_len = input_ids.shape[1]

    # During decoding:
    # key_len = expanded_prompt_len + generated_tokens_seen
    # At current_step=0, after feeding the first generated token:
    # key_len = expanded_prompt_len + 1
    expanded_prompt_len = int(token_attn_key_len) - (int(current_step) + 1)

    # Case 1: image tokens are already expanded in input_ids.
    if len(raw_img_positions) > 1:
        return raw_img_positions.detach().cpu().long()

    # Case 2: single placeholder expanded internally.
    placeholder_pos = int(raw_img_positions[0].item())

    num_image_tokens = expanded_prompt_len - (raw_prompt_len - 1)

    if num_image_tokens <= 0:
        return torch.empty(0, dtype=torch.long)

    image_start = placeholder_pos
    image_end = placeholder_pos + num_image_tokens

    return torch.arange(image_start, image_end, dtype=torch.long)


def _resolve_layer_ids(
    num_layers: int,
    selected_layers: Optional[Sequence[int]] = None,
) -> List[int]:
    """
    Resolve layer ids, supporting negative indices.
    """

    if selected_layers is None:
        return list(range(num_layers))

    layer_ids = []

    for layer_id in selected_layers:
        if layer_id < 0:
            layer_id = num_layers + layer_id

        if layer_id < 0 or layer_id >= num_layers:
            raise ValueError(
                f"Layer id {layer_id} is out of range for {num_layers} layers."
            )

        layer_ids.append(int(layer_id))

    return layer_ids


def extract_image_attn_by_layer(
    attentions: Sequence[torch.Tensor],
    input_ids: torch.Tensor,
    token_attn_key_len: Optional[int] = None,
    current_step: int = 0,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    image_token_indices: Optional[torch.Tensor] = None,
    selected_layers: Optional[Sequence[int]] = None,
    keep_attn_on_cpu: bool = True,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """
    Extract image attention for the last generated token.

    Args:
        attentions:
            Model output attentions. Each layer usually has shape:
                [batch, heads, query_len, key_len]

        input_ids:
            Original prompt input ids. For LLaVA this contains one <image>
            placeholder.

        token_attn_key_len:
            Optional explicit key length.

        current_step:
            Generated-token step index.

        model/tokenizer:
            Needed if image_token_indices is not supplied.

        image_token_indices:
            Optional cached visual-token positions.

        selected_layers:
            Optional layer subset, e.g. [-8, -4, -1].

        keep_attn_on_cpu:
            Store extracted attention on CPU to save VRAM.

    Returns:
        image_attn_by_layer:
            Dict[layer_id] -> Tensor[num_heads, num_image_tokens]

        image_token_indices:
            Resolved image-token indices.
    """

    if attentions is None or len(attentions) == 0:
        raise ValueError("No attentions were provided.")

    num_layers = len(attentions)
    layer_ids = _resolve_layer_ids(num_layers, selected_layers)

    first_attn = attentions[layer_ids[0]]
    first_token_attn = first_attn[0, :, -1, :]

    if token_attn_key_len is None:
        token_attn_key_len = first_token_attn.shape[-1]

    if image_token_indices is None:
        if model is None or tokenizer is None:
            raise ValueError(
                "model and tokenizer are required when image_token_indices is None."
            )

        image_token_indices = resolve_image_token_indices(
            input_ids=input_ids,
            token_attn_key_len=token_attn_key_len,
            current_step=current_step,
            model=model,
            tokenizer=tokenizer,
        )

    image_attn_by_layer: Dict[int, torch.Tensor] = {}

    for layer_id in layer_ids:
        attn = attentions[layer_id]

        # [heads, key_len]
        token_attn = attn[0, :, -1, :].detach()

        valid_img_idx = image_token_indices[
            image_token_indices < token_attn.shape[-1]
        ]

        if len(valid_img_idx) == 0:
            continue

        if keep_attn_on_cpu:
            token_attn_cpu = token_attn.float().cpu()
            img_idx = valid_img_idx.cpu()
            image_attn = token_attn_cpu[:, img_idx]
        else:
            img_idx = valid_img_idx.to(token_attn.device)
            image_attn = token_attn[:, img_idx]

        image_attn_by_layer[int(layer_id)] = image_attn

    return image_attn_by_layer, image_token_indices.detach().cpu()


def get_kept_lh_from_step(
    step: Dict[str, Any],
    image_grid_shape: Optional[Tuple[int, int]] = None,
    inputs: Optional[Dict[str, Any]] = None,
    attn_sum_threshold: float = 0.49,
    bottom_row_threshold: float = 0.05,
    min_layer: int = 1,
) -> List[Dict[str, Any]]:
    """
    Select useful layer-head maps from a PHG step.

    Filtering:
        - enough attention mass on image tokens
        - finite spatial entropy
        - not overly focused on bottom row
        - layer > min_layer

    Fallback:
        if no layer-head passes the filters, return the highest image-attending
        non-bottom-row head.
    """

    image_attn_by_layer = step["image_attn_by_layer"]

    results = []

    for layer_id, image_attn in image_attn_by_layer.items():
        image_attn = image_attn.detach().float().cpu()
        num_heads = image_attn.shape[0]

        for head_id in range(num_heads):
            attn_1d = image_attn[head_id]
            attn_sum = float(attn_1d.sum().item())

            if attn_sum < attn_sum_threshold:
                spatial_entropy_value = float("inf")
                bottom_row_focus = False
                num_components = 0
            else:
                attn_2d = image_attn_to_grid(
                    attn_1d,
                    image_grid_shape=image_grid_shape,
                    inputs=inputs,
                )

                entropy_result = spatial_entropy(
                    attn_2d,
                    threshold=1e-3,
                )

                bottom_row_focus = bool(
                    attn_2d.shape[0] > 0
                    and (attn_2d[-1, :] > bottom_row_threshold).any()
                )

                spatial_entropy_value = float(entropy_result["spatial_entropy"])
                num_components = int(entropy_result["num_components"])

            results.append(
                {
                    "layer": int(layer_id),
                    "head": int(head_id),
                    "attn_sum": attn_sum,
                    "spatial_entropy": spatial_entropy_value,
                    "bottom_row_focus": bottom_row_focus,
                    "num_components": num_components,
                }
            )

    kept = [
        item
        for item in results
        if np.isfinite(item["spatial_entropy"])
        and item["attn_sum"] >= attn_sum_threshold
        and not item["bottom_row_focus"]
        and item["layer"] > min_layer
    ]

    if len(kept) == 0:
        by_sum = sorted(
            results,
            key=lambda item: item["attn_sum"],
            reverse=True,
        )

        kept = [
            item
            for item in by_sum
            if not item["bottom_row_focus"]
        ][:1]

    kept.sort(key=lambda item: item["spatial_entropy"])

    return kept