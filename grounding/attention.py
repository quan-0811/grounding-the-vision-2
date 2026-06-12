# grounding/attention.py

"""
Attention extraction and layer-head selection utilities.

This file converts raw generated-token attentions into PHG image-attention maps.

Output contract:
    image_attn_by_layer[layer_id] = Tensor[num_heads, num_image_tokens]

This is important because grounding/masks.py selects individual layer-head maps.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from grounding.ads import spatial_entropy
from grounding.grid import image_attn_to_grid


# ============================================================
# Image-token resolution
# ============================================================

def get_image_token_id(model, tokenizer=None, processor=None) -> int:
    """
    Resolve image-token id for different LVLM families.

    LLaVA:
        model.config.image_token_index

    Qwen2-VL:
        model.config.image_token_id
        tokenizer token: <|image_pad|>

    Some processors expose:
        processor.image_token
    """

    config = getattr(model, "config", None)

    for attr in [
        "image_token_index",
        "image_token_id",
    ]:
        value = getattr(config, attr, None)

        if value is not None:
            return int(value)

    text_config = getattr(config, "text_config", None)

    for attr in [
        "image_token_index",
        "image_token_id",
    ]:
        value = getattr(text_config, attr, None)

        if value is not None:
            return int(value)

    if tokenizer is not None:
        candidate_tokens = [
            "<|image_pad|>",  # Qwen2-VL
            "<image>",
            "<image_token>",
            "<im_patch>",
        ]

        for token in candidate_tokens:
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
            except Exception:
                token_id = None

            if token_id is None:
                continue

            unk_token_id = getattr(tokenizer, "unk_token_id", None)

            if unk_token_id is not None and int(token_id) == int(unk_token_id):
                continue

            return int(token_id)

    if processor is not None and tokenizer is not None:
        image_token = getattr(processor, "image_token", None)

        if image_token is not None:
            try:
                token_id = tokenizer.convert_tokens_to_ids(image_token)
            except Exception:
                token_id = None

            if token_id is not None:
                return int(token_id)

    raise ValueError("Could not resolve image token id.")


def resolve_image_token_indices(
    input_ids,
    model=None,
    tokenizer=None,
    processor=None,
    image_token_indices=None,
) -> torch.Tensor:
    """
    Resolve positions of image placeholder tokens in input_ids.

    Returns:
        1D LongTensor of image-token positions for batch item 0.
    """

    if image_token_indices is not None:
        if torch.is_tensor(image_token_indices):
            return image_token_indices.long()

        return torch.tensor(
            image_token_indices,
            dtype=torch.long,
            device=input_ids.device,
        )

    if input_ids is None:
        raise ValueError("input_ids is required to resolve image token indices.")

    if input_ids.dim() == 2:
        ids = input_ids[0]
    else:
        ids = input_ids

    image_token_id = get_image_token_id(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
    )

    indices = torch.where(ids == int(image_token_id))[0]

    if indices.numel() == 0:
        raise ValueError(
            f"No image token positions found for image_token_id={image_token_id}."
        )

    return indices.long()


# ============================================================
# Layer / attention helpers
# ============================================================

def _resolve_layer_ids(
    num_layers: int,
    selected_layers: Optional[Sequence[int]] = None,
) -> List[int]:
    """
    Resolve layer ids, supporting negative indices.
    """

    if selected_layers is None:
        return list(range(num_layers))

    layer_ids: List[int] = []

    for layer_id in selected_layers:
        real_layer_id = int(layer_id)

        if real_layer_id < 0:
            real_layer_id = num_layers + real_layer_id

        if real_layer_id < 0 or real_layer_id >= num_layers:
            continue

        layer_ids.append(real_layer_id)

    return layer_ids


def _unwrap_layer_attention(layer_attn):
    """
    Some models return tuple/list objects per layer.
    Extract the actual tensor.
    """

    if isinstance(layer_attn, (tuple, list)):
        if len(layer_attn) == 0:
            return None

        layer_attn = layer_attn[0]

    if layer_attn is None:
        return None

    if not torch.is_tensor(layer_attn):
        return None

    return layer_attn


def _extract_last_query_attention(layer_attn: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Convert one layer's raw attention into:
        Tensor[num_heads, key_len]

    Supported shapes:
        [batch, heads, query_len, key_len]
        [heads, query_len, key_len]
        [batch, query_len, key_len]
        [query_len, key_len]
    """

    if layer_attn.dim() == 4:
        # [B, H, Q, K] -> [H, K]
        return layer_attn[0, :, -1, :].float()

    if layer_attn.dim() == 3:
        # Ambiguous:
        #   [H, Q, K]
        #   [B, Q, K]
        #
        # If first dim is 1, treat it as batch.
        if layer_attn.shape[0] == 1:
            return layer_attn[0, -1, :].float().unsqueeze(0)

        # Otherwise treat first dim as heads.
        return layer_attn[:, -1, :].float()

    if layer_attn.dim() == 2:
        # [Q, K] -> [1, K]
        return layer_attn[-1, :].float().unsqueeze(0)

    return None


# ============================================================
# Main extraction
# ============================================================

def extract_image_attn_by_layer(
    attentions,
    input_ids,
    current_step,
    model=None,
    tokenizer=None,
    processor=None,
    image_token_indices=None,
    selected_layers=None,
    keep_attn_on_cpu=True,
):
    """
    Extract attention from the currently generated token to image-token positions.

    Returns:
        image_attn_by_layer:
            dict[layer_idx -> Tensor[num_heads, num_image_tokens]]

        image_token_indices:
            resolved image-token positions
    """

    _ = current_step

    if attentions is None:
        return None, image_token_indices

    if len(attentions) == 0:
        return None, image_token_indices

    if selected_layers is None:
        selected_layers = [-1]

    if image_token_indices is None:
        image_token_indices = resolve_image_token_indices(
            input_ids=input_ids,
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            image_token_indices=image_token_indices,
        )

    if not torch.is_tensor(image_token_indices):
        image_token_indices = torch.tensor(
            image_token_indices,
            dtype=torch.long,
            device=input_ids.device,
        )

    image_token_indices = image_token_indices.long()

    image_attn_by_layer: Dict[int, torch.Tensor] = {}

    num_layers = len(attentions)
    layer_ids = _resolve_layer_ids(
        num_layers=num_layers,
        selected_layers=selected_layers,
    )

    expected_num_image_tokens = int(image_token_indices.numel())

    for real_layer_idx in layer_ids:
        layer_attn = _unwrap_layer_attention(attentions[real_layer_idx])

        if layer_attn is None:
            continue

        token_attn = _extract_last_query_attention(layer_attn)

        if token_attn is None:
            continue

        # token_attn: [num_heads, key_len]
        key_len = int(token_attn.shape[-1])

        valid_image_indices = image_token_indices.to(token_attn.device)
        valid_image_indices = valid_image_indices[
            valid_image_indices < key_len
        ]

        if valid_image_indices.numel() == 0:
            continue

        # If the original prompt has many image tokens but only <=1 survives
        # against the attention key axis, this is almost always cached-step
        # attention, not full-prefix attention. Do not return fake grounding.
        if expected_num_image_tokens > 1 and int(valid_image_indices.numel()) <= 1:
            continue

        image_attn = token_attn.index_select(
            dim=-1,
            index=valid_image_indices.long(),
        )

        # image_attn: [num_heads, num_valid_image_tokens]
        image_attn = image_attn.float()

        if keep_attn_on_cpu:
            image_attn = image_attn.detach().cpu()
        else:
            image_attn = image_attn.detach()

        image_attn_by_layer[int(real_layer_idx)] = image_attn

    if len(image_attn_by_layer) == 0:
        return None, image_token_indices

    return image_attn_by_layer, image_token_indices


# ============================================================
# Layer-head selection
# ============================================================

def _ensure_head_token_attention(image_attn: torch.Tensor) -> torch.Tensor:
    """
    Normalize attention to shape [num_heads, num_image_tokens].

    Accepted:
        [num_image_tokens]
        [num_heads, num_image_tokens]
    """

    image_attn = image_attn.detach().float().cpu()

    if image_attn.dim() == 1:
        return image_attn.unsqueeze(0)

    if image_attn.dim() == 2:
        return image_attn

    raise ValueError(
        f"Expected image attention dim 1 or 2, got shape={tuple(image_attn.shape)}"
    )


def get_kept_lh_from_step(
    step: Dict[str, Any],
    image_grid_shape: Optional[Tuple[int, int]] = None,
    inputs: Optional[Dict[str, Any]] = None,
    model=None,
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

    image_attn_by_layer = step.get("image_attn_by_layer")

    if image_attn_by_layer is None:
        return []

    results: List[Dict[str, Any]] = []

    for layer_id, image_attn in image_attn_by_layer.items():
        image_attn = _ensure_head_token_attention(image_attn)
        num_heads = int(image_attn.shape[0])

        for head_id in range(num_heads):
            attn_1d = image_attn[head_id]
            attn_sum = float(attn_1d.sum().item())

            spatial_entropy_value = float("inf")
            bottom_row_focus = False
            num_components = 0

            if int(attn_1d.numel()) > 1 and attn_sum >= attn_sum_threshold:
                try:
                    attn_2d = image_attn_to_grid(
                        attn_1d,
                        image_grid_shape=image_grid_shape,
                        inputs=inputs,
                        model=model,
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

                except Exception:
                    spatial_entropy_value = float("inf")
                    bottom_row_focus = False
                    num_components = 0

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