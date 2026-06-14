"""
Attention-mask construction utilities.

These functions convert selected image-attention maps into binary object masks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

try:
    from scipy.ndimage import binary_closing, label
except ImportError as exc:
    raise ImportError(
        "grounding.masks requires scipy. Install with: pip install scipy"
    ) from exc

from grounding.grid import image_attn_to_grid
from grounding.attention import get_kept_lh_from_step, _ensure_head_token_attention


def remove_singletons(mask_bool: np.ndarray) -> np.ndarray:
    """
    Remove connected components with area < 2.
    """

    structure = np.ones((3, 3), dtype=bool)

    labeled, _ = label(
        mask_bool.astype(bool),
        structure=structure,
    )

    counts = np.bincount(labeled.ravel())

    keep = np.zeros_like(counts, dtype=bool)

    if len(keep) > 1:
        keep[1:] = counts[1:] >= 2

    return keep[labeled]


def _get_layer_attention(
    image_attn_by_layer: Dict[Any, torch.Tensor],
    layer_id: Any,
) -> Optional[torch.Tensor]:
    """
    Resolve layer attention robustly.

    Some traces store layer ids as int, some as string.
    """

    if layer_id in image_attn_by_layer:
        return image_attn_by_layer[layer_id]

    try:
        int_layer_id = int(layer_id)

        if int_layer_id in image_attn_by_layer:
            return image_attn_by_layer[int_layer_id]
    except Exception:
        pass

    str_layer_id = str(layer_id)

    if str_layer_id in image_attn_by_layer:
        return image_attn_by_layer[str_layer_id]

    return None

def get_object_mask_from_step(
    step: Dict[str, Any],
    image_grid_shape: Optional[Tuple[int, int]] = None,
    inputs: Optional[Dict[str, Any]] = None,
    model=None,
    top_n_heads: int = 5,
    attn_sum_threshold: float = 0.49,
) -> Optional[np.ndarray]:
    """
    Build a binary object mask from selected attention heads.

    Expected step format:
        step["image_attn_by_layer"][layer_id] = Tensor[num_heads, num_image_tokens]

    Also tolerates:
        Tensor[num_image_tokens]

    Returns:
        np.ndarray[H, W] with uint8 values, or None.
    """

    image_attn_by_layer = step.get("image_attn_by_layer")

    if image_attn_by_layer is None:
        return None

    kept = get_kept_lh_from_step(
        step=step,
        image_grid_shape=image_grid_shape,
        inputs=inputs,
        model=model,
        attn_sum_threshold=attn_sum_threshold,
    )

    if len(kept) == 0:
        return None

    binary_masks = []

    for selected in kept[:top_n_heads]:
        layer_id = selected["layer"]
        head_id = int(selected.get("head", 0))

        image_attn = _get_layer_attention(
            image_attn_by_layer=image_attn_by_layer,
            layer_id=layer_id,
        )

        if image_attn is None:
            continue

        image_attn = _ensure_head_token_attention(image_attn)

        if head_id < 0 or head_id >= image_attn.shape[0]:
            continue

        attn_1d = image_attn[head_id]

        # Reject fake cached-step attention.
        # For Qwen, attention length 1 means we did not get full-prefix image attention.
        if int(attn_1d.numel()) <= 1:
            continue

        attn_2d = image_attn_to_grid(
            attn_1d,
            image_grid_shape=image_grid_shape,
            inputs=inputs,
            model=model,
        ).numpy()

        attn_min = float(attn_2d.min())
        attn_max = float(attn_2d.max())

        attn_2d = (attn_2d - attn_min) / (attn_max - attn_min + 1e-8)

        mean_val = np.mean(attn_2d)
        activated = np.maximum(attn_2d - mean_val * 2.0, 0)

        binary = (activated > 1e-8).astype(np.int32)
        binary = remove_singletons(binary)

        binary = binary_closing(
            binary,
            structure=np.ones((3, 3)),
        ).astype(np.int32)

        binary_masks.append(binary)

    if len(binary_masks) == 0:
        return None

    mask = np.median(
        np.stack(binary_masks, axis=0),
        axis=0,
    )

    return (mask > 0).astype(np.uint8)