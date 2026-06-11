"""
Attention-mask construction utilities.

These functions convert selected image-attention maps into binary object masks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from scipy.ndimage import binary_closing, label
except ImportError as exc:
    raise ImportError(
        "grounding.masks requires scipy. Install with: pip install scipy"
    ) from exc

from grounding.grid import image_attn_to_grid
from grounding.attention import get_kept_lh_from_step


def remove_singletons(mask_bool: np.ndarray) -> np.ndarray:
    """
    Remove connected components with area < 2.

    Args:
        mask_bool:
            Binary mask.

    Returns:
        Cleaned binary mask.
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


def get_object_mask_from_step(
    step: Dict[str, Any],
    image_grid_shape: Optional[Tuple[int, int]] = None,
    inputs: Optional[Dict[str, Any]] = None,
    top_n_heads: int = 5,
    attn_sum_threshold: float = 0.49,
) -> Optional[np.ndarray]:
    """
    Build a binary object mask from selected attention heads.

    The step must contain:
        step["image_attn_by_layer"][layer_id] = Tensor[num_heads, num_image_tokens]

    Returns:
        np.ndarray[H, W] with uint8 values, or None.
    """

    kept = get_kept_lh_from_step(
        step=step,
        image_grid_shape=image_grid_shape,
        inputs=inputs,
        attn_sum_threshold=attn_sum_threshold,
    )

    if len(kept) == 0:
        return None

    image_attn_by_layer = step["image_attn_by_layer"]
    binary_masks = []

    for selected in kept[:top_n_heads]:
        layer_id = selected["layer"]
        head_id = selected["head"]

        image_attn = image_attn_by_layer[layer_id].detach().float().cpu()
        attn_1d = image_attn[head_id]

        attn_2d = image_attn_to_grid(
            attn_1d,
            image_grid_shape=image_grid_shape,
            inputs=inputs,
        ).numpy()

        attn_2d = (
            attn_2d - attn_2d.min()
        ) / (
            attn_2d.max() - attn_2d.min() + 1e-8
        )

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