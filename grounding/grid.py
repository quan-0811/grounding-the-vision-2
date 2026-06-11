"""
Visual-token grid utilities.

For LLaVA-1.5:
    image_grid_shape is usually 24 x 24 because LLaVA-1.5 uses 576 image tokens.

For Qwen2-VL / InternVL2 later:
    the grid can be rectangular, so this file also supports image_grid_thw.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


def resolve_image_grid_shape(
    num_image_tokens: int,
    image_grid_shape: Optional[Tuple[int, int]] = None,
    inputs: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int]:
    """
    Resolve image attention grid shape.

    Priority:
        1. Explicit image_grid_shape=(H, W)
        2. inputs["image_grid_thw"], useful for Qwen2-VL later
        3. Square fallback, useful for LLaVA-1.5 576 -> 24x24

    Args:
        num_image_tokens:
            Number of visual tokens in the attention vector.

        image_grid_shape:
            Optional explicit grid shape.

        inputs:
            Optional prepared model inputs.

    Returns:
        (grid_h, grid_w)
    """

    if image_grid_shape is not None:
        grid_h, grid_w = image_grid_shape

        if int(grid_h) * int(grid_w) != int(num_image_tokens):
            raise ValueError(
                f"image_grid_shape={image_grid_shape} gives "
                f"{int(grid_h) * int(grid_w)} tokens, but attention has "
                f"{num_image_tokens} image tokens."
            )

        return int(grid_h), int(grid_w)

    if inputs is not None and "image_grid_thw" in inputs:
        grid_thw = inputs["image_grid_thw"]

        if torch.is_tensor(grid_thw):
            grid_thw = grid_thw.detach().cpu()

        # Most wrappers store this as [B, 3].
        if hasattr(grid_thw, "dim") and grid_thw.dim() == 2:
            t, h, w = grid_thw[0].tolist()
        else:
            t, h, w = grid_thw.tolist()

        if int(t) != 1:
            raise ValueError(
                f"Video/multi-frame grid detected: T={t}, H={h}, W={w}. "
                "Current PHG grounding expects a single image."
            )

        if int(h) * int(w) != int(num_image_tokens):
            raise ValueError(
                f"image_grid_thw gives H*W={int(h) * int(w)}, "
                f"but attention has {num_image_tokens} image tokens."
            )

        return int(h), int(w)

    side = int(num_image_tokens ** 0.5)

    if side * side == int(num_image_tokens):
        return side, side

    raise ValueError(
        f"Cannot infer image grid from {num_image_tokens} image tokens. "
        "Pass image_grid_shape=(grid_h, grid_w)."
    )


def image_attn_to_grid(
    image_attn_1d: torch.Tensor,
    image_grid_shape: Optional[Tuple[int, int]] = None,
    inputs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Reshape a 1D image-attention vector into a 2D visual-token grid.

    Args:
        image_attn_1d:
            Tensor with shape [num_image_tokens].

        image_grid_shape:
            Optional explicit grid shape.

        inputs:
            Optional model inputs containing image_grid_thw.

    Returns:
        Tensor with shape [grid_h, grid_w].
    """

    image_attn_1d = image_attn_1d.detach().float()
    num_image_tokens = int(image_attn_1d.numel())

    grid_h, grid_w = resolve_image_grid_shape(
        num_image_tokens=num_image_tokens,
        image_grid_shape=image_grid_shape,
        inputs=inputs,
    )

    return image_attn_1d.reshape(grid_h, grid_w)