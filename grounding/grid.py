"""
Visual-token grid utilities.

For LLaVA-1.5:
    image_grid_shape is usually 24 x 24 because LLaVA-1.5 uses 576 image tokens.

For Qwen2-VL:
    image_grid_thw is usually the pre-merge vision grid.
    The text-side visual token grid is often:
        H / spatial_merge_size  x  W / spatial_merge_size

Example:
    image_grid_thw = [1, 46, 30]
    raw tokens      = 46 * 30 = 1380
    merge size      = 2
    text tokens     = 23 * 15 = 345
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


def _get_qwen_spatial_merge_size(model=None) -> int:
    """
    Resolve Qwen-style spatial_merge_size.

    Qwen2-VL usually has spatial_merge_size=2 in vision_config.
    If unavailable, return 1.
    """

    if model is None:
        return 1

    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)

    for obj in [vision_config, config]:
        if obj is None:
            continue

        value = getattr(obj, "spatial_merge_size", None)

        if value is not None:
            try:
                return max(1, int(value))
            except Exception:
                pass

    return 1


def _to_thw_tuple(image_grid_thw) -> Tuple[int, int, int]:
    """
    Normalize image_grid_thw into (T, H, W).
    Supports tensor/list forms:
        [3]
        [B, 3]
    """

    grid = image_grid_thw

    if torch.is_tensor(grid):
        grid = grid.detach().cpu()

    if hasattr(grid, "dim") and grid.dim() == 2:
        t, h, w = grid[0].tolist()
    else:
        t, h, w = grid.tolist()

    return int(t), int(h), int(w)


def _infer_merged_grid_from_thw(
    num_image_tokens: int,
    t: int,
    h: int,
    w: int,
    model=None,
) -> Optional[Tuple[int, int]]:
    """
    Infer text-side visual grid from Qwen image_grid_thw.

    Priority:
        1. raw H x W match
        2. model.config.vision_config.spatial_merge_size
        3. infer merge size from token count
    """

    num_image_tokens = int(num_image_tokens)

    if t != 1:
        raise ValueError(
            f"Video/multi-frame grid detected: T={t}, H={h}, W={w}. "
            "Current PHG grounding expects a single image."
        )

    raw_tokens = int(h) * int(w)

    # LLaVA/simple case or unmerged attention.
    if raw_tokens == num_image_tokens:
        return int(h), int(w)

    # Qwen common case: text-side visual tokens are spatially merged.
    merge_size = _get_qwen_spatial_merge_size(model)

    if merge_size > 1:
        if h % merge_size == 0 and w % merge_size == 0:
            merged_h = h // merge_size
            merged_w = w // merge_size
            merged_tokens = merged_h * merged_w

            if merged_tokens == num_image_tokens:
                return int(merged_h), int(merged_w)

    # If model did not expose merge size, infer it.
    # Common Qwen merge_size is 2, but this keeps it generic.
    for candidate_merge in range(2, 9):
        if h % candidate_merge != 0 or w % candidate_merge != 0:
            continue

        merged_h = h // candidate_merge
        merged_w = w // candidate_merge
        merged_tokens = merged_h * merged_w

        if merged_tokens == num_image_tokens:
            return int(merged_h), int(merged_w)

    return None


def _factor_grid(num_image_tokens: int) -> Tuple[int, int]:
    """
    Fallback rectangular factorization.

    This is only a fallback when no image_grid_thw is available.
    It prefers a square-ish grid.
    """

    num_image_tokens = int(num_image_tokens)

    side = int(num_image_tokens ** 0.5)

    if side * side == num_image_tokens:
        return side, side

    for h_candidate in range(side, 0, -1):
        if num_image_tokens % h_candidate == 0:
            return h_candidate, num_image_tokens // h_candidate

    raise ValueError(
        f"Cannot infer image grid from {num_image_tokens} image tokens."
    )


def resolve_image_grid_shape(
    num_image_tokens: int,
    image_grid_shape: Optional[Tuple[int, int]] = None,
    inputs: Optional[Dict[str, Any]] = None,
    model=None,
) -> Tuple[int, int]:
    """
    Resolve image attention grid shape.

    Priority:
        1. Explicit image_grid_shape=(H, W)
        2. inputs["image_grid_thw"] with Qwen merge handling
        3. Square/rectangular fallback

    Args:
        num_image_tokens:
            Number of visual tokens in the attention vector.

        image_grid_shape:
            Optional explicit grid shape.

        inputs:
            Optional prepared model inputs containing image_grid_thw.

        model:
            Optional model, used to read Qwen spatial_merge_size.

    Returns:
        (grid_h, grid_w)
    """

    num_image_tokens = int(num_image_tokens)

    if num_image_tokens <= 0:
        raise ValueError(
            f"num_image_tokens must be positive, got {num_image_tokens}."
        )

    if image_grid_shape is not None:
        grid_h, grid_w = image_grid_shape
        grid_h = int(grid_h)
        grid_w = int(grid_w)

        if grid_h * grid_w != num_image_tokens:
            raise ValueError(
                f"image_grid_shape={image_grid_shape} gives "
                f"{grid_h * grid_w} tokens, but attention has "
                f"{num_image_tokens} image tokens."
            )

        return grid_h, grid_w

    if inputs is not None and "image_grid_thw" in inputs:
        t, h, w = _to_thw_tuple(inputs["image_grid_thw"])

        inferred = _infer_merged_grid_from_thw(
            num_image_tokens=num_image_tokens,
            t=t,
            h=h,
            w=w,
            model=model,
        )

        if inferred is not None:
            return inferred

        raw_tokens = int(t) * int(h) * int(w)
        merge_size = _get_qwen_spatial_merge_size(model)

        merged_info = "unknown"

        if merge_size > 1 and h % merge_size == 0 and w % merge_size == 0:
            merged_info = str(
                int(t) * (int(h) // merge_size) * (int(w) // merge_size)
            )

        # Do not silently accept num_image_tokens=1 here.
        # That usually means cached-step attention was used instead of
        # full-prefix attention.
        raise ValueError(
            f"image_grid_thw gives raw H*W={raw_tokens}, "
            f"merged tokens={merged_info}, "
            f"but attention has {num_image_tokens} image tokens. "
            "If attention has 1 token for Qwen2-VL, patch PHG to recompute "
            "full-prefix attention with use_cache=False before grounding."
        )

    return _factor_grid(num_image_tokens)


def image_attn_to_grid(
    image_attn_1d: torch.Tensor,
    image_grid_shape: Optional[Tuple[int, int]] = None,
    inputs: Optional[Dict[str, Any]] = None,
    model=None,
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

        model:
            Optional model for Qwen spatial_merge_size.

    Returns:
        Tensor with shape [grid_h, grid_w].
    """

    image_attn_1d = image_attn_1d.detach().float()
    num_image_tokens = int(image_attn_1d.numel())

    grid_h, grid_w = resolve_image_grid_shape(
        num_image_tokens=num_image_tokens,
        image_grid_shape=image_grid_shape,
        inputs=inputs,
        model=model,
    )

    return image_attn_1d.reshape(grid_h, grid_w)