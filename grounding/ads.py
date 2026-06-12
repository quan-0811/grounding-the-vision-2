"""
Attention dispersion metrics.

ADS is used by PHG as a compactness proxy.

Lower ADS:
    attention is concentrated / compact

Higher ADS:
    attention is diffuse / spread across the image
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

try:
    from scipy.ndimage import label
except ImportError as exc:
    raise ImportError(
        "grounding.ads requires scipy. Install with: pip install scipy"
    ) from exc

from grounding.grid import image_attn_to_grid


def spatial_entropy(
    attn_map_2d: torch.Tensor,
    threshold: float = 1e-3,
) -> Dict[str, Any]:
    """
    Compute component-level spatial entropy for a 2D attention map.

    This is used to select compact layer-head maps.

    Args:
        attn_map_2d:
            Tensor with shape [H, W].

        threshold:
            Threshold over activated attention.

    Returns:
        {
            "spatial_entropy": float,
            "labeled_array": np.ndarray,
            "num_components": int,
        }
    """

    attn_map_2d = attn_map_2d.detach().float().cpu()

    normalized = (
        attn_map_2d - attn_map_2d.min()
    ) / (
        attn_map_2d.max() - attn_map_2d.min() + 1e-8
    )

    mean_val = torch.mean(normalized)
    activated = torch.relu(normalized - mean_val * 2.0)

    activated_np = activated.numpy()
    binary = (activated_np > threshold).astype(np.int32)

    labeled_array, num_components = label(
        binary,
        structure=np.ones((3, 3), dtype=np.int32),
    )

    total = float(activated.sum().item())

    if total <= 0:
        return {
            "spatial_entropy": float("inf"),
            "labeled_array": labeled_array,
            "num_components": 0,
        }

    probs = []

    for component_id in range(1, num_components + 1):
        component_sum = activated_np[labeled_array == component_id].sum()

        if component_sum > 0:
            probs.append(component_sum / total)

    if len(probs) == 0:
        entropy = 0.0
    else:
        entropy = -sum(p * np.log(p) for p in probs if p > 0)

    return {
        "spatial_entropy": float(entropy),
        "labeled_array": labeled_array,
        "num_components": int(num_components),
    }


def compute_ads_from_attention_map(
    attn_map_2d: torch.Tensor,
    foreground_ratio: float = 0.10,
    eps: float = 1e-8,
) -> float:
    """
    Compute ADS-style attention dispersion score.

    ADS(A) = (1 - m_foreground) * H_background

    Args:
        attn_map_2d:
            Tensor with shape [H, W].

        foreground_ratio:
            Top-ratio attention mass treated as foreground.

    Returns:
        ADS score. Lower is more compact.
    """

    attn_map_2d = attn_map_2d.detach().float().cpu()
    attn_map_2d = torch.clamp(attn_map_2d, min=0)

    flat = attn_map_2d.flatten()
    total = flat.sum()

    if float(total.item()) <= eps:
        return float("inf")

    probs = flat / (total + eps)

    num_tokens = probs.numel()
    top_k = max(1, int(np.ceil(foreground_ratio * num_tokens)))

    _, top_indices = torch.topk(probs, k=top_k)

    foreground_mask = torch.zeros_like(probs, dtype=torch.bool)
    foreground_mask[top_indices] = True

    background_mask = ~foreground_mask

    foreground_mass = float(probs[foreground_mask].sum().item())

    background_probs = probs[background_mask]

    if background_probs.numel() == 0 or float(background_probs.sum().item()) <= eps:
        background_entropy = 0.0
    else:
        background_probs = background_probs / (background_probs.sum() + eps)
        entropy = -torch.sum(background_probs * torch.log(background_probs + eps))
        background_entropy = float(entropy.item()) / float(np.log(max(num_tokens, 2)))

    ads = (1.0 - foreground_mass) * background_entropy

    return float(ads)


def compute_ads_from_step(
    step: Dict[str, Any],
    image_grid_shape: Optional[Tuple[int, int]] = None,
    inputs: Optional[Dict[str, Any]] = None,
    foreground_ratio: float = 0.10,
    top_n_heads: int = 3,
    attn_sum_threshold: float = 0.49,
) -> float:
    """
    Compute ADS from selected layer-head maps in one PHG decoding step.

    The step must contain:
        step["image_attn_by_layer"][layer_id] = Tensor[num_heads, num_image_tokens]

    Returns:
        Minimum ADS among selected layer-heads.
    """

    from grounding.attention import get_kept_lh_from_step

    kept = get_kept_lh_from_step(
        step=step,
        image_grid_shape=image_grid_shape,
        inputs=inputs,
        attn_sum_threshold=attn_sum_threshold,
    )

    if len(kept) == 0:
        return float("inf")

    image_attn_by_layer = step["image_attn_by_layer"]
    ads_values = []

    for selected in kept[:top_n_heads]:
        layer_id = selected["layer"]
        head_id = selected["head"]

        image_attn = image_attn_by_layer[layer_id].detach().float().cpu()
        attn_1d = image_attn[head_id]

        attn_2d = image_attn_to_grid(
            attn_1d,
            image_grid_shape=image_grid_shape,
            inputs=inputs,
        )

        ads = compute_ads_from_attention_map(
            attn_2d,
            foreground_ratio=foreground_ratio,
        )

        ads_values.append(ads)

    if len(ads_values) == 0:
        return float("inf")

    return float(min(ads_values))