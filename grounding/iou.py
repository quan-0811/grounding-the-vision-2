"""
Mask IoU utilities for PHG object-memory compatibility.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute IoU between two binary masks.

    Args:
        a:
            First mask.

        b:
            Second mask.

    Returns:
        IoU in [0, 1].
    """

    a = a.astype(bool)
    b = b.astype(bool)

    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    return float(intersection) / float(union + 1e-8)


def mask_is_compatible(
    new_mask: Optional[np.ndarray],
    masks_list: List[np.ndarray],
    iou_thresh: float = 0.5,
) -> bool:
    """
    Check whether a new object mask is compatible with existing memory masks.

    PHG uses this to avoid accepting repeated / conflicting object regions.

    Returns:
        True if new_mask is valid and does not overlap too much with memory.
    """

    if new_mask is None:
        return False

    new_mask = new_mask.astype(bool)

    if int(new_mask.sum()) == 0:
        return False

    for old_mask in masks_list:
        if old_mask is None:
            continue

        iou = compute_iou(new_mask, old_mask)

        if iou > iou_thresh:
            return False

    return True