"""
Object-level PHG scoring.

This file scores generated text segments using:

    - noun extraction
    - noun-token alignment
    - attention mask extraction
    - ADS compactness
    - IoU compatibility against PHG memory
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from models.base import TensorDict
from grounding import (
    OUTLIER_NOUNS,
    compute_ads_from_step,
    detect_nouns,
    find_noun_token_start,
    get_object_mask_from_step,
    mask_is_compatible,
)

from phg.memory import PHGMemory
from phg.types import PHGConfig, SegmentScore


def score_segment(
    tokenizer: Any,
    segment_token_ids: List[int],
    segment_steps: List[Dict[str, Any]],
    memory: PHGMemory,
    config: PHGConfig,
    inputs: Optional[TensorDict] = None,
) -> SegmentScore:
    """
    Score one generated segment.

    Returns:
        SegmentScore containing accepted objects, accepted masks, suspicious
        objects, and hallucination_score.
    """

    segment_text = tokenizer.decode(
        segment_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    if len(segment_token_ids) == 0 or len(segment_steps) == 0:
        return SegmentScore(text=segment_text)

    try:
        nouns = set(detect_nouns(segment_text))
    except LookupError as exc:
        raise LookupError(
            "NLTK data is missing. Run:\n"
            "import nltk\n"
            "nltk.download('punkt')\n"
            "nltk.download('averaged_perceptron_tagger')\n"
            "For newer NLTK versions, also run:\n"
            "nltk.download('punkt_tab')\n"
            "nltk.download('averaged_perceptron_tagger_eng')"
        ) from exc

    known_objects = set(memory.known_objects)
    base_masks = memory.compatibility_masks

    score = SegmentScore(text=segment_text)

    for noun in nouns:
        noun_norm = noun.lower().strip()

        if noun_norm in OUTLIER_NOUNS:
            continue

        if noun_norm in known_objects:
            continue

        noun_start = find_noun_token_start(
            tokenizer,
            segment_token_ids,
            noun_norm,
        )

        if noun_start is None or noun_start >= len(segment_steps):
            continue

        noun_step = segment_steps[noun_start]

        noun_mask = get_object_mask_from_step(
            noun_step,
            image_grid_shape=config.image_grid_shape,
            inputs=inputs,
            top_n_heads=5,
            attn_sum_threshold=0.49,
        )

        ads = compute_ads_from_step(
            noun_step,
            image_grid_shape=config.image_grid_shape,
            inputs=inputs,
            foreground_ratio=config.ads_foreground_ratio,
            top_n_heads=3,
            attn_sum_threshold=0.49,
        )

        has_valid_mask = (
            noun_mask is not None
            and int(noun_mask.astype(bool).sum()) > 0
        )

        compact_enough = bool(
            np.isfinite(ads)
            and ads <= config.ads_thresh
        )

        compatible = mask_is_compatible(
            noun_mask,
            base_masks + score.accepted_masks,
            iou_thresh=config.iou_thresh,
        )

        accepted = bool(
            has_valid_mask
            and compact_enough
            and compatible
        )

        detail = {
            "noun": noun_norm,
            "token_start": int(noun_start),
            "ads": float(ads),
            "has_valid_mask": bool(has_valid_mask),
            "compact_enough": bool(compact_enough),
            "iou_compatible": bool(compatible),
            "accepted": bool(accepted),
        }

        score.details.append(detail)

        if accepted:
            score.accepted_objects.append(noun_norm)
            score.accepted_masks.append(noun_mask)
            known_objects.add(noun_norm)
        else:
            score.suspicious_objects.append(noun_norm)
            score.hallucination_score += 1

    return score