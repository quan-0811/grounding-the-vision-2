"""
Grounding utilities for PHG.

This package contains model-agnostic utilities for:

    - resolving visual-token grids
    - extracting image attention from generated-token attentions
    - selecting useful layer-head maps
    - computing ADS / spatial entropy
    - building object masks
    - computing IoU compatibility
    - extracting nouns from generated text

The expected PHG step format is:

    step = {
        "step": int,
        "token_id": int,
        "token_text": str,
        "image_attn_by_layer": {
            layer_id: Tensor[num_heads, num_image_tokens]
        }
    }

This format is produced later by decoding/stepwise.py.
"""

from grounding.ads import (
    compute_ads_from_attention_map,
    compute_ads_from_step,
    spatial_entropy,
)

from grounding.attention import (
    extract_image_attn_by_layer,
    get_image_token_id,
    get_kept_lh_from_step,
    resolve_image_token_indices,
)

from grounding.grid import (
    image_attn_to_grid,
    resolve_image_grid_shape,
)

from grounding.iou import (
    compute_iou,
    mask_is_compatible,
)

from grounding.masks import (
    get_object_mask_from_step,
    remove_singletons,
)

from grounding.noun_extraction import (
    OUTLIER_NOUNS,
    detect_nouns,
    find_noun_token_start,
    find_sublist_start,
)