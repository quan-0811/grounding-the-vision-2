"""
PHG object-memory management.

PHG uses two memories:

    M_g: global accepted objects/masks across completed sentences.
    M_s: current-sentence accepted objects/masks.

At a sentence boundary:
    M_g <- M_g union M_s
    M_s <- empty
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from phg.types import SegmentScore


@dataclass
class PHGMemory:
    """
    Object memory for PHG.
    """

    global_objects: List[str] = field(default_factory=list)
    global_masks: List[np.ndarray] = field(default_factory=list)

    sentence_objects: List[str] = field(default_factory=list)
    sentence_masks: List[np.ndarray] = field(default_factory=list)

    processed_prefix_len: int = 0

    @property
    def known_objects(self) -> List[str]:
        return list(dict.fromkeys(self.global_objects + self.sentence_objects))

    @property
    def compatibility_masks(self) -> List[np.ndarray]:
        return list(self.global_masks) + list(self.sentence_masks)

    def set_processed_prefix_len(self, value: int) -> None:
        self.processed_prefix_len = int(value)

    def add_to_sentence(
        self,
        objects: List[str],
        masks: List[np.ndarray],
    ) -> None:
        for obj, mask in zip(objects, masks):
            if obj not in self.known_objects:
                self.sentence_objects.append(obj)
                self.sentence_masks.append(mask)

    def add_score_to_sentence(self, score: SegmentScore) -> None:
        self.add_to_sentence(
            objects=score.accepted_objects,
            masks=score.accepted_masks,
        )

    def commit_sentence(self) -> None:
        """
        Commit M_s into M_g, then reset M_s.
        """

        for obj, mask in zip(self.sentence_objects, self.sentence_masks):
            if obj not in self.global_objects:
                self.global_objects.append(obj)
                self.global_masks.append(mask)

        self.reset_sentence()

    def reset_sentence(self) -> None:
        self.sentence_objects = []
        self.sentence_masks = []

    def to_trace(self) -> Dict[str, Any]:
        return {
            "global_objects": list(self.global_objects),
            "sentence_objects": list(self.sentence_objects),
            "known_objects": self.known_objects,
            "num_global_masks": len(self.global_masks),
            "num_sentence_masks": len(self.sentence_masks),
            "processed_prefix_len": int(self.processed_prefix_len),
        }