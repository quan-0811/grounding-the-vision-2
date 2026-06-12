"""
Shared PHG dataclasses and type aliases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch


PHGDecodingMode = Literal["greedy", "dola", "vcd"]


@dataclass
class PHGConfig:
    """
    Config for PHG generation.

    Current main target:
        LLaVA-1.5-7B on COCO val2017.

    Base decoding modes:
        greedy + PHG
        dola + PHG
        vcd + PHG
    """

    decoding_mode: PHGDecodingMode = "greedy"

    # Object grounding thresholds.
    iou_thresh: float = 0.5
    ads_thresh: float = 0.45
    ads_foreground_ratio: float = 0.10

    # PHG outer loop.
    max_rounds: int = 5

    # Per-round sentence/segment generation.
    max_new_tokens: int = 64
    min_new_tokens: int = 3
    stop_at_sentence_end: bool = True

    # Uncertainty / candidate behavior.
    top_k: int = 3
    accumulate_prob: float = 0.5
    checkpoint_once: bool = True
    stop_if_sentence_end_in_candidates: bool = True

    # Token selection.
    do_sample: bool = False
    temperature: float = 1.0
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None

    # Attention / grounding.
    selected_layers: Optional[Sequence[int]] = None
    image_grid_shape: Optional[Tuple[int, int]] = (24, 24)
    keep_attn_on_cpu: bool = True

    # DoLA settings.
    dola_layers: Union[str, Sequence[int]] = "low"
    dola_relative_top: Optional[float] = 0.1
    dola_select_strategy: Literal["js", "first", "last"] = "js"

    # VCD settings.
    cd_alpha: float = 1.0
    cd_beta: float = 0.1
    noise_step: int = 500
    image_tensor_key: str = "pixel_values"

    # Optional initial generated answer prefix.
    prefix_ids: Optional[Sequence[int]] = None

    # Debug prints.
    debug: bool = False


@dataclass
class CandidateRecord:
    """
    One candidate token at an uncertainty checkpoint.
    """

    token_id: int
    token_text: str
    prob: float
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_id": int(self.token_id),
            "token_text": self.token_text,
            "prob": float(self.prob),
            "rank": int(self.rank),
        }


@dataclass
class CheckpointState:
    """
    PHG uncertainty checkpoint.

    generated_ids:
        Answer-side generated token ids before the uncertain token.

    input_ids:
        Full model input ids after appending generated_ids to the original
        multimodal prompt.

    position:
        Absolute position in answer-token space.
    """

    generated_ids: List[int]
    input_ids: Optional[torch.Tensor]
    text: str
    position: int

    def to_trace(self) -> Dict[str, Any]:
        return {
            "generated_ids": [int(x) for x in self.generated_ids],
            "text": self.text,
            "position": int(self.position),
            "has_input_ids": self.input_ids is not None,
        }


@dataclass
class SegmentScore:
    """
    Object-grounding score for one generated segment.
    """

    text: str
    accepted_objects: List[str] = field(default_factory=list)
    accepted_masks: List[np.ndarray] = field(default_factory=list)
    suspicious_objects: List[str] = field(default_factory=list)
    hallucination_score: int = 0
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_trace(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "accepted_objects": list(self.accepted_objects),
            "suspicious_objects": list(self.suspicious_objects),
            "hallucination_score": int(self.hallucination_score),
            "details": self.details,
        }


@dataclass
class CandidateBranch:
    """
    Candidate continuation and its PHG score.
    """

    candidate: CandidateRecord
    output: Dict[str, Any]
    score: SegmentScore

    @property
    def ranking_tuple(self) -> Tuple[int, int]:
        return (
            int(self.score.hallucination_score),
            int(self.candidate.rank),
        )

    def to_trace(self) -> Dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "ranking_tuple": self.ranking_tuple,
            "score": self.score.to_trace(),
            "generated_text": self.output.get("generated_text", ""),
            "full_generated_text": self.output.get("full_generated_text", ""),
            "stop_reason": self.output.get("stop_reason"),
        }


@dataclass
class PHGOutput:
    """
    Main PHG output.
    """

    final_text: str
    final_generated_ids: List[int]

    objects: List[str]
    global_objects: List[str]
    sentence_objects: List[str]

    processed_prefix_len: int

    decision_trace: List[Dict[str, Any]]
    round_outputs: List[Dict[str, Any]]

    # Not JSON-safe.
    masks: List[np.ndarray] = field(default_factory=list)
    global_masks: List[np.ndarray] = field(default_factory=list)
    sentence_masks: List[np.ndarray] = field(default_factory=list)